"""Interface 2.0 offline CLI: own-unit canonical M0, with explicit PCE eligibility."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import tomllib
import numpy as np
import pandas as pd
from .canonical_m0 import forecast as m0_forecast
from .horizons import monthly_horizon_steps
from .pce_adapter import try_forecast

def read_panels(directory):
    found = sorted(directory.glob('*.parquet'))
    if not found:
        found = sorted(directory.parent.glob('*.parquet'))
    if not found:
        raise ValueError('no own-unit parquet panel found')
    return {p.name:pd.read_parquet(p) for p in found},{p.name:p for p in found}

def histories_from_panels(panels,assets,asof):
    histories = {}
    for asset in assets:
        for name,df in panels.items():
            col = next((c for c in ('asset','asset_id') if c in df.columns),None)
            if col is None or not {'date','value'} <= set(df.columns):
                continue
            selected=df.loc[df[col].astype(str)==asset].copy()
            if selected.empty:
                continue
            dates=pd.to_datetime(selected['date'],errors='raise').dt.strftime('%Y-%m-%d')
            selected=selected.loc[dates<=asof].copy()
            selected['_date']=dates.loc[selected.index]
            selected=selected.sort_values('_date')
            if selected.empty:
                continue
            # No future-row numeric conversion. Values used by M0 are the own target asset only.
            histories[asset]={'dates':selected['_date'].to_numpy(dtype='datetime64[D]'),
                              'values':pd.to_numeric(selected['value'],errors='raise').to_numpy(dtype=float)}
            break
        if asset not in histories:
            raise ValueError('target asset unavailable in own-unit cutoff prefix: '+asset)
    return histories

def sampling_steps(histories,horizons,asof,card,spec):
    assets=list(histories)
    counts=np.tile(np.asarray(horizons,dtype=float),(len(assets),1))
    # Explicit monthly observation labels use anchor-to-period counts, never release dates.
    monthly=all(len(h['dates'])>=3 and np.median(np.diff(h['dates']).astype('timedelta64[D]').astype(float))>20 for h in histories.values())
    explicit=any('observation_periods' in source.get('targets',{}) or any('observation_period' in r for r in source.get('questions',[]) if isinstance(r,dict)) for source in (card,spec))
    if monthly and explicit:
        last={a:str(h['dates'][-1]) for a,h in histories.items()}
        return monthly_horizon_steps(assets,horizons,last,asof=asof,card=card,forecast_spec=spec), 'explicit monthly observation periods'
    # Published M0 section 3.7: absent/malformed target dates retain authored horizon keys.
    target_dates={}
    for source in (card,spec):
        tgt=source.get('targets',{})
        values=tgt.get('target_dates')
        hs=tgt.get('horizons',horizons); aa=tgt.get('asset_ids',assets)
        if isinstance(values,list) and len(values)==len(hs):
            for a in aa:
                for h,date in zip(hs,values):
                    if (a,h) in target_dates and target_dates[(a,h)]!=date:
                        raise ValueError('conflicting public target-date metadata')
                    target_dates[(a,h)]=date
        for row in source.get('questions',[]):
            if isinstance(row,dict) and all(k in row for k in ('asset','horizon','target_date')):
                key=(row['asset'],row['horizon']); date=row['target_date']
                if key in target_dates and target_dates[key]!=date:
                    raise ValueError('conflicting public cell target dates')
                target_dates[key]=date
    for ai,asset in enumerate(assets):
        dates=np.asarray(histories[asset]['dates'],dtype='datetime64[D]')[-300:]
        if len(dates)<3:
            continue
        gaps=np.diff(dates).astype('timedelta64[D]').astype(float)
        valid=gaps<=max(10*float(np.median(gaps)),5.)
        if not np.any(valid) or np.any(gaps<=0):
            continue
        spacing=float(gaps[valid].mean()); anchor=dt.date.fromisoformat(str(dates[-1]))
        for hi,h in enumerate(horizons):
            value=target_dates.get((asset,h))
            try:
                target=dt.date.fromisoformat(value) if isinstance(value,str) else None
            except ValueError:
                target=None
            if target is None:
                continue
            steps=12*(target.year-anchor.year)+target.month-anchor.month if spacing>20 else round((target-anchor).days/spacing)
            if steps>0 and max(steps,h)/max(min(steps,h),1)>=2:
                counts[ai,hi]=steps
    return counts,'published M0 public-date conversion; absent dates retain authored horizon'

def finite_guard(samples,assets,horizons):
    if samples.shape != (1000,len(assets),len(horizons)) or not np.isfinite(samples).all():
        raise ValueError('finite 1000-draw exact-grid guard failed')

def main(argv=None):
    ap=argparse.ArgumentParser(prog='forecast',description='Offline canonical M0 and strictly eligible fixed PCE model')
    ap.add_argument('--panels',type=Path,required=True)
    ap.add_argument('--text',type=Path,required=True)
    ap.add_argument('--asof',required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--card',type=Path)
    ap.add_argument('--baseline-only',action='store_true',help='diagnostic: force canonical M0')
    args=ap.parse_args(argv)
    dt.date.fromisoformat(args.asof)
    card_path=args.card
    if card_path is None:
        card_path=next((p for p in (args.panels/'card.toml',args.panels.parent/'card.toml') if p.is_file()),None)
    if card_path is None:
        raise ValueError('card.toml absent from own input unit')
    unit_root=card_path.parent.resolve()
    if not args.panels.resolve().is_relative_to(unit_root):
        raise ValueError('panels path must belong to the same input unit')
    card=tomllib.loads(card_path.read_text(encoding='utf-8'))
    if card['provenance']['data_cutoff']!=args.asof:
        raise ValueError('CLI asof disagrees with card data_cutoff')
    assets=list(card['targets']['asset_ids']); horizons=list(card['targets']['horizons'])
    if not assets or len(set(assets))!=len(assets) or not horizons or any(type(h) is not int or h<=0 for h in horizons) or len(set(horizons))!=len(horizons):
        raise ValueError('invalid authored target grid')
    if int(card.get('scoring',{}).get('params',{}).get('n_draws_min',200))>1000:
        raise ValueError('card draw floor exceeds the fixed 1000-draw contract')
    target_type=card['targets'].get('target_type','level')
    if target_type not in ('level','log_return','yield'):
        raise ValueError('unsupported canonical target type')
    spec_path=unit_root/'forecast_spec.json'
    spec=json.loads(spec_path.read_text(encoding='utf-8')) if spec_path.is_file() else {}
    panels,paths=read_panels(args.panels)
    histories=histories_from_panels(panels,assets,args.asof)
    steps,step_reason=sampling_steps(histories,horizons,args.asof,card,spec)
    # Always prepare unchanged canonical fallback before considering the conditional branch.
    samples,baseline=m0_forecast(histories,horizons,target_type=target_type,unit_id=card['task']['id'],n_draws=1000,steps=steps)
    if args.baseline_only:
        eligibility={'applied':False,'model':'canonical M0 fallback','reason':'baseline-only diagnostic mode'}
    else:
        conditional,eligibility=try_forecast(panels,paths,card,spec,unit_root,args.asof)
        if conditional is not None:
            samples=conditional
    finite_guard(samples,assets,horizons)
    out=args.out; out.parent.mkdir(parents=True,exist_ok=True)
    values=samples.reshape(-1)
    frame=pd.DataFrame({'draw':np.repeat(np.arange(1000),len(assets)*len(horizons)),
                        'asset':np.tile(np.repeat(assets,len(horizons)),1000),
                        'horizon':np.tile(horizons,1000*len(assets)), 'value':values})
    if len(frame)!=1000*len(assets)*len(horizons) or frame.duplicated(['draw','asset','horizon']).any():
        raise ValueError('exact output-grid guard failed')
    frame.to_parquet(out,index=False)
    meta={'unit_id':card['task']['id'],'asof':args.asof,'representation':'samples',
          'asset_ids':assets,'horizons':horizons,'n_draws':1000,'target':target_type,
          'rationale':{'file':'forecast_rationale.md','method':eligibility['model']}}
    (out.parent/'forecast_meta.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    ledger={'unit_id':card['task']['id'],'asof':args.asof,'own_input_only':True,'network_used':False,
            'pce_eligibility':eligibility,'canonical_M0':baseline,'step_binding':step_reason,
            'finite_draws_guard_passed':True,'no_clipping_or_rounding':True}
    reason=eligibility['reason']
    rationale=f'''# Forecast rationale — {card['task']['id']}

At cutoff {args.asof}, this forecast contains 1000 finite joint draws over the authored
assets {assets} and horizon keys {horizons}. All fitted quantities use this unit's
supplied cutoff prefix. No outside data or text body was read; no network was used.

Selected model: {eligibility['model']}. Eligibility decision: {reason}.

The canonical fallback is the published trailing-300 M0 Gaussian walk, with own-prefix
mean increments, aligned sample covariance and shared min(step_i, step_j) path covariance.
Horizon binding: {step_reason}. No mean/width heuristics, clipping, or quantization are used.

The optional PCE branch retains the fixed ridge20/120-row/min60/OOF60 core and seed1729.
It requires all four explicit native PCE/CPI index series, SA and provider metadata,
own-unit source/evidence hashes, row availability timestamps, exact target statistic,
observation month and future release date. Only one observation month and one authored
horizon are supported. MoM is 100*(index draw / preceding own PCE index - 1).
YoY and unverified inputs use the canonical fallback with a recorded reason.

PCE source timestamp declarations are bound to own-unit files but are not independently
audited real-time vintages. Inner OOF uses the supplied historical vintage retrospectively.
This output makes no Final performance or first-place guarantee.

## Own-input decision ledger

```json
{json.dumps(ledger,indent=2,allow_nan=False)}
```
'''
    (out.parent/'forecast_rationale.md').write_text(rationale,encoding='utf-8')
    print(json.dumps({'out':str(out),'draws':1000,'cells':len(assets)*len(horizons),'pce_applied':eligibility['applied'],'reason':reason}))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
