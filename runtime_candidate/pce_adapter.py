"""Bind explicitly documented own-unit monthly indices before the fixed PCE core.

Asset names never select a source. No network, stored coefficient, outside-unit
history, release-calendar guess, or future target value is consulted.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import jsonschema
from . import pce_core

class Ineligible(ValueError):
    pass

MAP = {"PCE_ALL": "PCEPI", "PCE_CORE": "PCEPILFE", "CPI_ALL": "CPIAUCSL", "CPI_CORE": "CPILFESL"}

def timestamp(value):
    parsed = dt.datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise Ineligible('timestamp requires an explicit UTC offset')
    return parsed.astimezone(dt.timezone.utc)

def owned_file(root: Path, relative: str):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or path.is_symlink():
        raise Ineligible('source/evidence file must be an ordinary own-unit file')
    return path

def verify_hash(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise Ineligible('own-unit source/evidence SHA256 mismatch')

def spec_cells(spec, assets, horizons):
    cells = {}
    target = spec.get('targets', {})
    periods = target.get('observation_periods')
    dates = target.get('forecast_dates')
    if isinstance(periods, list) and isinstance(dates, list) and len(periods) == len(dates) == len(horizons):
        for asset in assets:
            for h, period, date in zip(horizons, periods, dates):
                cells[(asset,h)] = (period,date,target.get('statistic'),target.get('value_unit'))
    for row in spec.get('questions', []):
        if isinstance(row, dict) and all(k in row for k in ('asset','horizon','observation_period','forecast_date','statistic','value_unit')):
            key = (row['asset'],row['horizon'])
            value = (row['observation_period'],row['forecast_date'],row['statistic'],row['value_unit'])
            if key in cells and cells[key] != value:
                raise Ineligible('conflicting target specification')
            cells[key] = value
    if set(cells) != {(a,h) for a in assets for h in horizons}:
        raise Ineligible('exact observation-period/statistic/unit/forecast-date grid absent in forecast_spec')
    return cells

def bind(panels, panel_paths, card, spec, unit_root, asof):
    metadata = spec.get('pce_conditional_input')
    if metadata is None:
        raise Ineligible('explicit pce_conditional_input metadata absent')
    schema = json.loads(Path(__file__).with_name('pce_input.schema.json').read_text(encoding='utf-8'))
    try:
        jsonschema.validate(metadata, schema, format_checker=jsonschema.FormatChecker())
    except jsonschema.ValidationError as exc:
        raise Ineligible('PCE metadata schema invalid at '+ '/'.join(str(p) for p in exc.absolute_path)) from None
    cutoff = timestamp(metadata['asof_utc'])
    if cutoff.date().isoformat() != asof:
        raise Ineligible('PCE cutoff date disagrees with CLI/card cutoff')
    targets = card['targets']
    assets, horizons = list(targets['asset_ids']), list(targets['horizons'])
    if targets.get('target_type','level') != 'level':
        raise Ineligible('PCE adapter supports level-valued index/MoM statistics only')
    cells = spec_cells(spec, assets, horizons)
    declared = {(r['asset'],r['horizon']): r for r in metadata['targets']}
    if len(declared) != len(metadata['targets']) or set(declared) != set(cells):
        raise Ineligible('PCE metadata target grid mismatch/duplicate')
    periods, roles = set(), set()
    for key,row in declared.items():
        value = (row['observation_period'],row['forecast_date'],row['statistic'],row['output_value_unit'])
        if value != cells[key]:
            raise Ineligible('PCE target metadata disagrees with own forecast_spec')
        if row['output_value_unit'] != targets.get('value_unit'):
            raise Ineligible('PCE target unit disagrees with card')
        if row['statistic'] == 'index_level' and row['output_value_unit'] != 'Index 2017=100':
            raise Ineligible('index-level output requires explicit Index 2017=100 native units')
        if row['statistic'] == 'mom_percent' and row['output_value_unit'] != 'Percent change from preceding month':
            raise Ineligible('MoM output requires explicit percent-change units; YoY unsupported')
        release = timestamp(row['scheduled_first_release_at'])
        if release <= cutoff or release.date().isoformat() != row['forecast_date']:
            raise Ineligible('PCE scheduled first release must be future and match forecast date')
        periods.add(row['observation_period']); roles.add(row['role'])
    if len(periods) != 1:
        raise Ineligible('fixed PCE core supports exactly one observation month')
    # Multiple authored horizon keys for the same released quantity would duplicate draws;
    # keep the validated one-target-month model to its research-supported single horizon shape.
    if len(horizons) != 1 or len(roles) != len(assets):
        raise Ineligible('PCE core supports one authored horizon and unique PCE role per asset')
    target = pd.Period(next(iter(periods)),freq='M')
    sources = {row['role']: row for row in metadata['sources']}
    if len(sources) != 4 or set(sources) != set(MAP):
        raise Ineligible('exact four PCE/CPI source roles required')
    series = {}
    for role,row in sources.items():
        if row['fred_series_id'] != MAP[role]:
            raise Ineligible('explicit FRED ID disagrees with fixed role mapping')
        if row['originator'] != ('BEA' if role.startswith('PCE') else 'BLS'):
            raise Ineligible('source originator mismatch')
        if row['units'] != ('Index 2017=100' if role.startswith('PCE') else 'Index 1982-1984=100'):
            raise Ineligible('source native index base mismatch')
        if timestamp(row['vintage_asof_utc']) > cutoff:
            raise Ineligible('source vintage is newer than cutoff')
        evidence = owned_file(unit_root,row['evidence_file'])
        verify_hash(evidence,row['evidence_sha256'])
        path = panel_paths.get(row['panel_file'])
        if path is None or not path.resolve().is_relative_to(unit_root.resolve()):
            raise Ineligible('source panel must be the selected own-unit panel')
        verify_hash(path,row['panel_sha256'])
        df = panels[row['panel_file']]
        names = [row[k] for k in ('asset_column','observation_period_column','value_column','available_at_column')]
        if any(name not in df.columns for name in names):
            raise Ineligible('source observation-period/value/availability columns absent')
        selected = df.loc[df[row['asset_column']].astype(str) == row['asset_id']].copy()
        raw_periods = selected[row['observation_period_column']].astype(str)
        if not raw_periods.str.fullmatch(r'\d{4}-\d{2}').all():
            raise Ineligible('source observation periods need explicit YYYY-MM labels')
        period_index = pd.PeriodIndex(raw_periods,freq='M')
        if period_index.has_duplicates:
            raise Ineligible('source monthly periods duplicated')
        # Date and clock filtering happen before pd.to_numeric in unchanged pure core.
        available = selected[row['available_at_column']].map(timestamp)
        if role.startswith('PCE') and np.any((period_index == target) & np.asarray(available <= cutoff)):
            raise Ineligible('target PCE index is already available at cutoff')
        upper = target-1 if role.startswith('PCE') else target
        take = (period_index <= upper) & np.asarray(available <= cutoff)
        series[role] = pd.Series(selected.loc[take,row['value_column']].to_numpy(), index=period_index[take]).sort_index()
    pce = pd.concat([series[r].rename(r) for r in ('PCE_ALL','PCE_CORE')],axis=1).sort_index()
    cpi = pd.concat([series[r].rename(r) for r in ('CPI_ALL','CPI_CORE')],axis=1).sort_index()
    clock = {'production_gate_verified': True,'target_observation_period':str(target),'asof_utc':cutoff.isoformat()}
    return pce,cpi,str(target),clock,declared

def try_forecast(panels, panel_paths, card, spec, unit_root, asof):
    try:
        pce,cpi,target,clock,cells = bind(panels,panel_paths,card,spec,unit_root,asof)
        predictions,ledger = pce_core.forecast(pce,cpi,target,clock)
        if not all(ledger['privacy']['current_cpi_log_change_available'].values()):
            raise Ineligible('required current CPI feature missing or masked; production uses canonical M0')
        native = predictions['conditional_CPI_distribution']
        assets,horizons = list(card['targets']['asset_ids']),list(card['targets']['horizons'])
        result = np.empty((1000,len(assets),len(horizons)))
        for ai,asset in enumerate(assets):
            for hi,h in enumerate(horizons):
                row=cells[(asset,h)]; ri=pce_core.PCE.index(row['role'])
                draw=native[:,ri]
                if row['statistic']=='mom_percent':
                    anchor=float(pce.iloc[-1,ri]); draw=100*(draw/anchor-1)
                result[:,ai,hi]=draw
        if result.shape != (1000,len(assets),len(horizons)) or not np.isfinite(result).all():
            raise Ineligible('PCE finite-sample/output-shape guard failed')
        return result,{'applied':True,'model':'fixed conditional_CPI_distribution','reason':'explicit own-unit metadata and histories passed guards','model_ledger':ledger,
                       'timestamp_evidence_independently_audited':False,'outer_vintage':'supplied own-unit vintage <= cutoff','inner_oof':'retrospective supplied vintage, not fully PIT'}
    except (Ineligible,ValueError,KeyError,TypeError,np.linalg.LinAlgError) as exc:
        return None,{'applied':False,'model':'canonical M0 fallback','reason':str(exc),'failure_type':type(exc).__name__}
