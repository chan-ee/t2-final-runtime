"""Rebuild M0 and its published input-only expected-error divisors.

This follows M0-BASELINE.md sections3/5 at T2 commit60509df. Each fitted mean
and covariance uses only the current supplied histories. Divisors require no
outcome, seed, stored scale, sibling panel or private-test material. Scoring
continues to use the imported official component functions.
"""
import zlib

import numpy as np
from scipy.special import gamma, hyp1f1, ndtr, ndtri
def _step_series(history, target_type):
    values=np.asarray(history['values'],dtype=np.float64)[-300:]
    dates=np.asarray(history['dates'],dtype='datetime64[D]')[-300:]
    if len(values)<3 or not np.isfinite(values).all():
        raise ValueError('M0 needs finite historical rows')
    gap=np.diff(dates).astype('timedelta64[D]').astype(np.int64)
    if np.any(gap<=0):raise ValueError('M0 needs unique increasing dates')
    valid=gap<=max(10*float(np.median(gap)),5.0)
    if target_type=='log_return':
        if np.median(np.abs(values))>=0.2:
            raise ValueError('Return panel failed M0 step-magnitude tripwire')
        changes=values[1:]
        anchor=0.0
    else:
        changes=np.diff(values)
        anchor=float(values[-1])
    return dates[1:][valid],changes[valid],anchor

LEVELS = (.01, .05, .95, .99)


def distribution(histories, horizons, *, target_type, steps=None):
    assets = list(histories)
    hs = list(horizons)
    series = {a: _step_series(histories[a], target_type) for a in assets}
    common = None
    for dates, _, _ in series.values():
        common = dates if common is None else np.intersect1d(common, dates, assume_unique=True)
    if common is None or len(common) < 2:
        raise ValueError('Insufficient aligned current-unit steps')
    matrix = np.column_stack([values[np.searchsorted(dates, common)] for dates, values, _ in series.values()])
    mu = matrix.mean(axis=0)
    covariance = np.atleast_2d(np.cov(matrix.T, ddof=1))
    anchor = np.asarray([series[a][2] for a in assets])
    count_grid = np.asarray(steps, dtype=float) if steps is not None else np.tile(np.asarray(hs,dtype=float),(len(assets),1))
    if count_grid.shape != (len(assets),len(hs)) or not np.isfinite(count_grid).all() or np.any(count_grid<=0):
        raise ValueError('Invalid per-cell sampling step counts')
    counts = count_grid.reshape(-1)
    ai = np.repeat(np.arange(len(assets)), len(hs))
    mean = anchor[ai] + counts*mu[ai]
    joint = np.minimum(counts[:, None], counts[None, :])*covariance[ai[:, None], ai[None, :]]
    joint[np.diag_indices(len(mean))] += 1e-10
    joint += 1e-9*np.eye(len(mean))
    fallback = False
    try:
        factor = np.linalg.cholesky(joint)
    except np.linalg.LinAlgError:
        joint = np.diag(np.diag(joint))
        factor = np.linalg.cholesky(joint)
        fallback = True
    if not np.isfinite(mean).all() or not np.isfinite(joint).all():
        raise ValueError('Nonfinite fitted distribution')
    return mean, joint, factor, {
        'model': 'published current-unit M0 joint Gaussian walk', 'assets': assets, 'horizons': hs,
        'cell_order': 'authored assets, then authored horizons', 'step_counts': counts.astype(int).tolist(),
        'aligned_history_steps': len(common), 'step_mean_authored_assets': mu.tolist(),
        'step_covariance_authored_assets': covariance.tolist(), 'diagonal_fallback': fallback,
        'training_scope': 'current supplied unit only', 'learned_at_runtime': True}


def expected_scales(mean, covariance, *, levels=LEVELS):
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (len(mean), len(mean)):
        raise ValueError('Mean/covariance shape mismatch')
    sd = np.sqrt(np.diag(covariance))
    marginal = float(sd.mean()/np.sqrt(np.pi))
    z = ndtri(np.asarray(levels))
    tail = float(np.mean(np.exp(-.5*z*z)/np.sqrt(2*np.pi))*sd.mean())
    joint = 0.
    minimum_pair_variance = None
    for i in range(len(mean)):
        for j in range(i+1, len(mean)):
            delta = float(mean[i]-mean[j])
            variance = float(covariance[i,i]+covariance[j,j]-2*covariance[i,j])
            if variance < -1e-10:
                raise ValueError('Negative Gaussian difference variance')
            if variance <= 0:
                continue
            s = np.sqrt(variance)
            t = delta/s
            absolute = s*np.sqrt(2/np.pi)*np.exp(-.5*t*t)+delta*(1-2*ndtr(-t))
            half = np.sqrt(s)*2**.25*gamma(.75)/np.sqrt(np.pi)*hyp1f1(-.25,.5,-.5*t*t)
            pair = float(absolute-half*half)
            # Mathematical variance is nonnegative. Only round-off is clipped.
            tolerance = 1e-10*max(1.,abs(absolute))
            if pair < -tolerance or not np.isfinite(pair):
                raise ValueError('Unstable folded-normal moment')
            pair = max(pair,0.)
            minimum_pair_variance = pair if minimum_pair_variance is None else min(minimum_pair_variance,pair)
            joint += 2*pair
    values = {'marginal': marginal, 'joint': float(joint), 'tail': tail}
    scales = {k: v if v > 0 else 1. for k,v in values.items()}
    if not all(np.isfinite(v) and v > 0 for v in scales.values()):
        raise ValueError('Invalid expected-error scale')
    return scales, {'normalization_epoch': 'expected_m0_error_2026-10-06T20:11Z',
                    'method': 'M0-BASELINE.md section5 exact Gaussian formulas; hypergeometric folded moments',
                    'depends_on_outcome': False, 'depends_on_seed': False,
                    'minimum_pair_variance': minimum_pair_variance,
                    'zero_component_rule': 'store1.0', 'card_clip_and_failure_value': 8.}


def forecast(histories, horizons, *, target_type, unit_id, n_draws=500, steps=None):
    mean, covariance, factor, diagnostic = distribution(histories,horizons,target_type=target_type,steps=steps)
    scales, normalization = expected_scales(mean,covariance)
    seed = zlib.crc32(unit_id.encode()) & 0x7fffffff
    samples = mean+np.random.default_rng(seed).standard_normal((n_draws,len(mean)))@factor.T
    diagnostic.update(seed=int(seed), n_draws=n_draws, expected_error_scales=scales,
                      normalization=normalization, mean_authored_cells=mean.tolist(),
                      sd_authored_cells=np.sqrt(np.diag(covariance)).tolist())
    return samples.reshape(n_draws,len(histories),len(horizons)),diagnostic
