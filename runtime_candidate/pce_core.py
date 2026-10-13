"""Pure fixed PCE/CPI proxy models: no source reading, network or scoring.

Own PCE ends before the target month; only CPI for that month may enter the
conditional feature. Sensitive CPI cells are dropped before numeric conversion.
OOF residuals refit only on prior mature PCE labels. Native additive covariance
is exactly shared for the lag-versus-conditional-mean comparison.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from .canonical_m0 import distribution, expected_scales
PROTOCOL = json.loads((HERE / "pce_constants.json").read_text(encoding="utf-8"))
PCE = PROTOCOL["pce_assets"]
CPI = PROTOCOL["cpi_assets"]

def scale_from_prefix(frame, assets, horizons):
    histories = {a: {"dates": frame.index.to_numpy(dtype="datetime64[D]"),
                     "values": frame[a].to_numpy()} for a in assets}
    mean, covariance, _, diagnostic = distribution(histories, horizons, target_type="level")
    scales, _ = expected_scales(mean, covariance)
    return scales, diagnostic


def monthly_index(index) -> pd.PeriodIndex:
    if isinstance(index, pd.PeriodIndex):
        if index.freqstr != "M":
            raise ValueError("index must identify monthly observation periods")
        result = index
    else:
        result = pd.DatetimeIndex(index).to_period("M")
    if result.has_duplicates or not result.is_monotonic_increasing:
        raise ValueError("monthly periods must be unique and increasing")
    return result


def validate_clock(metadata: dict, target: pd.Period) -> dict:
    # Production calendar is bound by the external own-unit adapter, never a fabricated 2024 date.
    if metadata.get("production_gate_verified") is not True:
        raise ValueError("own-unit production metadata gate not verified")
    if metadata.get("target_observation_period") != str(target):
        raise ValueError("target-period metadata mismatch")
    return {"calendar_cutoff": metadata["asof_utc"],
            "date_only_release_evidence": False,
            "source_vintage_is_pit": False,
            "first_release_predictive_evidence": False,
            "availability_evidence": "own-unit explicit timestamps; not independently PIT audited",
            "inner_oof_vintage_limit": "retrospective supplied vintage, not fold-specific vintage"}


def prepare_prefixes(pce_prefix: pd.DataFrame, cpi_prefix: pd.DataFrame,
                     target: pd.Period) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if list(pce_prefix.columns) != PCE or list(cpi_prefix.columns) != CPI:
        raise ValueError("prefixes need exact canonical PCE and CPI column orders")
    pce_dates, cpi_dates = monthly_index(pce_prefix.index), monthly_index(cpi_prefix.index)
    # Check metadata before any value conversion, including poisoned future rows.
    if not len(pce_dates) or pce_dates[-1] != target-1:
        raise ValueError("PCE prefix must end at the preceding observation month")
    if len(cpi_dates) and cpi_dates[-1] > target:
        raise ValueError("future CPI observation period is forbidden")
    if not pce_dates.equals(pd.period_range(pce_dates[0], pce_dates[-1], freq="M")):
        raise ValueError("PCE history has a missing observation month")
    pce_raw = pce_prefix.copy()
    pce_raw.index = pce_dates
    pce = pd.DataFrame({asset: pd.to_numeric(pce_raw[asset], errors="raise") for asset in PCE}, index=pce_dates).astype(float)
    if not np.isfinite(pce.to_numpy()).all() or (pce <= 0).any().any():
        raise ValueError("PCE index values must be finite and positive")
    desired = pd.period_range(pce_dates[0], target, freq="M")
    cpi_raw = cpi_prefix.copy()
    cpi_raw.index = cpi_dates
    columns, masked = {}, {}
    for asset in CPI:
        forbidden = pd.PeriodIndex(PROTOCOL["privacy_mask"][asset], freq="M")
        # Do not interpret, float-cast or difference forbidden cells.
        safe = cpi_raw.loc[~cpi_dates.isin(forbidden), asset]
        numeric = pd.to_numeric(safe, errors="raise").astype(float)
        valid = numeric.dropna()
        if not np.isfinite(valid.to_numpy()).all() or (valid <= 0).any():
            raise ValueError("unmasked CPI indices must be finite and positive")
        columns[asset] = numeric.reindex(desired)
        masked[asset] = [str(period) for period in cpi_dates if period in forbidden]
    cpi = pd.DataFrame(columns, index=desired)
    # Reindexing puts NaN in holes; diff never jumps over a missing month.
    log_cpi = 100*np.log(cpi).diff()
    return pce, log_cpi, {"masked_before_numeric_conversion": masked,
                          "current_cpi_log_change_available": {asset: bool(pd.notna(log_cpi.loc[target, asset])) for asset in CPI}}


def common_frames(pce: pd.DataFrame, cpi_changes: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    changes = 100*np.log(pce).diff()
    frames = {}
    for asset, proxy in zip(PCE, CPI):
        frames[asset] = pd.DataFrame({"cpi": cpi_changes[proxy],
            "pce_lag": changes[asset].shift(1), "label": changes[asset]}).dropna()
    return frames, changes


def fit_ridge(frame: pd.DataFrame, columns: list[str]) -> dict:
    train = frame.iloc[-PROTOCOL["training_window_complete_labels"]:]
    if len(train) < PROTOCOL["minimum_training_complete_labels"]:
        raise ValueError("insufficient common complete training labels")
    x = train[columns].to_numpy(dtype=float)
    center, scale = x.mean(axis=0), x.std(axis=0, ddof=1)
    scale = np.where(scale > 1e-12, scale, 1.)
    design = np.column_stack([np.ones(len(x)), (x-center)/scale])
    penalty = np.eye(design.shape[1])*PROTOCOL["ridge_penalty"]
    penalty[0, 0] = PROTOCOL["ridge_intercept_penalty"]
    coefficients = np.linalg.solve(design.T@design+penalty, design.T@train.label.to_numpy())
    return {"columns": columns, "center": center, "scale": scale, "coefficients": coefficients,
            "labels": len(train), "first_label_period": str(train.index[0]),
            "last_label_period": str(train.index[-1])}


def ridge_point(fit: dict, inputs: list[float]) -> float:
    x = np.r_[1., (np.asarray(inputs)-fit["center"])/fit["scale"]]
    value = float(x@fit["coefficients"])
    if not np.isfinite(value):
        raise ValueError("nonfinite fitted log-change point")
    return value


def points_at(frames: dict, pce_changes: pd.DataFrame, cpi_changes: pd.DataFrame,
              period: pd.Period) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    lag_points, conditional_points, ledger = [], [], []
    for asset, proxy in zip(PCE, CPI):
        common = frames[asset].loc[frames[asset].index < period]
        lag_fit = fit_ridge(common, ["pce_lag"])
        conditional_fit = fit_ridge(common, ["cpi", "pce_lag"])
        assert lag_fit["labels"] == conditional_fit["labels"]
        assert lag_fit["last_label_period"] == conditional_fit["last_label_period"]
        assert pd.Period(lag_fit["last_label_period"], freq="M") < period
        lag_change = float(pce_changes.loc[period-1, asset])
        lag_point = ridge_point(lag_fit, [lag_change])
        available = period in cpi_changes.index and pd.notna(cpi_changes.loc[period, proxy])
        conditional = ridge_point(conditional_fit, [float(cpi_changes.loc[period, proxy]), lag_change]) if available else lag_point
        lag_points.append(lag_point)
        conditional_points.append(conditional)
        ledger.append({"asset": asset, "prediction_observation_period": str(period),
            "training_labels": lag_fit["labels"], "first_training_label_period": lag_fit["first_label_period"],
            "last_training_label_period": lag_fit["last_label_period"], "cpi_proxy_available": bool(available),
            "conditional_mean_is_lag_fallback": not bool(available),
            "lag_standardized_coefficients": lag_fit["coefficients"].tolist(),
            "conditional_standardized_coefficients": conditional_fit["coefficients"].tolist()})
    return np.array(lag_points), np.array(conditional_points), ledger


def covariance_root(covariance: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh((covariance+covariance.T)/2)
    if not np.isfinite(eigenvalues).all() or eigenvalues.min() < -1e-10:
        raise ValueError("invalid empirical residual covariance")
    return eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.)))


def oof_covariances(pce: pd.DataFrame, cpi_changes: pd.DataFrame,
                    frames: dict, pce_changes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    periods = pce.index[-PROTOCOL["oof_residual_calendar_months"]:]
    if len(periods) != PROTOCOL["oof_residual_calendar_months"]:
        raise ValueError("insufficient mature OOF calendar episodes")
    lag_errors, conditional_errors, ledger = [], [], []
    for period in periods:
        lag, conditional, fold = points_at(frames, pce_changes, cpi_changes, period)
        actual_ratio = pce.loc[period].to_numpy()/pce.loc[period-1].to_numpy()
        lag_errors.append(actual_ratio-np.exp(lag/100))
        conditional_errors.append(actual_ratio-np.exp(conditional/100))
        ledger.append({"mature_residual_period": str(period), "fitting": fold,
                       "current_proxy_fallback_used": any(row["conditional_mean_is_lag_fallback"] for row in fold)})
    lag_errors, conditional_errors = np.asarray(lag_errors), np.asarray(conditional_errors)
    lag_cov, conditional_cov = np.cov(lag_errors, rowvar=False, ddof=1), np.cov(conditional_errors, rowvar=False, ddof=1)
    covariance_root(lag_cov)
    covariance_root(conditional_cov)
    ledger.append({"residual_means_not_used_for_bias_retuning": True,
                   "lag_relative_residual_mean": lag_errors.mean(axis=0).tolist(),
                   "conditional_relative_residual_mean": conditional_errors.mean(axis=0).tolist()})
    return lag_cov, conditional_cov, ledger


def forecast(pce_prefix: pd.DataFrame, cpi_prefix: pd.DataFrame,
             target_observation_period: str, calendar_metadata: dict) -> tuple[dict[str, np.ndarray], dict]:
    target = pd.Period(target_observation_period, freq="M")
    clock = validate_clock(calendar_metadata, target)
    pce, cpi_changes, privacy = prepare_prefixes(pce_prefix, cpi_prefix, target)
    frames, pce_changes = common_frames(pce, cpi_changes)
    lag_point, conditional_point, fits = points_at(frames, pce_changes, cpi_changes, target)
    lag_relative_cov, conditional_relative_cov, oof = oof_covariances(pce, cpi_changes, frames, pce_changes)
    anchor = pce.iloc[-1].to_numpy()
    lag_center, conditional_center = anchor*np.exp(lag_point/100), anchor*np.exp(conditional_point/100)
    lag_cov = anchor[:, None]*lag_relative_cov*anchor[None, :]
    all_proxy_known = all(privacy["current_cpi_log_change_available"].values())
    full_cov = (anchor[:, None]*conditional_relative_cov*anchor[None, :]) if all_proxy_known else lag_cov
    timestamps = pce.index.to_timestamp()
    histories = {asset: {"dates": timestamps.to_numpy(dtype="datetime64[D]"),
                         "values": pce[asset].to_numpy()} for asset in PCE}
    m0_center, m0_cov, m0_factor, _ = distribution(histories, [1], target_type="level")
    standard = np.random.default_rng(PROTOCOL["common_draw_seed"]).standard_normal((PROTOCOL["draws"], len(PCE)))
    lag_noise = standard@covariance_root(lag_cov).T
    full_noise = standard@covariance_root(full_cov).T if all_proxy_known else lag_noise
    samples = {"M0_month": m0_center[None, :]+standard@m0_factor.T,
               "PCE_lag_only": lag_center[None, :]+lag_noise,
               "conditional_CPI_mean": conditional_center[None, :]+lag_noise,
               "conditional_CPI_distribution": conditional_center[None, :]+full_noise}
    if not all(np.isfinite(array).all() for array in samples.values()):
        raise ValueError("nonfinite prediction samples")
    assert np.allclose(samples["conditional_CPI_mean"]-samples["PCE_lag_only"],
                       (conditional_center-lag_center)[None, :], rtol=0, atol=2e-12)
    if not all_proxy_known:
        assert np.array_equal(samples["conditional_CPI_distribution"], samples["conditional_CPI_mean"])
    pce_timestamp_frame = pce.copy()
    pce_timestamp_frame.index = timestamps
    scales = {}
    for asset in PCE:
        scales[asset], _ = scale_from_prefix(pce_timestamp_frame, [asset], [1])
    scales["joint"], _ = scale_from_prefix(pce_timestamp_frame, PCE, [1])
    ledger = {"target_observation_period": str(target), "research_horizon_key": 1,
              "preceding_pce_observation_period": str(pce.index[-1]),
              "clock": clock, "calendar_metadata": calendar_metadata, "privacy": privacy,
              "fitting": fits, "oof": oof,
              "lag_log_change_points_percent": lag_point.tolist(),
              "conditional_log_change_points_percent": conditional_point.tolist(),
              "lag_native_centers": lag_center.tolist(), "conditional_native_centers": conditional_center.tolist(),
              "lag_native_covariance": lag_cov.tolist(), "full_native_covariance": full_cov.tolist(),
              "full_covariance_is_whole_joint_lag_fallback": not all_proxy_known,
              "expected_M0_scales_input_PCE_only": scales,
              "sampling_space": "native additive Gaussian; transformed-log-regression centers",
              "no_future_target_pce_read": True, "current_vintage_research_proxy_only": True,
              "negative_sample_counts_not_clipped": {key: int(np.sum(value <= 0)) for key, value in samples.items()}}
    return samples, ledger
