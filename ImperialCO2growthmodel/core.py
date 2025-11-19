# -*- coding: utf-8 -*-
from __future__ import annotations
import math, warnings
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import pandas as pd

from .models import (
    logistic_cumulative, logistic_rate, logistic_inflection_and_peak,
    gompertz_cumulative_tau, gompertz_rate_tau, gompertz_inflection_and_peak_abs
)
from .calibrate import (
    r_from_rate_logistic, r_from_rate_gompertz,
    r_from_cumulative_logistic, r_from_cumulative_gompertz,
)
from .constants import ALPHA, ONE_PLUS_E_ALPHA, BETA, E_BETA

def _info(msg: str): print(f"[INFO] {msg}")
def _warn(msg: str): warnings.warn(msg)
def _err(msg: str):  raise ValueError(msg)

def calculate_cagr(P_t0: float, P_t1: float, t0: float, t1: float) -> float:
    if P_t0 <= 0 or P_t1 <= 0 or t1 <= t0:
        return float('nan')
    return (P_t1 / P_t0) ** (1.0 / (t1 - t0)) - 1.0

def _validate_inputs(model, start_year, target_year, S0, capacity, Q_target, P_target, tn_min, growth_limit, end_year):
    if str(model).lower() not in ('logistic', 'gompertz'):
        _err("model must be 'Logistic' or 'Gompertz'.")
    if start_year >= target_year:
        _err("target_year must be greater than start_year.")
    if S0 <= 0:
        _err("Starting-year cumulative (S0) must be > 0.")
    if capacity is not None and capacity <= S0:
        _err("Storage resource (C) must be greater than S0.")
    if (Q_target is None) and (P_target is None):
        _err("Provide at least one of target-year *annual rate* (Q_target) or *cumulative* (P_target).")
    if (Q_target is not None) and Q_target <= 0:
        _err("Q_target must be positive (Gt/yr).")
    if (P_target is not None) and P_target <= S0:
        _err("P_target must be greater than S0 and smaller than the resource capacity.")
    if end_year is not None and end_year <= start_year:
        _err("end_year must be greater than start_year.")
    if growth_limit is not None and growth_limit <= 0:
        _err("growth_limit must be positive (e.g., 0.2 for 20%).")

def _compute_inflection_and_peak(model: str, start_year: int, capacity: float, r: float, S0: float):
    if model.lower() == 'logistic':
        tn, tp, _ = logistic_inflection_and_peak(capacity, r, start_year, S0); return tn, tp
    else:
        tn, tp = gompertz_inflection_and_peak_abs(start_year, capacity, S0, r); return tn, tp

def _generate_series(model: str, start_year: int, end_year: int, capacity: float, r: float, S0: float) -> pd.DataFrame:
    years = list(range(start_year, end_year + 1))
    if model.lower() == 'logistic':
        cum = [logistic_cumulative(y, capacity, r, start_year, S0) for y in years]
        rate = [logistic_rate(y, capacity, r, start_year, S0) for y in years]
    else:
        b = math.log(capacity / S0)
        tau_list = [y - start_year for y in years]
        cum = [gompertz_cumulative_tau(tau, capacity, b, r) for tau in tau_list]
        rate = [gompertz_rate_tau(tau, capacity, b, r) for tau in tau_list]
    df = pd.DataFrame({'Year': years, 'Cumulative_Gt': cum, 'Rate_Gt_per_yr': rate})
    return df

def _summary_table(model: str, start_year: int, target_year: int, capacity: float, r: float, S0: float, end_year: int):
    tn, tp = _compute_inflection_and_peak(model, start_year, capacity, r, S0)
    df = _generate_series(model, start_year, max(end_year, int(math.ceil(tp))+1), capacity, r, S0)
    df['Growth_Status'] = df['Year'].apply(lambda y: 'inflection' if abs(y - tn) < 0.5 else ('accelerating' if y < tn else 'decelerating'))

    def interp_at(year_float: float):
        y0 = int(math.floor(year_float)); y1 = y0 + 1
        if y0 < start_year: y0 = start_year; y1 = start_year + 1
        if y1 > df['Year'].max(): y1 = df['Year'].max(); y0 = y1 - 1
        frac = year_float - y0
        tmp_r0 = df.loc[df['Year'] == y0, 'Rate_Gt_per_yr']
        r0 = float(tmp_r0.iloc[0]) if not tmp_r0.empty else float('nan')

        tmp_r1 = df.loc[df['Year'] == y1, 'Rate_Gt_per_yr']
        r1 = float(tmp_r1.iloc[0]) if not tmp_r1.empty else float('nan')

        tmp_c0 = df.loc[df['Year'] == y0, 'Cumulative_Gt']
        c0 = float(tmp_c0.iloc[0]) if not tmp_c0.empty else float('nan')

        tmp_c1 = df.loc[df['Year'] == y1, 'Cumulative_Gt']
        c1 = float(tmp_c1.iloc[0]) if not tmp_c1.empty else float('nan')

        return (c0 + frac*(c1 - c0), r0 + frac*(r1 - r0))

    cum_tn, rate_tn = interp_at(tn)
    cum_tp, rate_tp = interp_at(tp)

    if target_year in df['Year'].values:
        _st = df.loc[df['Year'] == target_year, 'Cumulative_Gt']
        storage_target = float(_st.iloc[0]) if not _st.empty else float('nan')
        _rt = df.loc[df['Year'] == target_year, 'Rate_Gt_per_yr']
        rate_target = float(_rt.iloc[0]) if not _rt.empty else float('nan')
    else:
        storage_target = float('nan')
        rate_target = float('nan')

    def val_at(y):
        if y < start_year or y > df['Year'].max():
            return float('nan')
        _c = df.loc[df['Year'] == y, 'Cumulative_Gt']
        return float(_c.iloc[0]) if not _c.empty else float('nan')
    

    def calculate_cagr(P_t0, P_t1, t0, t1):
        if P_t0 <= 0 or P_t1 <= 0 or t1 <= t0: return float('nan')
        return (P_t1 / P_t0) ** (1.0 / (t1 - t0)) - 1.0

    cagr_start_tn      = calculate_cagr(S0, cum_tn, start_year, tn) if tn > start_year else float('nan')
    cagr_start_target  = calculate_cagr(S0, val_at(target_year), start_year, target_year) if target_year >= start_year else float('nan')
    cagr_target_plus50 = calculate_cagr(val_at(target_year), val_at(min(target_year+50, df['Year'].max())), target_year, min(target_year+50, df['Year'].max()))
    cagr_start_end     = calculate_cagr(S0, val_at(end_year), start_year, end_year) if end_year <= df['Year'].max() else float('nan')

    summary = pd.DataFrame([{
        'Model': model.capitalize(),
        'S0_Gt': S0,
        'Capacity_Gt': capacity,
        'r_per_year': r,
        'Tn_year': tn,
        'Tp_year': tp,
        'Rate_at_Tn_Gt_yr': rate_tn,
        'Rate_at_Tp_Gt_yr': rate_tp,
        'Cum_at_Tn_Gt': cum_tn,
        'Cum_at_Tp_Gt': cum_tp,
        f'Storage_{target_year}_Gt': storage_target,
        f'Rate_{target_year}_Gt_yr': rate_target,
        'CAGR_start_Tn_%': cagr_start_tn * 100.0 if cagr_start_tn == cagr_start_tn else float('nan'),
        f'CAGR_{start_year}_{target_year}_%': cagr_start_target * 100.0 if cagr_start_target == cagr_start_target else float('nan'),
        f'CAGR_{target_year}_{target_year+50}_%': cagr_target_plus50 * 100.0 if cagr_target_plus50 == cagr_target_plus50 else float('nan'),
        f'CAGR_{start_year}_{end_year}_%': cagr_start_end * 100.0 if cagr_start_end == cagr_start_end else float('nan'),
    }])
    return df, summary, tn, tp

def build_single_model(
    model: str, start_year: int, target_year: int, S0: float,
    capacity: Optional[float] = None, Q_target: Optional[float] = None, P_target: Optional[float] = None,
    tn_min: Optional[float] = None, growth_limit: Optional[float] = None, end_year: Optional[int] = None,
    branch: str = 'left'
) -> Dict[str, Any]:
    _validate_inputs(model, start_year, target_year, S0, capacity, Q_target, P_target, tn_min, growth_limit, end_year)
    end_year = end_year if end_year is not None else (start_year + 400)
    model_l = model.lower()
    warnings_list: List[str] = []

    if capacity is None and (Q_target is not None) and (growth_limit is not None):
        _info("Capacity not provided: a growth-limited trade-off scan is required to infer minimal resource.")
        _err("Capacity missing: please call through the high-level API which can infer minimal resource.")

    if capacity is None:
        _err("Please provide capacity (C), or use the high-level API to auto-solve minimal resource.")

    if Q_target is not None:
        if model_l == 'logistic':
            r = r_from_rate_logistic(Q_target, capacity, S0, start_year, target_year, branch=branch)
        else:
            r = r_from_rate_gompertz(Q_target, capacity, S0, start_year, target_year)
    elif P_target is not None:
        if model_l == 'logistic':
            r = r_from_cumulative_logistic(P_target, capacity, S0, start_year, target_year)
        else:
            r = r_from_cumulative_gompertz(P_target, capacity, S0, start_year, target_year)
    else:
        _err("Internal error: neither Q_target nor P_target was available.")

    yearly_df, summary_df, tn, tp = _summary_table(model_l, start_year, target_year, capacity, r, S0, end_year)

    if tn_min is not None and tn < tn_min:
        msg = f"Inflection-year constraint not met: Tn={tn:.2f} < required {tn_min:.0f}."
        warnings_list.append(msg); _warn(msg)

    if growth_limit is not None:
        y0 = int(math.floor(tn)); y1 = y0 + 1
        if y0 < start_year: y0 = start_year; y1 = start_year + 1
        if y1 > yearly_df['Year'].max(): y1 = yearly_df['Year'].max(); y0 = y1 - 1
        frac = tn - y0
        _tmp0 = yearly_df.loc[yearly_df['Year'] == y0, 'Cumulative_Gt']
        c0 = float(_tmp0.iloc[0]) if not _tmp0.empty else float('nan')

        _tmp1 = yearly_df.loc[yearly_df['Year'] == y1, 'Cumulative_Gt']
        c1 = float(_tmp1.iloc[0]) if not _tmp1.empty else float('nan')
        P_tn = c0 + frac*(c1 - c0)
        cagr_to_tn = calculate_cagr(S0, P_tn, start_year, tn)
        if cagr_to_tn == cagr_to_tn and cagr_to_tn > (growth_limit + 1e-12):
            msg = f"CAGR constraint not met: CAGR(start→Tn) ≈ {cagr_to_tn*100:.2f}% > limit {growth_limit*100:.2f}%."
            warnings_list.append(msg); _warn(msg)

    yearly_df['Rate_Mt_per_yr'] = yearly_df['Rate_Gt_per_yr'] * 1000.0

    params = {
        'model': model_l, 'start_year': start_year, 'target_year': target_year,
        'S0_Gt': S0, 'capacity_Gt': capacity, 'r_per_year': r, 'Tn_year': tn, 'Tp_year': tp,
        'Q_target_Gt_yr': Q_target, 'P_target_Gt': P_target
    }
    return {'yearly': yearly_df, 'summary': summary_df, 'warnings': warnings_list, 'params': params}

def tradeoff_table(model: str, start_year: int, target_year: int, S0: float, Q_target: float,
                   cagr_min: float = 0.001, cagr_max: float = 0.20, n_grid: int = 220) -> pd.DataFrame:
    import numpy as np
    if Q_target <= 0 or S0 <= 0 or target_year <= start_year:
        raise ValueError("tradeoff_table: invalid inputs (Q_target>0, S0>0, target_year>start_year required).")
    cagr_vals = np.linspace(cagr_min, cagr_max, n_grid)
    delta_t = target_year - start_year
    from .constants import ALPHA, ONE_PLUS_E_ALPHA, BETA, E_BETA
    def min_capacity_logistic(c: float) -> float | None:
        def r_from_cagr(C: float) -> float | None:
            if C <= S0 * ONE_PLUS_E_ALPHA: return None
            k = (C - S0)/S0
            ln1pc = math.log(1.0 + c)
            denom = math.log(C) - math.log(ONE_PLUS_E_ALPHA) - math.log(S0)
            num_shift = math.log(k) - ALPHA
            if ln1pc <= 0 or denom <= 0 or num_shift <= 0: return None
            return ln1pc * num_shift / denom
        def feasible(C: float) -> float:
            r = r_from_cagr(C)
            if r is None: return -float('inf')
            R = logistic_rate(start_year + delta_t, C, r, start_year, S0)
            return R - Q_target
        C_low = S0 * ONE_PLUS_E_ALPHA * (1.0 + 1e-9)
        C_high = max(4.0 * C_low, 2.0)
        f_high = feasible(C_high)
        it = 0
        while (not math.isfinite(f_high) or f_high < 0) and C_high < 1e7 and it < 200:
            C_high *= 2.0; f_high = feasible(C_high); it += 1
        if not math.isfinite(f_high) or f_high < 0: return None
        for _ in range(200):
            C_mid = 0.5*(C_low + C_high); f_mid = feasible(C_mid)
            if not math.isfinite(f_mid): C_low = C_mid; continue
            if f_mid >= 0: C_high = C_mid
            else: C_low = C_mid
            if abs(C_high - C_low) <= 1e-6 * max(1.0, C_high): break
        return C_high
    def min_capacity_gompertz(c: float) -> float | None:
        def r_from_cagr(C: float) -> float | None:
            if C <= S0 * math.exp(E_BETA): return None
            b = math.log(C / S0)
            ln1pc = math.log(1.0 + c)
            denom = math.log(C) - E_BETA - math.log(S0)
            num_shift = math.log(b) - BETA
            if ln1pc <= 0 or denom <= 0 or num_shift <= 0: return None
            return ln1pc * num_shift / denom
        def feasible(C: float) -> float:
            r = r_from_cagr(C)
            if r is None: return -float('inf')
            b = math.log(C / S0)
            R = gompertz_rate_tau(delta_t, C, b, r)
            return R - Q_target
        C_low = S0 * math.exp(E_BETA) * (1.0 + 1e-9)
        C_high = max(4.0 * C_low, 2.0)
        f_high = feasible(C_high)
        it = 0
        while (not math.isfinite(f_high) or f_high < 0) and C_high < 1e7 and it < 200:
            C_high *= 2.0; f_high = feasible(C_high); it += 1
        if not math.isfinite(f_high) or f_high < 0: return None
        for _ in range(200):
            C_mid = 0.5*(C_low + C_high); f_mid = feasible(C_mid)
            if not math.isfinite(f_mid): C_low = C_mid; continue
            if f_mid >= 0: C_high = C_mid
            else: C_low = C_mid
            if abs(C_high - C_low) <= 1e-6 * max(1.0, C_high): break
        return C_high
    caps = []
    for c in cagr_vals:
        cap = min_capacity_logistic(float(c)) if model.lower() == 'logistic' else min_capacity_gompertz(float(c))
        caps.append(float('nan') if cap is None else cap)
    return pd.DataFrame({'CAGR_%': cagr_vals * 100.0, 'Required_Resource_Gt': caps})

def export_csvs_single(result: Dict[str, Any], outdir: str) -> Dict[str, str]:
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    yearly = result['yearly']; summary = result['summary']
    cum_csv = str(out / 'cumulative_by_year.csv')
    rate_csv = str(out / 'annual_rate_by_year.csv')
    summary_csv = str(out / 'summary_table.csv')
    yearly[['Year','Cumulative_Gt']].to_csv(cum_csv, index=False)
    yearly[['Year','Rate_Gt_per_yr','Rate_Mt_per_yr','Growth_Status']].to_csv(rate_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    return {'cumulative_csv': cum_csv, 'rate_csv': rate_csv, 'summary_csv': summary_csv}