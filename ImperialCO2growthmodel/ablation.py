# -*- coding: utf-8 -*-
"""
Ablation scenarios where growth constraints (Tn-min, CAGR limit) are disabled.

Scenario 1 (peak -> target-year metrics):
    Given base_year t0, end_year, anchor P(t0)=S0, capacity C, and a list of peak-year rates Qmax (Mt/yr).
    For each Qmax:
        Logistic: r = 4*Qmax/C
        Gompertz: r = e*Qmax/C
    Build yearly series (cumulative & rate), compute Tn, Tp, P(target_year), Q(target_year),
    P(Tn), Peak rate, and Growth = CAGR(t0->Tn). Export CSVs per-scenario and one global summary CSV.

Scenario 2 (solve capacity from Qmax and target-year Q):
    Given base_year t0, end_year, anchor P(t0)=S0, a Qmax (Mt/yr), target_year t*, and target-year rate Q(t*) (Mt/yr).
    For chosen model(s), solve capacity C such that:
        Logistic: r = 4*Qmax/C and R(t*; C, r) = Q_target
        Gompertz: r = e*Qmax/C and R(t*; C, r) = Q_target
    Then build yearly series + summary (same fields as Scenario 1).
"""
from __future__ import annotations
import math
from typing import Dict, Any, Iterable, Optional
import numpy as np
import pandas as pd
from pathlib import Path

from .models import (
    logistic_cumulative, logistic_rate, logistic_inflection_and_peak,
    gompertz_cumulative_tau, gompertz_rate_tau, gompertz_inflection_and_peak_abs,
)
from .core import export_csvs_single
from .constants import ALPHA, BETA, ONE_PLUS_E_ALPHA, E_BETA

def _info(msg: str): print(f"[INFO] {msg}")
def _err(msg: str): raise ValueError(msg)

def _series_from_given_r(model: str, start_year: int, end_year: int, C: float, r: float, S0: float) -> pd.DataFrame:
    years = list(range(start_year, end_year + 1))
    if model.lower() == 'logistic':
        cum = [logistic_cumulative(y, C, r, start_year, S0) for y in years]
        rate = [logistic_rate(y, C, r, start_year, S0) for y in years]
        tn, tp, _ = logistic_inflection_and_peak(C, r, start_year, S0)
    else:
        b = math.log(C / S0)
        taus = [y - start_year for y in years]
        cum = [gompertz_cumulative_tau(tau, C, b, r) for tau in taus]
        rate = [gompertz_rate_tau(tau, C, b, r) for tau in taus]
        tn, tp = gompertz_inflection_and_peak_abs(start_year, C, S0, r)
    df = pd.DataFrame({'Year': years, 'Cumulative_Gt': cum, 'Rate_Gt_per_yr': rate})
    df['Rate_Mt_per_yr'] = df['Rate_Gt_per_yr'] * 1000.0
    df['Growth_Status'] = df['Year'].apply(
        lambda y: 'inflection' if abs(y - tn) < 0.5 else ('accelerating' if y < tn else 'decelerating')
    )
    return df

def _cagr(P0: float, P1: float, t0: float, t1: float) -> float:
    if P0 <= 0 or P1 <= 0 or t1 <= t0: return float('nan')
    return (P1 / P0) ** (1.0 / (t1 - t0)) - 1.0

def _summary_from_given_r(model: str, start_year: int, end_year: int, target_year: int,
                          C: float, r: float, S0: float, yearly: pd.DataFrame) -> pd.DataFrame:
    if model.lower() == 'logistic':
        tn, tp, _ = logistic_inflection_and_peak(C, r, start_year, S0)
        P_tn = C / ONE_PLUS_E_ALPHA
        peak_rate = C * r / 4.0
    else:
        tn, tp = gompertz_inflection_and_peak_abs(start_year, C, S0, r)
        P_tn = C * math.exp(-E_BETA)
        peak_rate = C * r / math.e

    def val_at(df, y, col):
        if y < df['Year'].min() or y > df['Year'].max():
            return float('nan')
        _s = df.loc[df['Year'] == y, col]
        return float(_s.iloc[0]) if not _s.empty else float('nan')

    P_target = val_at(yearly, target_year, 'Cumulative_Gt')
    Q_target = val_at(yearly, target_year, 'Rate_Gt_per_yr')
    growth = _cagr(S0, P_tn, start_year, tn) * 100.0  # %

    return pd.DataFrame([{
        'Model': model.capitalize(),
        'S0_Gt': S0,
        'Capacity_Gt': C,
        'r_per_year': r,
        'Tn_year': tn,
        'Tp_year': tp,
        f'Storage_{start_year}_Gt': S0,
        f'Storage_{target_year}_Gt': P_target,
        f'Rate_{target_year}_Gt_yr': Q_target,
        'Cum_at_Tn_Gt': P_tn,
        'Peak_rate_Gt_yr': peak_rate,
        'CAGR_start_Tn_%': growth
    }])

# -----------------------------
# Scenario 1: known C & Qmax list -> compute 2050 metrics (both models)
# -----------------------------
def ablation_peak_to_target_rates(
    base_year: int,
    end_year: int,
    S0: float,
    capacity_gt: float,
    qmax_list_mt: Iterable[float],
    target_year: int,
    export_to: Optional[str] = None,
    export_include: Iterable[str] = ('cumulative','rate','summary'),
    label_prefix: str = 'Qmax'
) -> Dict[str, Any]:
    """
    For each Qmax (Mt/yr), compute r for both models:
      Logistic: r = 4*Qmax/C
      Gompertz: r = e*Qmax/C
    anchored by P(base_year)=S0, then build yearly series and a summary row.

    Returns:
      {'results': {label: result_dict, ...},
       'global_summary': DataFrame,
       'written': {label: paths, ...},  # only when export_to is provided
       'global_summary_csv': '.../summary_all_runs.csv'  # only when export_to
      }
    """
    results = {}
    rows = []
    qmax_list_gt = [float(q)/1000.0 for q in qmax_list_mt]

    for model in ('Logistic','Gompertz'):
        for qmax_mt, qmax_gt in zip(qmax_list_mt, qmax_list_gt):
            C = capacity_gt
            r = (4.0 * qmax_gt / C) if model=='Logistic' else (math.e * qmax_gt / C)
            yearly = _series_from_given_r(model, base_year, end_year, C, r, S0)
            summary = _summary_from_given_r(model, base_year, end_year, target_year, C, r, S0, yearly)
            label = f"{model}_{label_prefix}{int(round(qmax_mt))}Mt"
            result = {'yearly': yearly, 'summary': summary, 'warnings': [], 'params': {
                'model': model.lower(), 'start_year': base_year, 'end_year': end_year,
                'target_year': target_year, 'S0_Gt': S0, 'capacity_Gt': C, 'r_per_year': r,
                'Qmax_Gt_yr': qmax_gt
            }, 'label': label}
            results[label] = result
            row = summary.iloc[0].to_dict()
            row.update({'Label': label, 'Qmax_Mt_yr': qmax_mt, 'Qmax_Gt_yr': qmax_gt})
            rows.append(row)

    global_summary = pd.DataFrame(rows)

    if export_to is not None:
        base = Path(export_to); base.mkdir(parents=True, exist_ok=True)
        written = {}
        for label, res in results.items():
            folder = base / label; folder.mkdir(parents=True, exist_ok=True)
            paths = export_csvs_single(res, str(folder))
            if 'cumulative' not in export_include:
                Path(paths['cumulative_csv']).unlink(missing_ok=True); paths.pop('cumulative_csv', None)
            if 'rate' not in export_include:
                Path(paths['rate_csv']).unlink(missing_ok=True); paths.pop('rate_csv', None)
            if 'summary' not in export_include:
                Path(paths['summary_csv']).unlink(missing_ok=True); paths.pop('summary_csv', None)
            written[label] = paths
        global_path = base / 'summary_all_runs.csv'
        global_summary.to_csv(global_path, index=False)
        return {'results': results, 'written': written, 'global_summary_csv': str(global_path), 'global_summary': global_summary}
    return {'results': results, 'global_summary': global_summary}

# -----------------------------
# Scenario 2: known Qmax & Q(target) -> solve C (single or both models)
# -----------------------------
def _solve_capacity_from_peak_and_target_single(
    model: str,
    base_year: int,
    end_year: int,
    S0: float,
    qmax_mt: float,
    q_target_mt: float,
    target_year: int,
    C_min: Optional[float] = None,
    C_max: Optional[float] = None,
) -> Dict[str, Any]:
    Qmax = qmax_mt / 1000.0
    Qtgt = q_target_mt / 1000.0
    if Qtgt > Qmax + 1e-15:
        _err(f"For {model}: target-year rate {q_target_mt} Mt/yr exceeds Qmax {qmax_mt} Mt/yr (infeasible).")
    t0, tstar = base_year, target_year

    def R_at(C: float) -> float:
        if C <= S0*(1+1e-12) or not math.isfinite(C): return float('-inf')
        r = (4.0*Qmax/C) if model.lower()=='logistic' else (math.e*Qmax/C)
        if model.lower()=='logistic':
            S = logistic_cumulative(tstar, C, r, t0, S0)
            return r * S * (1.0 - S/C)
        else:
            b = math.log(C / S0)
            tau = tstar - t0
            s = b * math.exp(-r * tau)
            return C * r * s * math.exp(-s)

    lo = C_min if (C_min is not None) else (S0 * 1.000001)
    hi = C_max if (C_max is not None) else max(10.0, S0*10.0)
    f_lo = R_at(lo) - Qtgt
    f_hi = R_at(hi) - Qtgt

    it = 0
    while (not math.isfinite(f_hi) or f_lo*f_hi > 0) and hi < 1e10 and it < 200:
        hi *= 2.0
        f_hi = R_at(hi) - Qtgt
        it += 1

    if (not math.isfinite(f_hi)) or f_lo*f_hi > 0:
        grid = np.logspace(math.log10(max(S0*1.000001, 1e-6)), math.log10(max(hi, S0*1e6)), num=4000)
        vals = [R_at(c)-Qtgt for c in grid]
        idx = np.where(np.diff(np.sign(vals)) != 0)[0]
        if len(idx) == 0:
            _err(f"{model}: could not bracket a solution for capacity under given (Qmax, Q_target).")
        lo, hi = grid[idx[0]], grid[idx[0]+1]

    for _ in range(240):
        mid = 0.5*(lo+hi)
        fm = R_at(mid) - Qtgt
        f_lo = R_at(lo) - Qtgt
        if f_lo*fm <= 0:
            hi = mid
        else:
            lo = mid
        if abs(hi-lo) <= 1e-10*max(1.0, hi): break
    C = 0.5*(lo+hi)

    r = (4.0*Qmax/C) if model.lower()=='logistic' else (math.e*Qmax/C)
    yearly = _series_from_given_r(model, base_year, end_year, C, r, S0)
    summary = _summary_from_given_r(model, base_year, end_year, target_year, C, r, S0, yearly)
    # explicitly set Q(target) value
    col = f'Rate_{target_year}_Gt_yr'
    summary.loc[:, col] = yearly.loc[yearly['Year']==target_year, 'Rate_Gt_per_yr'].values[0]
    return {'yearly': yearly, 'summary': summary, 'warnings': [], 'params': {
        'model': model.lower(), 'start_year': base_year, 'end_year': end_year, 'target_year': target_year,
        'S0_Gt': S0, 'capacity_Gt': C, 'r_per_year': r, 'Qmax_Gt_yr': Qmax, 'Q_target_Gt_yr': Qtgt
    }, 'label': f"{model}_solveC_Qmax{int(round(qmax_mt))}Mt_Q{int(round(q_target_mt))}Mt"}

def ablation_solve_capacity_from_peak_and_target(
    base_year: int,
    end_year: int,
    S0: float,
    qmax_mt: float,
    q_target_mt: float,
    target_year: int,
    model: str | Iterable[str] = ('Logistic','Gompertz'),
    export_to: Optional[str] = None,
    export_include: Iterable[str] = ('cumulative','rate','summary')
) -> Dict[str, Any]:
    """
    Given Qmax (Mt/yr) and target-year rate Q(t*) (Mt/yr), solve capacity C for the chosen model(s),
    then produce yearly series + summary (no growth-limit or Tn-min constraints).
    """
    models = [model] if isinstance(model, str) else list(model)
    outputs = {}
    for m in models:
        res = _solve_capacity_from_peak_and_target_single(m, base_year, end_year, S0, qmax_mt, q_target_mt, target_year)
        outputs[res['label']] = res

    if export_to is not None:
        base = Path(export_to); base.mkdir(parents=True, exist_ok=True)
        written = {}
        for label, res in outputs.items():
            folder = base / label; folder.mkdir(parents=True, exist_ok=True)
            paths = export_csvs_single(res, str(folder))
            if 'cumulative' not in export_include:
                Path(paths['cumulative_csv']).unlink(missing_ok=True); paths.pop('cumulative_csv', None)
            if 'rate' not in export_include:
                Path(paths['rate_csv']).unlink(missing_ok=True); paths.pop('rate_csv', None)
            if 'summary' not in export_include:
                Path(paths['summary_csv']).unlink(missing_ok=True); paths.pop('summary_csv', None)
            written[label] = paths
        summary_all = pd.concat([v['summary'].assign(Label=k) for k,v in outputs.items()], ignore_index=True)
        global_csv = base / 'summary_all_runs.csv'
        summary_all.to_csv(global_csv, index=False)
        return {'results': outputs, 'written': written, 'global_summary_csv': str(global_csv), 'global_summary': summary_all}
    return {'results': outputs}