# -*- coding: utf-8 -*-
from __future__ import annotations
import math, warnings
from typing import Optional, Dict, Any, Tuple, List, Iterable, Union
from pathlib import Path

def check_requirements(raise_on_missing: bool = False) -> Dict[str, Optional[str]]:
    info = {}
    try:
        import pandas as pd  # noqa
        info['pandas'] = getattr(pd, '__version__', 'present')
    except Exception:
        info['pandas'] = None
        print("[ERROR] pandas is required. Install via: pip install pandas")
        if raise_on_missing: raise
    try:
        import numpy as np  # noqa
        info['numpy'] = getattr(np, '__version__', 'present')
    except Exception:
        info['numpy'] = None
        print("[ERROR] numpy is required (for trade-off tables). Install via: pip install numpy")
    return info

from .core import build_single_model, tradeoff_table, export_csvs_single

def _ensure_list(x, length: Optional[int] = None):
    if isinstance(x, (list, tuple)):
        return list(x)
    if length is None:
        return [x]
    return [x for _ in range(length)]

def build_models(
    model: str,
    start_year: int,
    target_year: int,
    S0: float,
    capacity: Optional[Union[float, List[float]]] = None,
    Q_target: Optional[Union[float, List[float]]] = None,
    P_target: Optional[Union[float, List[float]]] = None,
    tn_min: Optional[float] = None,
    growth_limit: Optional[float] = None,
    end_year: Optional[int] = None,
    branch: str = 'left',
    labels: Optional[List[str]] = None,
    infer_capacity_when_missing: bool = True,
    tradeoff_for_inference_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    check_requirements(raise_on_missing=True)
    def _len(x):
        if isinstance(x, (list, tuple)): return len(x)
        return 1
    lens = [_len(capacity), _len(Q_target), _len(P_target)]
    N = max(lens)

    cap_list = _ensure_list(capacity, N)
    q_list   = _ensure_list(Q_target, N)
    p_list   = _ensure_list(P_target, N)

    if not (len(cap_list) == len(q_list) == len(p_list) == N):
        raise ValueError("build_models: broadcasting failed. Ensure capacity/Q_target/P_target are scalars or lists with compatible lengths.")

    if labels is None:
        labels = [f"scenario_{i+1}" for i in range(N)]
    elif len(labels) != N:
        raise ValueError("labels must match the number of scenarios.")

    results: List[Dict[str, Any]] = []
    for i in range(N):
        cap_i, q_i, p_i = cap_list[i], q_list[i], p_list[i]
        if cap_i is None and (q_i is not None) and (growth_limit is not None) and infer_capacity_when_missing:
            kwargs = dict(model=model, start_year=start_year, target_year=target_year, S0=S0, Q_target=q_i, cagr_min=0.001, cagr_max=growth_limit, n_grid=120)
            if isinstance(tradeoff_for_inference_kwargs, dict):
                kwargs.update(tradeoff_for_inference_kwargs)
            df_trade = tradeoff_table(**kwargs)
            finite_caps = df_trade['Required_Resource_Gt'].dropna()
            if finite_caps.empty:
                raise ValueError(f"No feasible resource found under growth_limit for scenario '{labels[i]}' to meet the target-year rate.")
            cap_i = float(finite_caps.min())

        res = build_single_model(
            model=model, start_year=start_year, target_year=target_year, S0=S0,
            capacity=cap_i, Q_target=q_i, P_target=p_i,
            tn_min=tn_min, growth_limit=growth_limit, end_year=end_year, branch=branch
        )
        res['label'] = labels[i]
        results.append(res)

    return results[0] if N == 1 else results

def export_csv_results(
    results: Union[Dict[str, Any], List[Dict[str, Any]]],
    outdir: str,
    include: Iterable[str] = ('cumulative','rate','summary','tradeoff'),
    tradeoff_df: Optional[Union['pd.DataFrame', List['pd.DataFrame']]] = None,
    tradeoff_filenames: Optional[Union[str, List[str]]] = None,
    per_scenario_subdir: bool = True,
) -> Dict[str, Any]:
    import pandas as pd
    from .core import export_csvs_single

    if isinstance(results, dict):
        results_list = [results]
    else:
        results_list = list(results)
    M = len(results_list)

    if tradeoff_df is None:
        trade_list = [None] * M
    elif isinstance(tradeoff_df, list):
        if len(tradeoff_df) != M:
            raise ValueError("tradeoff_df list length must match number of scenarios.")
        trade_list = tradeoff_df
    else:
        trade_list = [tradeoff_df] * M

    if tradeoff_filenames is None:
        trade_names = [None] * M
    elif isinstance(tradeoff_filenames, list):
        if len(tradeoff_filenames) != M:
            raise ValueError("tradeoff_filenames list length must match number of scenarios.")
        trade_names = tradeoff_filenames
    else:
        trade_names = [tradeoff_filenames] * M

    out = {}
    base = Path(outdir); base.mkdir(parents=True, exist_ok=True)

    for i, res in enumerate(results_list):
        label = res.get('label', f"scenario_{i+1}")
        folder = base / label if (per_scenario_subdir and len(results_list) > 1) else base
        folder.mkdir(parents=True, exist_ok=True)

        written = {}
        if any(k in include for k in ('cumulative','rate','summary')):
            paths = export_csvs_single(res, str(folder))
            if 'cumulative' not in include:
                try: Path(paths['cumulative_csv']).unlink()
                except Exception: pass
                paths.pop('cumulative_csv', None)
            if 'rate' not in include:
                try: Path(paths['rate_csv']).unlink()
                except Exception: pass
                paths.pop('rate_csv', None)
            if 'summary' not in include:
                try: Path(paths['summary_csv']).unlink()
                except Exception: pass
                paths.pop('summary_csv', None)
            written.update(paths)

        if 'tradeoff' in include:
            df = trade_list[i]
            if df is None:
                print(f"[INFO] No tradeoff_df provided for '{label}'. Skipping trade-off export.")
            else:
                name = trade_names[i] or 'tradeoff.csv'
                path = folder / name
                df.to_csv(path, index=False)
                written['tradeoff_csv'] = str(path)

        out[label] = written

    return out