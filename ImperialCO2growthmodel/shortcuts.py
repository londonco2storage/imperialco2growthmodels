# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Dict, Any, List, Iterable, Union
from .api import build_models, export_csv_results
from .core import tradeoff_table

def logisticgrowth(
    start_year: int, target_year: int, S0: float,
    capacity: Optional[Union[float, List[float]]] = None,
    Q_target: Optional[Union[float, List[float]]] = None,
    P_target: Optional[Union[float, List[float]]] = None,
    tn_min: Optional[float] = None, growth_limit: Optional[float] = None, end_year: Optional[int] = None,
    branch: str = 'left', labels: Optional[List[str]] = None,
    export_to: Optional[str] = None, export_include: Iterable[str] = ('cumulative','rate','summary'),
    tradeoff: Optional[Dict[str, Any]] = None,
):
    results = build_models('Logistic', start_year, target_year, S0, capacity, Q_target, P_target,
                           tn_min, growth_limit, end_year, branch, labels,
                           infer_capacity_when_missing=True, tradeoff_for_inference_kwargs=None)
    trade_list = None
    if tradeoff is not None:
        results_list = [results] if isinstance(results, dict) else list(results)
        trade_list = []
        for res in results_list:
            q = res['params']['Q_target_Gt_yr']
            if q is None: trade_list.append(None); continue
            kwargs = dict(model='Logistic', start_year=start_year, target_year=target_year, S0=S0,
                          Q_target=q, cagr_min=0.001, cagr_max=growth_limit if growth_limit is not None else 0.20, n_grid=60)
            kwargs.update(tradeoff or {})
            trade_list.append(tradeoff_table(**kwargs))
    if export_to is not None:
        return results, export_csv_results(results, export_to, include=export_include if tradeoff is None else set(export_include)|{'tradeoff'}, tradeoff_df=trade_list)
    return results

def Gompertzgrowth(
    start_year: int, target_year: int, S0: float,
    capacity: Optional[Union[float, List[float]]] = None,
    Q_target: Optional[Union[float, List[float]]] = None,
    P_target: Optional[Union[float, List[float]]] = None,
    tn_min: Optional[float] = None, growth_limit: Optional[float] = None, end_year: Optional[int] = None,
    labels: Optional[List[str]] = None,
    export_to: Optional[str] = None, export_include: Iterable[str] = ('cumulative','rate','summary'),
    tradeoff: Optional[Dict[str, Any]] = None,
):
    results = build_models('Gompertz', start_year, target_year, S0, capacity, Q_target, P_target,
                           tn_min, growth_limit, end_year, 'left', labels,
                           infer_capacity_when_missing=True, tradeoff_for_inference_kwargs=None)
    trade_list = None
    if tradeoff is not None:
        results_list = [results] if isinstance(results, dict) else list(results)
        trade_list = []
        for res in results_list:
            q = res['params']['Q_target_Gt_yr']
            if q is None: trade_list.append(None); continue
            kwargs = dict(model='Gompertz', start_year=start_year, target_year=target_year, S0=S0,
                          Q_target=q, cagr_min=0.001, cagr_max=growth_limit if growth_limit is not None else 0.20, n_grid=60)
            kwargs.update(tradeoff or {})
            trade_list.append(tradeoff_table(**kwargs))
    if export_to is not None:
        return results, export_csv_results(results, export_to, include=export_include if tradeoff is None else set(export_include)|{'tradeoff'}, tradeoff_df=trade_list)
    return results