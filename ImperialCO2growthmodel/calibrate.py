# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from typing import Optional, List
from .models import logistic_cumulative, logistic_rate, gompertz_cumulative_tau, gompertz_rate_tau
from .solvers import bracket_sign_changes, bisection
from .constants import ALPHA, ONE_PLUS_E_ALPHA, BETA, E_BETA

# ---- 1) r from target *rate* at target_year ----
def r_from_rate_logistic(target_rate: float, C: float, S0: float, t0: float, target_year: float,
                         branch: str = 'left', r_min: float = 1e-5, r_max: float = 1.0, n_grid: int = 2000) -> float:
    dt = target_year - t0
    if not (C > S0 > 0 and target_rate > 0 and dt > 0):
        raise ValueError("r_from_rate_logistic: invalid inputs (need C>S0>0, target_rate>0, target_year>t0).")
    rs = [r_min + i*(r_max - r_min)/(n_grid-1) for i in range(n_grid)]
    def f(r): return logistic_rate(t0 + dt, C, r, t0, S0) - target_rate
    intervals = bracket_sign_changes(f, rs)
    roots = []
    for a, b in intervals:
        root = bisection(f, a, b, tol=1e-12, max_iter=200)
        if root is not None: roots.append(root)
    if not roots:
        raise ValueError("r_from_rate_logistic: no real root in the searched interval. Try widening r range.")
    roots = sorted(set(round(r, 12) for r in roots))
    return roots[0] if branch == 'left' else roots[-1]

def r_from_rate_gompertz(target_rate: float, C: float, S0: float, t0: float, target_year: float,
                         r_min: float = 1e-6, r_max: float = 1.5, n_grid: int = 3000) -> float:
    if not (C > S0 > 0 and target_rate > 0 and target_year > t0):
        raise ValueError("r_from_rate_gompertz: invalid inputs (need C>S0>0, target_rate>0, target_year>t0).")
    b = math.log(C / S0)
    tau = target_year - t0
    def f(r):
        if r <= 0: return float('inf')
        try: return gompertz_rate_tau(tau, C, b, r) - target_rate
        except Exception: return float('inf')
    rs = [r_min + i*(r_max - r_min)/(n_grid-1) for i in range(n_grid)]
    intervals = bracket_sign_changes(f, rs)
    roots = []
    for a, bnd in intervals:
        root = bisection(f, a, bnd, tol=1e-12, max_iter=200)
        if root is not None: roots.append(root)
    if not roots:
        raise ValueError("r_from_rate_gompertz: no real root in the searched interval. Try widening r range.")
    roots = sorted(set(round(r, 12) for r in roots))
    return roots[0]

# ---- 2) r from target *cumulative* at target_year (closed-form) ----
def r_from_cumulative_logistic(target_P: float, C: float, S0: float, t0: float, target_year: float) -> float:
    dt = target_year - t0
    if not (C > S0 > 0 and target_P > S0 and target_P < C and dt > 0):
        raise ValueError("r_from_cumulative_logistic: invalid inputs or infeasible target cumulative.")
    arg = (C - target_P) * S0 / (target_P * (C - S0))
    if arg <= 0:
        raise ValueError("r_from_cumulative_logistic: infeasible target cumulative (argument <= 0).")
    return -math.log(arg) / dt

def r_from_cumulative_gompertz(target_P: float, C: float, S0: float, t0: float, target_year: float) -> float:
    if not (C > S0 > 0 and target_P > S0 and target_P < C and target_year > t0):
        raise ValueError("r_from_cumulative_gompertz: invalid inputs or infeasible target cumulative.")
    b = math.log(C / S0)
    tau = target_year - t0
    x = (1.0 / b) * math.log(C / target_P)  # must be in (0,1)
    if not (0.0 < x < 1.0):
        raise ValueError("r_from_cumulative_gompertz: infeasible target cumulative (x must be in (0,1)).")
    return -math.log(x) / tau