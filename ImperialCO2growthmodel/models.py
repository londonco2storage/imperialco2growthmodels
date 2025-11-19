# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from typing import Tuple

# ---------- Logistic ----------
def logistic_cumulative(t: float, C: float, r: float, t0: float, S0: float) -> float:
    if not (S0 > 0 and C > S0 and r > 0):
        raise ValueError("Invalid params for logistic_cumulative: require C>S0>0 and r>0.")
    k = (C - S0) / S0
    return C / (1.0 + k * math.exp(-r * (t - t0)))

def logistic_rate(t: float, C: float, r: float, t0: float, S0: float) -> float:
    S_t = logistic_cumulative(t, C, r, t0, S0)
    return r * S_t * (1.0 - S_t / C)

def logistic_inflection_and_peak(C: float, r: float, t0: float, S0: float) -> Tuple[float, float, float]:
    tp = t0 + math.log((C - S0) / S0) / r
    tn = tp - math.log(2.0 + math.sqrt(3.0)) / r
    t2 = tp + math.log(2.0 + math.sqrt(3.0)) / r
    return tn, tp, t2

# ---------- Gompertz ----------
def gompertz_cumulative_tau(tau: float, C: float, b: float, r: float) -> float:
    """
    Gompertz cumulative storage P(τ) with τ = t - t0:
        P(τ) = C * exp(-b * exp(-r * τ)),  where b = ln(C/S0) > 0.
    """
    return C * math.exp(-b * math.exp(-r * tau))

def gompertz_rate_tau(tau: float, C: float, b: float, r: float) -> float:
    """
    Gompertz annual storage rate:
        dP/dt = C * r * s * exp(-s),  with s = b * exp(-r * τ).
        Equivalently: dP/dt = r * P * ln(C/P).
    """
    s = b * math.exp(-r * tau)
    return C * r * s * math.exp(-s)

def gompertz_inflection_and_peak_abs(t0: float, C: float, S0: float, r: float):
    """
    Peak/inflection times for the Gompertz *rate* curve (absolute years):
        τp = ln(b)/r,  τn = τp - ln((3+√5)/2)/r,  with b = ln(C/S0).
        Returns (t0+τn, t0+τp).
    """
    b = math.log(C / S0)
    tau_p = math.log(b) / r
    tau_n = tau_p - math.log((3.0 + math.sqrt(5.0)) / 2.0) / r
    return (t0 + tau_n, t0 + tau_p)