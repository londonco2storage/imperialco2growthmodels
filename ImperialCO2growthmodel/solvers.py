# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Callable, Optional, List
import math

def is_finite(x: float) -> bool:
    return x == x and abs(x) != float('inf')

def bracket_sign_changes(f: Callable[[float], float], xs: List[float]) -> List[tuple]:
    intervals = []
    prev_x = xs[0]
    prev_f = f(prev_x)
    for x in xs[1:]:
        fx = f(x)
        if (is_finite(prev_f) and is_finite(fx)) and (prev_f == 0 or fx == 0 or (prev_f * fx < 0)):
            intervals.append((prev_x, x))
        prev_x, prev_f = x, fx
    return intervals

def bisection(f: Callable[[float], float], a: float, b: float, tol: float = 1e-12, max_iter: int = 200) -> Optional[float]:
    fa, fb = f(a), f(b)
    if not (is_finite(fa) and is_finite(fb)):
        return None
    if fa == 0.0: return a
    if fb == 0.0: return b
    if fa * fb > 0: return None
    left, right = a, b
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        fm = f(mid)
        if not is_finite(fm):
            right = mid; continue
        if abs(right - left) <= tol * max(1.0, abs(right)):
            return mid
        if fa * fm <= 0:
            right = mid; fb = fm
        else:
            left = mid; fa = fm
    return 0.5 * (left + right)