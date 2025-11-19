# -*- coding: utf-8 -*-
"""
ImperialCO2growthmodel
======================
CO₂ storage growth modeling under Logistic and Gompertz assumptions.

Public API
----------
Core:
- build_models(...): Calibrate a model (or multiple scenarios) and return yearly series + summary.
- tradeoff_table(...): Build a "CAGR(start→Tn) vs Required Resource" table.
- export_csv_results(...): Unified CSV exporter (choose which tables to save and where).

Shortcuts:
- logisticgrowth(...): Convenience wrapper for Logistic scenarios (+ optional export).
- Gompertzgrowth(...): Convenience wrapper for Gompertz scenarios (+ optional export).

Utilities:
- check_requirements(): Print versions and guidance for required packages.
"""
from .api import build_models, export_csv_results, check_requirements
from .core import tradeoff_table
from .shortcuts import logisticgrowth, Gompertzgrowth
from .ablation import (
    ablation_peak_to_target_rates,
    ablation_solve_capacity_from_peak_and_target,
)