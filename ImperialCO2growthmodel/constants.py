# -*- coding: utf-8 -*-
import math
ALPHA = math.log(2.0 + math.sqrt(3.0))            # ≈ 1.316957 (Logistic rate left-inflection offset)
ONE_PLUS_E_ALPHA = 1.0 + math.exp(ALPHA)          # = 3 + sqrt(3) ≈ 4.732051
BETA = math.log((3.0 + math.sqrt(5.0)) / 2.0)     # ≈ 0.962424 (Gompertz rate left-inflection offset)
E_BETA = math.exp(BETA)                            # = (3+sqrt(5))/2 ≈ 2.618034