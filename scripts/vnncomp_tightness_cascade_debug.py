#!/usr/bin/env python3
"""Does the 'cascade' ablation variant actually run, and does it change any interval box?"""

import numpy as np
import vnncomp_tightness_audit as T

specs, _, r64 = T.load("acasxu", "1.0")
rec = next(r for r in specs if "3_1" in r["onnx"] and r["group"] == 0)
lo_d, hi_d, _ = T.ours(rec, "default")
lo_c, hi_c, _ = T.ours(rec, "cascade")
print(
    "cascade stats:",
    {k: (v if k != "errors" else v[:3]) for k, v in T.CASCADE_STATS.items()},
)
print("default  lb:", np.round(lo_d, 4), " ub:", np.round(hi_d, 4))
print("cascade  lb:", np.round(lo_c, 4), " ub:", np.round(hi_c, 4))
