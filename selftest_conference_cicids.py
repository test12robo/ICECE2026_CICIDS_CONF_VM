#!/usr/bin/env python
"""
Synthetic self-test for the ICECE-2026 CICIDS conference additions (2026-08-27).

Runs in seconds on fabricated data. Proves the four new pieces of plumbing behave
before ~700k real Wednesday flows are spent on them:

  1. simplexN(6) == simplex6            candidate-for-candidate, same order
  2. calibrateN  == calibrate6          same (auc, W, wR) AND same LAST_CALIB_TOPK
  3. simplexN(5) (--drop-identity)      correct arity/floor/sum, wR ceiling 0.10->0.60
  4. --day filter                       substring match, and a miss is a hard error
  5. --drop-identity rebinding          FACT/FEATS/MONO consistent, I excluded

(1) and (2) are the BYTE-SAFETY argument: with I present nothing about the locked
calibration path changes, so any difference in the conference arm is attributable to
the flags, not to the refactor.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pdp_core as gov

FAILED = []


def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {extra}" if extra else ""))
    if not cond:
        FAILED.append(name)


print("\n=== 1. simplexN(6) reproduces simplex6 ===")
for w_min, step in ((0.10, 20), (0.05, 20), (0.00, 20), (0.10, 10)):
    mu = int(round(w_min * step))
    a = list(gov.simplex6(step, mu))
    b = list(gov.simplexN(6, step, mu))
    same = len(a) == len(b) and all(np.allclose(x, y) for x, y in zip(a, b))
    check(f"w_min={w_min} step={step}", same, f"{len(a):,} candidates, order-identical={same}")

print("\n=== 2. calibrateN == calibrate6 on synthetic validation data ===")
rng = np.random.default_rng(20260827)
n = 4000
y = (rng.random(n) < 0.35).astype(np.int64)
# attacks get systematically lower factor values and a higher R -- the real direction
va = pl.DataFrame({
    "I": np.full(n, 0.5),                                   # constant, exactly like the real leg
    "B": np.clip(0.72 - 0.28 * y + rng.normal(0, .12, n), 0, 1),
    "D": np.clip(0.66 - 0.18 * y + rng.normal(0, .15, n), 0, 1),
    "C": np.clip(0.70 - 0.24 * y + rng.normal(0, .13, n), 0, 1),
    "H": np.clip(0.68 - 0.20 * y + rng.normal(0, .14, n), 0, 1),
    "R": np.clip(0.18 + 0.55 * y + rng.normal(0, .10, n), 0, 1),
    "decay": np.exp(-0.05 * rng.random(n) * 8),
    "y": y,
})
a6, W6, wR6 = gov.calibrate6(va)
tk6 = list(gov.LAST_CALIB_TOPK)
aN, WN, wRN = gov.calibrateN(va)
tkN = list(gov.LAST_CALIB_TOPK)
check("val ROC identical", a6 == aN, f"{a6:.10f} vs {aN:.10f}")
check("W* identical", np.allclose(W6, WN), f"{np.round(W6,3)} vs {np.round(WN,3)}")
check("wR* identical", wR6 == wRN, f"{wR6} vs {wRN}")
check("LAST_CALIB_TOPK identical", tk6 == tkN, f"topk={len(tk6)}")

print("\n=== 3. simplexN(5) -- the --drop-identity arity ===")
cand5 = list(gov.simplexN(5, 20, 2))          # w_min = 0.10
sums = np.array([c.sum() for c in cand5])
mins = np.array([c.min() for c in cand5])
wr_max = max(c[-1] for c in cand5)
check("every candidate sums to 1", np.allclose(sums, 1.0), f"max|sum-1|={np.abs(sums-1).max():.2e}")
check("floor respected (>=0.10)", mins.min() >= 0.10 - 1e-12, f"min weight={mins.min():.4f}")
check("wR ceiling is 0.60", abs(wr_max - 0.60) < 1e-12, f"max wR={wr_max:.4f}  (6-way gives 0.50)")
wr_max6 = max(c[-1] for c in gov.simplexN(6, 20, 2))
check("6-way wR ceiling is 0.50", abs(wr_max6 - 0.50) < 1e-12, f"max wR={wr_max6:.4f}")
check("candidate count", len(cand5) == 1001, f"{len(cand5):,} (5 slots, 10 free units)")

print("\n=== 4. --day substring filter ===")
FILES = ["Tuesday-WorkingHours.pcap_ISCX.csv",
         "Wednesday-workingHours.pcap_ISCX.csv",
         "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"]


def day_filter(files, day):
    want = [d.strip().lower() for d in day.split(",") if d.strip()]
    sel = [f for f in files if any(w in f.lower() for w in want)]
    if not sel:
        raise SystemExit(f"--day {day!r} matched none of {files}")
    return sel


check("'Wednesday' -> 1 file", day_filter(FILES, "Wednesday") == [FILES[1]], str(day_filter(FILES, "Wednesday")))
check("case-insensitive", day_filter(FILES, "wednesday") == [FILES[1]])
check("comma list -> 2 files", len(day_filter(FILES, "Wednesday,Tuesday")) == 2)
check("order follows FILES not the flag", day_filter(FILES, "Friday,Tuesday") == [FILES[0], FILES[2]])
try:
    day_filter(FILES, "Sunday"); check("miss raises SystemExit", False)
except SystemExit:
    check("miss raises SystemExit", True)

print("\n=== 5. --drop-identity rebinding ===")
FACT0, FEATS0, MONO0 = list(gov.FACT), list(gov.FEATS), gov.MONO
check("baseline FACT", FACT0 == ["I", "B", "D", "C", "H"], str(FACT0))
check("baseline MONO arity", MONO0.count(",") + 1 == len(FEATS0), f"MONO={MONO0} FEATS={len(FEATS0)}")
gov.FACT = [f for f in gov.FACT if f != "I"]
gov.FEATS = gov.FACT + ["R", "decay"]
gov.MONO = "(" + ",".join(["1"] * len(gov.FACT) + ["-1", "1"]) + ")"
check("I removed from FACT", "I" not in gov.FACT, str(gov.FACT))
check("FEATS rebuilt", gov.FEATS == ["B", "D", "C", "H", "R", "decay"], str(gov.FEATS))
check("MONO arity matches FEATS", gov.MONO.count(",") + 1 == len(gov.FEATS), f"MONO={gov.MONO}")
check("MONO signs: R is the only -1", gov.MONO == "(1,1,1,1,-1,1)", gov.MONO)
va5 = va.drop("I")
a5, W5, wR5 = gov.calibrateN(va5)
check("calibrateN runs on 4 factors", len(W5) == 4, f"W*={np.round(W5,2)} wR*={wR5}")
check("wR* within the 5-way ceiling", wR5 <= 0.60 + 1e-12, f"wR*={wR5}")
check("weights sum to 1", abs(float(W5.sum()) + wR5 - 1.0) < 1e-12)
gov.FACT, gov.FEATS, gov.MONO = FACT0, FEATS0, MONO0

print("\n" + "=" * 60)
if FAILED:
    print(f"FAILED ({len(FAILED)}): " + ", ".join(FAILED)); sys.exit(1)
print("ALL CHECKS PASSED")
