"""
pdp_core -- the trust-equation primitives shared by the FSL-PDP legs.

VENDORED 2026-08-11 from `formula_supervised_pdp/cicids/governance_eval_cicids.py`
by exact line-range extraction (see make_cicids_clean.py), NOT by retyping.

The CICIDS experiment used only these 7 symbols out of that 37 KB module, which
also carried a full CLI, its own main(), and calibration modes CICIDS never used.
Importing the whole thing made the dependency look far larger than it was.

Equivalence is verified end-to-end: the clean package reproduces the 2026-06-18
`metrics_multiday.json` before any behaviour change is applied.

    FACT, FEATS, MONO   factor names, model feature order, monotone constraints
    make_target         teacher-target transform on T_add (linear minmax, no S-curve)
    roc                 rank AUC
    simplex6            6-way weight simplex with a defense-in-depth floor
    t_add               T = (S @ W) * decay - wR * R
    calibrate6          grid-search W*, wR* on validation additive ROC
"""
import numpy as np


# --- FACT / FEATS / MONO  (governance_eval_cicids.py lines 24-24) ---
FACT = ["I", "B", "D", "C", "H"]; FEATS = FACT + ["R", "decay"]; MONO = "(1,1,1,1,1,-1,1)"


# --- make_target  (governance_eval_cicids.py lines 36-42) ---
def make_target(mode, tmin, tmax):
    """Teacher-target transform on T_add (NO S-curve — sigmoid retired 2026-06-16, linear ≥ sigmoid).
    linear=minmax(T_add)→[0,1] (train min/max, causal; DEFAULT); raw=T_add unscaled (ablation)."""
    rng = max(float(tmax) - float(tmin), 1e-9)
    if mode == "linear":  return lambda Tadd: np.clip((Tadd - tmin) / rng, 0.0, 1.0)
    if mode == "raw":     return lambda Tadd: np.asarray(Tadd, dtype=np.float64)
    raise ValueError(f"unknown target transform: {mode}")


# --- roc  (governance_eval_cicids.py lines 45-49) ---
def roc(score, y):
    y = np.asarray(y); npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0: return float("nan")
    score = np.asarray(score)
    if np.ptp(score) == 0: return 0.5  # constant score is uninformative -- avoid a stable-sort tie-order artifact
    o = np.argsort(score, kind="stable"); s = score[o]
    ranks = np.arange(1, len(score)+1, dtype=np.float64)
    new_grp = np.empty(len(s), dtype=bool); new_grp[0] = True; new_grp[1:] = s[1:] != s[:-1]
    grp = np.cumsum(new_grp) - 1
    r = np.empty(len(score)); r[o] = (np.bincount(grp, weights=ranks) / np.bincount(grp))[grp]
    return (r[y == 1].sum() - npos*(npos+1)/2) / (npos*nneg)


# --- simplex6  (governance_eval_cicids.py lines 83-95) ---
def simplex6(step=20, min_units=0):
    # all 6 weights over [I,B,D,C,H,R] sum to 1, each >= min_units/step (defense-in-depth FLOOR:
    # no signal — incl R — can be zeroed). Enumerate the free budget, then add the floor.
    base = step - 6 * min_units
    if base < 0:
        raise ValueError("w_min too large: need 6*w_min <= 1")
    for a in range(base + 1):
        for b in range(base + 1 - a):
            for c in range(base + 1 - a - b):
                for d in range(base + 1 - a - b - c):
                    for e in range(base + 1 - a - b - c - d):
                        f = base - a - b - c - d - e
                        yield (np.array([a, b, c, d, e, f], float) + min_units) / step


# --- t_add  (governance_eval_cicids.py lines 98-98) ---
def t_add(S, R, W, wR, decay): return (S @ W) * decay - wR * R   # T = (Σ W·s)·e^(−λΔt) − wR·R


# --- calibrate6  (governance_eval_cicids.py lines 124-135) ---
# Populated by calibrate6 on every call: the ranked grid points and the margin between
# the winner and the runner-up. ADDED 2026-08-12 after a VM rerun selected a DIFFERENT
# weight vector (B .15/wR .45 -> B .20/wR .40) off a ~1e-4 change in the external IDS R.
# `best` is chosen by a strict `>` over an unordered grid scan, so when the top candidates
# are separated by less than float noise the winner is effectively arbitrary. Recording the
# margin makes that fragility auditable instead of invisible. This does NOT change the
# selection -- it only observes it.
LAST_CALIB_TOPK: list = []


def calibrate6(va, w_min=0.10, step=20, topk=8):
    """UNIFIED 6-way simplex: w_I+w_B+w_D+w_C+w_H+w_R = 1, each >= w_min (defense-in-depth FLOOR:
    every signal incl R always participates; nothing collapses to 0). T = (S·W5)·decay − w_R·R."""
    global LAST_CALIB_TOPK
    S = va.select(FACT).to_numpy(); R = va["R"].to_numpy(); y = va["y"].to_numpy()
    dc = va["decay"].to_numpy()
    min_units = int(round(w_min * step))
    best = (-1, None, None)
    scored = []
    for w in simplex6(step, min_units):
        W, wR = w[:5], w[5]
        a = roc(-((S @ W) * dc - wR * R), y)
        scored.append((a, tuple(round(float(x), 4) for x in W), float(wR)))
        if a > best[0]: best = (a, W, float(wR))
    scored.sort(key=lambda t: -t[0])
    LAST_CALIB_TOPK = [
        {"val_roc": round(float(a), 8),
         "W": dict(zip(FACT, w)), "wR": round(wr, 4),
         "gap_to_best": round(float(scored[0][0] - a), 10)}
        for a, w, wr in scored[:topk]
    ]
    return best




# =====================================================================================
# CONFERENCE ADDITION (2026-08-27) -- arity-generalized simplex + calibration.
#
# Ported VERBATIM from the UNSW conference pipeline
# (ICECE2026_CONFERENCE_PAPER/pipeline/steps/step9_pdp.py:191-254), which itself
# generalized this leg's simplex6/calibrate6. Kept here so the CICIDS Wednesday arm
# and the UNSW arm run the SAME calibration code path with the same tie-breaking.
#
# simplex6/calibrate6 above are UNTOUCHED and remain the locked leg's path.
# `simplexN(6)` reproduces `simplex6` candidate-for-candidate in the same order --
# asserted by selftest_conference_cicids.py.
# =====================================================================================
def simplexN(n, step=20, min_units=0):
    """All `n` weights (len(FACT) factors, then w_R) sum to 1, each >= min_units/step.

    Same construction as simplex6: pre-charge every slot its floor, enumerate how the remaining
    free units are split over the first n-1 slots, and give the LAST slot the remainder so the sum
    closes exactly with no rounding drift. Emits candidates in the same lexicographic order.
    """
    base = step - n * min_units
    if base < 0:
        raise ValueError(f"w_min too large: need {n}*w_min <= 1 (n={n}, min_units={min_units})")

    def rec(slots_left, rem):
        if slots_left == 1:
            yield [rem]                      # remainder slot -- this is what forces sum == step
            return
        for v in range(rem + 1):
            for tail in rec(slots_left - 1, rem - v):
                yield [v] + tail

    for c in rec(n, base):
        yield (np.array(c, float) + min_units) / step


def calibrateN(va, w_min=0.10, step=20, topk=8):
    """Arity-generalized calibrate6: simplex over len(FACT) factor weights + w_R, summing to 1,
    each >= w_min. Identical to calibrate6 when FACT has 5 entries.

    Reads the MODULE-LEVEL FACT at call time, so --drop-identity can rebind it first."""
    global LAST_CALIB_TOPK
    S = va.select(FACT).to_numpy(); R = va["R"].to_numpy(); y = va["y"].to_numpy()
    dc = va["decay"].to_numpy()
    k = len(FACT)
    min_units = int(round(w_min * step))
    best = (-1, None, None)
    scored = []
    for w in simplexN(k + 1, step, min_units):
        W, wR = w[:k], w[k]
        a = roc(-((S @ W) * dc - wR * R), y)
        scored.append((a, tuple(round(float(x), 4) for x in W), float(wR)))
        if a > best[0]: best = (a, W, float(wR))
    scored.sort(key=lambda t: -t[0])
    LAST_CALIB_TOPK = [
        {"val_roc": round(float(a), 8),
         "W": dict(zip(FACT, w)), "wR": round(wr, 4),
         "gap_to_best": round(float(scored[0][0] - a), 10)}
        for a, w, wr in scored[:topk]
    ]
    return best
