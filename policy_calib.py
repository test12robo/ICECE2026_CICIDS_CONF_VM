"""Shared dual-threshold POLICY calibration (dataset-agnostic core of the FSL-PDP contribution).

Used by every leg's governance script so the policy logic is identical across UNSW / CICIDS / LANL
(only the trust score f_θ and the structural exposure signal E_t are dataset-specific). These are the
exact functions validated on the UNSW leg (2026-06-17).

Decision rule (per row, given the score T and its resource tier t):
    T < d_t            -> Deny
    d_t <= T < a_t     -> Restrict   (= step-up MFA / "access privileges are reduced", NIST §3.3.1)
    T >= a_t           -> direct Allow
where a_t (direct-Allow threshold) is bounded by per-tier AttackAllowRate ≤ aar_max[t] and d_t (Deny
threshold) by per-tier BenignDenyRate ≤ bdr_max[t]. Both calibrated on VALIDATION only.

Policy modes:
    global       (B1) one pooled gate for every tier (conventional single threshold)
    monotonic    (B2) sensitivity-ordered isotonic ladder  -> (allow_monotone=T, deny_monotone=T)
    independent  (B3) per-tier optima, ordering NOT enforced -> (F, F)  (cost-of-monotonicity arm)
    exposure   (PROP) per-tier budgets from structural R_t=f(S_t,E_t); HYBRID (F, T) keeps step-up band
"""
import numpy as np

MODE_MONO = {"monotonic": (True, True), "independent": (False, False), "exposure": (False, True)}


def decide(score, tier, allow, deny):
    a = np.array([allow[int(t)] for t in tier]); d_ = np.array([deny[int(t)] for t in tier])
    out = np.full(len(score), 1, np.int8); out[score >= a] = 0; out[score < d_] = 2
    return out


def calibrate_tiers_constrained(score_va, y_va, tier_va, tiers, aar_max, bdr_max,
                                allow_monotone=True, deny_monotone=True, band_floor=0.0,
                                band_floor_max_frac=1.0):
    """Per-tier 'tuning' (NIST SP 800-207 §3.3.1 score-based TA): maximize benign-Allow s.t.
      allow[t] = (1 − aar_max[t]) quantile of val-ATTACK scores  (smallest thr with AttackAllowRate_t ≤ aar_max[t])
      deny[t]  = bdr_max[t]       quantile of val-BENIGN scores   (largest thr with BenignDenyRate_t ≤ bdr_max[t])
    Two INDEPENDENT constraint-safe isotonic projections: allow_monotone → running-max from bottom (raising
    allow only TIGHTENS AAR); deny_monotone → running-min from top (lowering deny only LOOSENS BDR, and is
    what OPENS the step-up band). deny clipped ≤ allow. No-attack tier falls back to benign median for allow."""
    allow_raw, deny_raw = {}, {}
    for t in tiers:
        sb = score_va[(y_va == 0) & (tier_va == t)]
        sa = score_va[(y_va == 1) & (tier_va == t)]
        deny_raw[t] = float(np.quantile(sb, float(np.clip(bdr_max[t], 0.0, 1.0)))) if len(sb) else 0.0
        # no val attacks at this tier => NO attack-allow constraint => be PERMISSIVE (allow = deny floor),
        # never restrict legitimate users on a tier with no observed threat (else benign friction explodes).
        allow_raw[t] = (float(np.quantile(sa, float(np.clip(1.0 - aar_max[t], 0.0, 1.0)))) if len(sa)
                        else deny_raw[t])
    allow, deny = {}, {}
    if allow_monotone:
        prev = None
        for t in sorted(tiers):
            allow[t] = allow_raw[t] if prev is None else max(allow_raw[t], prev); prev = allow[t]
    else:
        allow = dict(allow_raw)
    if deny_monotone:
        nxt = None
        for t in sorted(tiers, reverse=True):
            deny[t] = deny_raw[t] if nxt is None else min(deny_raw[t], nxt); nxt = deny[t]
    else:
        deny = dict(deny_raw)
    for t in tiers:
        allow[t] = float(np.clip(allow[t], 0.0, 1.0))
        # band_floor (2026-08-09): guarantee a minimum step-up band. With the default 0.0 this is
        # the original rule `deny = min(deny, allow)` exactly (bit-for-bit backwards compatible).
        # With band_floor > 0 the Restrict band is never narrower than band_floor, so step-up MFA
        # stays reachable on every tier — the BDR budget still binds wherever it is tighter.
        # PER-TIER EFFECTIVE FLOOR (2026-08-11). A single global band_floor can be WIDER
        # than a tier's entire allow threshold, in which case `allow - band_floor` goes
        # negative, deny clips to 0.0, and DENY BECOMES UNREACHABLE at that tier.
        # Measured on CICIDS: allow[T1] = 0.036 < band_floor 0.20 -> deny[T1] = 0.000, so
        # 99.6% of tier-1 attacks were Restricted and NONE could be denied; that single
        # tier accounted for ~96% of all restricted attacks and dragged attack hard-Deny
        # from 94.2% down to 23.0%.
        # The floor is meant to GUARANTEE a step-up band, not to erase Deny. So cap it at
        # a fraction of the tier's own allow threshold: the band is still floored where
        # there is room, and Deny stays reachable everywhere.
        eff_floor = min(band_floor, allow[t] * band_floor_max_frac)
        deny[t] = float(np.clip(min(deny[t], allow[t] - eff_floor), 0.0, 1.0))
    return allow, deny, allow_raw, deny_raw


def calibrate_global(score_va, y_va, tiers, aar_g, bdr_g):
    """B1: a single (allow,deny) gate for every tier — no resource differentiation. Pooled val:
    allow = (1−aar_g) quantile of all val-ATTACK; deny = bdr_g quantile of all val-BENIGN."""
    sa = score_va[y_va == 1]; sb = score_va[y_va == 0]
    a = float(np.quantile(sa, float(np.clip(1.0 - aar_g, 0.0, 1.0)))) if len(sa) else 0.5
    d = float(np.quantile(sb, float(np.clip(bdr_g, 0.0, 1.0)))) if len(sb) else 0.0
    a = float(np.clip(a, 0.0, 1.0)); d = float(np.clip(min(d, a), 0.0, 1.0))
    return {t: a for t in tiers}, {t: d for t in tiers}, {t: a for t in tiers}, {t: d for t in tiers}


def exposure_budgets(tiers, E, aar_hi, aar_lo, bdr_lo, bdr_hi, w_s, w_e):
    """Map structural risk R_t = f(S_t, E_t) → per-tier (aar_max, bdr_max). S_t = tier rank in [0,1].
    aar_max[t] = aar_hi − (aar_hi−aar_lo)·(w_S·S_t + w_E·E_t)   (tightens with sensitivity OR exposure
    → exposed-but-low-sensitivity tier earns a strict budget = principled NON-monotonicity).
    bdr_max[t] = bdr_lo + (bdr_hi−bdr_lo)·S_t                    (usability slack grows with criticality)."""
    ts = sorted(tiers); n = len(ts)
    S = {t: (i / (n - 1) if n > 1 else 0.0) for i, t in enumerate(ts)}
    aar, bdr = {}, {}
    for t in tiers:
        rho = min(max(w_s * S[t] + w_e * E[t], 0.0), 1.0)
        aar[t] = round(aar_hi - (aar_hi - aar_lo) * rho, 4)
        bdr[t] = round(bdr_lo + (bdr_hi - bdr_lo) * S[t], 4)
    return aar, bdr, S


def normalize_exposure(raw):
    """Min-max normalize a per-tier raw exposure dict to [0,1] (higher = more exposed)."""
    vals = list(raw.values()); lo, hi = min(vals), max(vals)
    return {t: ((raw[t] - lo) / (hi - lo) if hi > lo else 0.0) for t in raw}


def tier_action_rates(y, d, tier, tiers):
    """Per-tier graduated-response breakdown: Allow/Restrict/Deny rates split by benign vs attack."""
    out = {}
    for t in tiers:
        m = tier == t; yt = y[m]; dt = d[m]
        ben = yt == 0; atk = yt == 1; nb = int(ben.sum()); na = int(atk.sum())
        def rates(mask, n):
            dd = dt[mask]
            return dict(n=int(n),
                        allow=round(float((dd == 0).mean()), 4) if n else None,
                        restrict=round(float((dd == 1).mean()), 4) if n else None,
                        deny=round(float((dd == 2).mean()), 4) if n else None)
        out[t] = dict(n=int(m.sum()), n_benign=nb, n_attack=na,
                      benign=rates(ben, nb), attack=rates(atk, na))
    return out


def temporal_stability(score, y, tier, order_key, tiers, allow, deny, k=5):
    """Threshold stability across k equal-count temporal sub-windows of the (future) test split,
    ordered by `order_key`. Reports per-tier ADR (attack not-Allow), UAR (benign Allow), BDR (benign
    hard-Deny) per window + drift (max−min) per tier."""
    idx = np.argsort(order_key, kind="stable")
    bnd = np.linspace(0, len(idx), k + 1).astype(int)
    wins = []
    for w in range(k):
        sel = idx[bnd[w]:bnd[w + 1]]
        sc, yy, tt = score[sel], y[sel], tier[sel]
        dd = decide(sc, tt, allow, deny)
        per = {}
        for t in tiers:
            mt = tt == t; ben = mt & (yy == 0); atk = mt & (yy == 1)
            nb, na = int(ben.sum()), int(atk.sum())
            per[str(t)] = dict(
                n=int(mt.sum()), n_attack=na, n_benign=nb,
                ADR=round(float((dd[atk] != 0).mean()), 4) if na else None,
                UAR=round(float((dd[ben] == 0).mean()), 4) if nb else None,
                BDR=round(float((dd[ben] == 2).mean()), 4) if nb else None)
        wins.append(per)
    drift = {}
    for t in tiers:
        for met in ("ADR", "UAR", "BDR"):
            vals = [wins[w][str(t)][met] for w in range(k) if wins[w][str(t)][met] is not None]
            drift.setdefault(str(t), {})[met] = (round(max(vals) - min(vals), 4) if vals else None)
    return wins, drift
