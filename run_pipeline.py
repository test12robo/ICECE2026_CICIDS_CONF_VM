"""CICIDS2017 MULTI-DAY — dual-threshold PDP with genuine MULTI-TIER attacks.

DAY SET (decided 2026-08-09, applied 2026-08-11): Tuesday + Wednesday + Friday-PortScan.
  Tuesday        FTP/SSH-Patator on :21/:22 -> 13,834 Tier-3 attacks (the only real Tier-3 volume)
  Wednesday      DoS -> Tier-2 volume, and the base paper's own day (comparability)
  Fri-PortScan   spans Tiers 1-4; the ONLY file carrying Tier-4 attacks
Together these are the only combination that populates every tier, so the per-tier dual-threshold
ladder is actually exercised. Friday-Morning (Bot) was dropped: it is Tier-1 only and adds no tier
coverage. Thursday remains excluded — it drops C .82->.58 and D .79->.68 because HTTP/HTTPS web
attacks mimic benign ports and TCP stacks (ablation: metrics_4day_thu.json).

Split = PER-FILE ROW-ORDER 60/20/20 (each day split in file order, then pooled). CICFlowMeter writes
flows in completion order, so this is chronological WITHIN a day but not across the pooled set — the
sort key is (source_ip, _file, _dt, event_id), i.e. file index outranks timestamp. State it that way;
do not call the pooled set strictly chronological.

CORRECTION 2026-08-11: an earlier version of this docstring claimed Mon/Tue/Thu were the stripped
79-column "ML-CSV" variant lacking Source IP and Timestamp. That is FALSE — all eight day-files in
dataset/ are the 85-column GeneratedLabelledFlows variant with both columns present (verified).
Tuesday was therefore excluded for no valid reason.

Run:  python run_pipeline.py
"""
import os

# Pin threads BEFORE importing xgboost/numpy/polars. Without this the external-IDS booster's
# subsample RNG (subsample=0.8/colsample_bytree=0.8, the only sampling model in this pipeline)
# is partitioned by the machine's core count and R is not reproducible across hosts -- the
# same failure UNSW's step7_external_risk.py documented and fixed on 2026-08-11 (14-thread
# laptop vs. 48-core VM gave DIFFERENT output from the same code/seed/inputs). This file had
# no thread-pinning at all before 2026-08-23 and relied solely on --ids-model freezing to
# sidestep the issue; added to close that gap and match UNSW's fix.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "14")

import json, sys
from pathlib import Path
import numpy as np, polars as pl, xgboost as xgb

HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset"          # raw CICIDS2017 CSVs live here
sys.path.insert(0, str(HERE))
import policy_calib as pcal      # shared dual-threshold policy core
import pdp_core as gov           # vendored trust-equation primitives
import trajectory_kernel as trk  # vendored sequential EMA kernel (from LANL phase8), 2026-08-21
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 3-DAY SUBSET (Wed-DoS + Fri-PortScan + Fri-Bot): the more-stationary configuration where the per-tier
# constrained ladder works and weights are factor-dominant (B.20/D.15, D-fingerprint .79, wR .35).
# PortScan spans Tiers 1-4 so every tier is exercised. (Full 8-day week archived in *.fullweek.bak_*.)
FILES = [                                     # chronological
    "Tuesday-WorkingHours.pcap_ISCX.csv",              # FTP/SSH-Patator on :21/:22 -> 13,834 Tier-3 attacks
    "Wednesday-workingHours.pcap_ISCX.csv",            # DoS -> Tier-2 volume; base-paper comparability
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",   # spans Tiers 1-4; only file with Tier-4 attacks
]
# Thursday-ablation check (metrics_4day_thu.json): adding Thu Web-Attacks + Infiltration drops C .82->.58
# and D .79->.68 (HTTP/HTTPS web attacks mimic benign ports + TCP stacks), calib drops D's weight .15->.10
# and lifts wR .35->.40, and the constrained ladder degrades (monotonic F1 .82->.67). Thursday EXCLUDED.
PORT_TO_TIER = {int(k): v for k, v in json.loads((HERE / "configs/port_to_tier.json").read_text()).items()}
LAMBDA = 0.05; SEED = 42

# Correct the 12-hour clock (see the block in main()). Set False only to reproduce
# the pre-2026-08-11 result, which was computed on mis-ordered timestamps.
FIX_12H_CLOCK = True

# Minimum step-up (Restrict) band width, enforced as deny[t] <= allow[t] - BAND_FLOOR.
# Added 2026-08-11 for parity with the UNSW leg, which has used 0.20 since 2026-08-09.
#
# WHY IT IS NEEDED HERE: policy_calib defaults band_floor=0.0 and this runner never
# passed it, so CICIDS had an essentially degenerate step-up band. Measured ladders:
#   pre-parse-fix   T1 0.000  T2 0.004  T3 0.000  T4 0.000   (benign Restrict 0.82%)
#   post-parse-fix  T1 0.000  T2 0.000  T3 0.000  T4 0.000   (benign Restrict 0.00%)
# i.e. the "graduated response" rested on a 0.004-wide band at ONE tier, and once the
# timestamps were corrected the score separation closed it entirely. band_floor makes
# the step-up band structural rather than incidental.
BAND_FLOOR = 0.20

# Cap the effective band floor at this fraction of each tier's own allow threshold,
# so a global floor can never drive that tier's deny to 0 and make Deny unreachable.
# 1.0 reproduces the pre-2026-08-11 behaviour exactly.
BAND_FLOOR_MAX_FRAC = 0.5   # LOCKED 2026-08-11.
# Binds ONLY where a global band_floor would exceed half a tier's own allow threshold,
# i.e. only on degenerate tiers. Measured effect on CICIDS: tier 1 (allow 0.036) regains a
# reachable Deny at 0.018, taking attack hard-Deny 23.0% -> 86.7% with benign hard-Deny
# UNCHANGED at 0.11% and ADR/UAR/F1/ROC identical. Tiers 2-4 are untouched because
# min(0.20, 0.699*0.5) = 0.20. Safe as a universal default: UNSW (allow 0.473-0.684) and
# LANL (allow ~0.775-0.804) are unaffected, since 0.5*allow > 0.20 on every one of their tiers.

# Per-tier budgets. AAR_MAX is the attack-Allow ceiling per tier and is THE dial for
# false negatives; BDR_MAX is the benign-Deny ceiling. Overridable from the CLI so the
# trade-off can be swept without editing code.
#   [0.02, 0.02, 0.02, 0.01]      the June-2026 setting (loose, flat across T1-T3)
#   [0.01, 0.007, 0.005, 0.003]   the UNSW graduated vector (tightens with criticality)
AAR_MAX = [0.01, 0.007, 0.005, 0.003]   # LOCKED 2026-08-11: UNSW's graduated vector.
# Adopted for a SHARED budget policy across all three legs, not for the metric: measured on
# CICIDS a 10x tightening (0.02 -> 0.002) moves FN only 1,540 -> 1,484 and leaves FP, UAR and
# the benign split byte-identical, because tier 2's allow threshold is pinned at 0.699
# regardless of budget. CICIDS FN is separability-limited at T2, not threshold-limited
# (unlike UNSW, where the same change halved FN). See the sweep in _REFERENCE/.
BDR_MAX = [0.02, 0.04, 0.08, 0.15]
SUFFIX = ""

# --- CONFERENCE ADDITIONS (2026-08-27) --------------------------------------------
# B1 GLOBAL-GATE budgets. run_pipeline already evaluated a "global" policy mode on every
# run, but with these two numbers HARDCODED at the call site (0.02 / 0.08). The ICECE
# conference arm needs 0.01 / 0.005 -- the same operator budgets the UNSW arm uses -- so
# they become settable. Defaults preserve the locked leg's behaviour exactly.
AAR_GLOBAL = 0.02   # attack-Allow budget, pooled over all tiers
BDR_GLOBAL = 0.08   # benign-Deny  budget, pooled over all tiers
# Weight-simplex floor. calibrate6 hardcoded w_min=0.10 via its default; exposing it keeps
# parity with the UNSW arm's --w-min sweep (E8).
W_MIN = 0.10
# DAY SUBSET. FILES above is the locked 3-day set. --day restricts it (conference arm runs
# Wednesday alone -- the base paper's own CICIDS day).
DROP_IDENTITY = False
DUMP_FULLDAY = False
FULLDAY = None
# Path to a frozen out/model_external_ids.json. None = retrain the external IDS.
# Set by --ids-model. See the note at the training site for why this matters on this leg.
IDS_MODEL = None
# ABLATION (2026-08-21), default "f_theta" = the lock. "formula" bypasses the monotone XGBoost
# regressor entirely and scores/decides directly off the closed-form additive trust formula
# (T_add, min-max scaled the same way f_theta's OWN training target is) -- i.e. what the PDP
# achieves with NO machine learning distillation step at all. `R` (the external IDS -- a
# SEPARATE XGBoost classifier feeding INTO the formula as one input) is unaffected either way;
# this flag only removes the f_theta distillation layer, not R. Set by --policy-score.
POLICY_SCORE = "f_theta"

# ABLATION (2026-08-24): f_theta's own XGBoost regularization, opt-in via --ftheta-*.
# Defaults reproduce the exact locked hyperparameters (subsample=1.0, colsample_bytree=1.0,
# min_child_weight=10, reg_lambda=1.0) -- unmodified `python run_pipeline.py` is byte-identical
# to before. Motivation: f_theta is currently LESS regularized than R itself (R uses
# subsample=0.8/colsample_bytree=0.8/min_child_weight=50/reg_lambda=10), and the Tier-2
# (Internal, DoS GoldenEye) benign false positives that make f_theta trail the formula on FP
# at the B2 lock (562 vs 483, see FSL_PDP_RESULTS_LEDGER.md) are hypothesized to be
# variance/overfit near that tier's threshold boundary, not a real discrimination gap --
# see ftheta_regularization_sweep_cicids.py.
FTHETA_SUBSAMPLE = 1.0
FTHETA_COLSAMPLE_BYTREE = 1.0
FTHETA_MIN_CHILD_WEIGHT = 10
FTHETA_REG_LAMBDA = 1.0

# How the History factor is expressed. `level` = mean of the source's earlier
# (B+C+D)/3, which is what the 2026-08-11 LOCK uses. Measured on UNSW 2026-08-12: as a
# LEVEL, H structurally duplicates the present whenever a source behaves consistently
# (corr(H, base) = +0.79 there), adding a noisy near-copy of B/C. Expressing it as a
# DEVIATION from the source's own norm makes it complementary instead. Set by --h-mode.
#
# `legacy_ema` ADDED 2026-08-21 (user: port LANL's H system to CICIDS) -- the same
# self-referential EMA LANL locks on: H(t)=alpha*T_init(t-1)+(1-alpha)*H(t-1), where
# T_init needs a NEW R_init risk-channel (see RINIT_* below, built from CICIDS's OWN 7
# sub-factors -- LANL's PROFILE_FEATURES do not exist on this leg). Uses the vendored
# `trajectory_kernel` (see that file's docstring) rather than the pure-Polars expanding-mean
# path the other 3 modes use, because the self-reference makes this a true sequential
# recurrence, not a vectorisable expression.
#
# *** PROMOTED TO DEFAULT 2026-08-21 (user decision, after measuring with frozen R):
# ADR .9521->.9741, F1 .9663->.9777, FN 1460->790, UAR/ROC/FP unchanged. Caveat, disclosed
# not hidden: factor_auc.H is IDENTICAL to `level`'s (.969 both) -- CICIDS's attack traffic
# is 100% from ONE source_ip (172.16.0.1, verified by direct count, not assumed), so ANY
# per-source_ip statistic ranks near-perfectly regardless of formula; the gain is a
# calibration/threshold-interaction effect (legacy_ema's H has a different value
# DISTRIBUTION, corr(H,base)=+0.92, that happens to land better against the recalibrated
# tier ladder), not evidence of extracting genuinely new behavioural information. User
# decision: both UNSW and CICIDS are lab-generated with limited source diversity, this
# leak is a property of the dataset, not fixable by re-engineering H -- accept it and use
# the better-performing formula. `level` remains available via --h-mode level. ***
H_MODE = "legacy_ema"

# --- legacy_ema's R_init risk channel (2026-08-21) -- see trajectory_kernel.py's docstring ---
# Same formula/constants as LANL's phase7_risk_channels.py (vendored, not retyped):
#     d(x) = mean_f( |x_f - median_f| / max(IQR_f, RINIT_IQR_FLOOR) )
#     R_normal_distance = clip(d_normal/RINIT_SCALE, 0, 1)
#     R_attack_match    = clip(1 - d_attack/RINIT_SCALE, 0, 1)
#     R_init            = clip(RINIT_W_ATTACK*R_attack_match + RINIT_W_NORMAL*R_normal_distance, 0, 1)
# Deliberately a DIFFERENT constant from the admission gate's own IQR_FLOOR (0.01, line ~139)
# -- that gates sub-factor admission; this is R_init's own distance-denominator floor, LANL's
# actual value there. RINIT_PROFILE_FEATURES is CICIDS's OWN 7 sub-factors -- LANL's 11-feature
# list (M, AF, PR, ...) has no CICIDS equivalent and is NOT used here; only the robust-distance
# formula and its 4 constants are shared cross-leg.
RINIT_PROFILE_FEATURES = ["sf_access_rate", "sf_service_usage", "sf_dev_initwin",
                          "sf_dev_minseg", "sf_dev_downup", "sf_dest_context", "sf_access_time"]
RINIT_IQR_FLOOR = 0.05
RINIT_SCALE     = 4.0
RINIT_W_ATTACK  = 0.55
RINIT_W_NORMAL  = 0.45

# Declared factor -> sub-factor decomposition, serialised into every metrics JSON so the
# thesis's instantiation table can be checked against what actually ran. MUST match the
# arithmetic in main(); a startup assertion enforces that the columns exist, and the
# per-sub-factor AUCs are printed so a degenerate one cannot hide inside a factor mean.
# `I` is empty: CICFlowMeter exposes no identity telemetry, so I takes the 0.5 neutral.
SUBFACTORS = {
    "I": [],
    "B": ["sf_access_rate", "sf_service_usage"],
    "D": ["sf_dev_initwin", "sf_dev_minseg", "sf_dev_downup"],
    "C": ["sf_dest_context", "sf_access_time"],
}

# ABLATION ONLY (2026-08-12), default OFF -- the locked result uses the sub-factors AS
# BUILT. Measuring showed 3 of 7 are oriented BACKWARDS on this dataset (trust-AUC < 0.5):
# sf_service_usage 0.427, sf_dev_downup 0.150, sf_access_time 0.480. Because a factor is a
# MEAN, a backwards member drags its factor down and the dilution is invisible in the
# factor-level AUC. UNSW rejects such sub-factors automatically via its IQR+stability
# admission gate; CICIDS has no gate, so this flag exists to QUANTIFY the cost.
#   "flip"  -> replace x with 1-x for the backwards members
#   "drop"  -> exclude them from their factor's mean (the UNSW gate's actual behaviour)
# Set by --reorient. NOT part of any locked run.
REORIENT = None
BACKWARDS_SF = ["sf_service_usage", "sf_dev_downup", "sf_access_time"]

# --- admission gate (ported from UNSW 2026-08-13) ------------------------------------
# Same rule and same constants as UNSW_PDP_V2/steps/step6_factors.py:120-121 and V1's
# phase-6 gate. Both criteria are LABEL-FREE and use train/val only:
#     iqr_ok = IQR(train[f])                    >= IQR_FLOOR
#     stable = |mean(train[f]) - mean(val[f])|  <= STABILITY_MAX
#
# WHAT IT DOES AND DOES NOT DO -- do not over-read this:
#   * DROPS dead sub-factors (no spread => no information regardless of orientation).
#     On UNSW this rejected NR at IQR exactly 0.0000.
#   * Does NOT drop BACKWARDS sub-factors. Direction is not a criterion. They are
#     flagged loudly and keep contributing to their factor's mean until someone decides
#     to re-orient them (--reorient), which is a separate judgement.
#
# `test_auc_trust` is RECORDED for diagnosis but is deliberately NOT a gate criterion:
# selecting sub-factors by test AUC would be test-set selection.
IQR_FLOOR = 0.01
STABILITY_MAX = 0.10
# what Layer 2 actually averaged; set in main(), serialised as subfactor_map_effective
_USE_EFFECTIVE: dict = {}
# "report"  = compute + serialise the verdict, change NOTHING  <- DEFAULT, = the lock
# "enforce" = exclude rejected sub-factors from their factor's mean
#
# *** USER DECISION 2026-08-13: ON THIS LEG THE GATE IS A REPORTED DIAGNOSTIC. DO NOT ENFORCE.
# Measured (frozen IDS, run verified inert -- 444/444 leaves identical to the lock):
#     sub-factor          IQR     |dmean|   AUC_trust   verdict
#     sf_access_rate      .8097   .0539     .9375       pass
#     sf_service_usage    .8147   .1923     .4269       REJECT
#     sf_dev_initwin      .4482   .0469     .6795       pass
#     sf_dev_minseg       .0171   .0023     .8440       pass
#     sf_dev_downup      1.0000   .2166     .1496       REJECT
#     sf_dest_context     .0539   .1333     .8683       REJECT   <-- the problem
#     sf_access_time      .0103   .0337     .4805       pass     <-- backwards, passes
# NOTHING fails the IQR floor here: CICIDS has no dead sub-factors, so all three rejections
# are STABILITY. And the gate rejects `sf_dest_context`, the STRONGEST member of C
# (AUC .868), while keeping the BACKWARDS `sf_access_time` (.4805) -- so enforcing would
# leave C as a single backwards sub-factor. The gate tests SPREAD and STABILITY, not
# INFORMATION; on this leg it diagnoses, it does not fix. The backwards members are a
# separate decision (--reorient), not a gate decision. ***
GATE_MODE = "report"
L3 = ["flow_packets_s", "flow_bytes_s", "fwd_packets_s", "bwd_packets_s", "flow_duration", "flow_iat_mean",
      "idle_mean", "active_mean", "syn_flag_count", "rst_flag_count", "psh_flag_count", "total_fwd_packets",
      "flow_iat_std", "flow_iat_max", "flow_iat_min", "fwd_iat_total", "fwd_iat_mean", "fwd_iat_std",
      "fwd_iat_max", "fwd_iat_min", "bwd_iat_total", "bwd_iat_mean", "bwd_iat_std", "bwd_iat_max", "bwd_iat_min",
      "total_backward_packets", "total_length_of_bwd_packets", "bwd_packet_length_max", "bwd_packet_length_mean",
      "bwd_packet_length_std", "packet_length_mean", "packet_length_std", "packet_length_variance",
      "max_packet_length", "average_packet_size", "avg_bwd_segment_size", "init_win_bytes_forward", "act_data_pkt_fwd"]
DEVICE = ["init_win_bytes_backward", "min_seg_size_forward", "down_up_ratio"]


def norm(c):
    c = c.strip().lower().replace("/", "_").replace(" ", "_").replace("-", "_")
    while "__" in c: c = c.replace("__", "_")
    return c


def logfreq(df, col):
    vc = df.group_by(col).len(); cnt = dict(zip(vc[col].to_list(), vc["len"].to_list()))
    mx = np.log1p(max(cnt.values()))
    return {k: float(np.clip(np.log1p(v) / mx, 0, 1)) for k, v in cnt.items()}


def load_file(fname, fidx):
    raw = pl.read_csv(str(DATA / fname), infer_schema_length=0, encoding="utf8-lossy")
    raw.columns = [norm(c) for c in raw.columns]
    ren = {"timestamp": "ts"}
    raw = raw.rename({k: v for k, v in ren.items() if k in raw.columns})
    keep = ["source_ip", "destination_port", "ts", "label"] + L3 + DEVICE
    raw = raw.select([c for c in keep if c in raw.columns])
    numeric = [c for c in (L3 + DEVICE + ["destination_port"]) if c in raw.columns]
    raw = raw.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in numeric])
    raw = raw.with_columns([pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(0.0)
                            .fill_null(0.0).alias(c) for c in numeric])
    raw = raw.filter(pl.col("label").is_not_null() & (pl.col("label").str.strip_chars() != ""))
    # per-file row-order temporal split (CICFlowMeter writes flows chronologically)
    n = raw.height
    pos = np.arange(n)
    sp = np.where(pos < int(0.6 * n), "train", np.where(pos < int(0.8 * n), "val", "test"))
    return raw.with_columns(pl.Series("rsplit", sp), pl.lit(fidx).alias("_file"))


def main():
    global FULLDAY
    df = pl.concat([load_file(f, i) for i, f in enumerate(FILES)], how="vertical_relaxed")
    df = df.with_columns(
        pl.arange(0, pl.len()).alias("event_id"),
        (pl.col("label").str.strip_chars().str.to_uppercase() != "BENIGN").cast(pl.Int8).alias("y"),
        pl.col("destination_port").cast(pl.Int64, strict=False).fill_null(0)
          .replace_strict(PORT_TO_TIER, default=1, return_dtype=pl.Int64).alias("tier"))

    # parse ts best-effort (D/M/YYYY H:MM); fallback to event_id order for hour/dt
    _ts = pl.col("ts").str.strip_chars()
    df = df.with_columns(pl.coalesce([
        _ts.str.to_datetime(format=f, strict=False) for f in
        ("%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %I:%M:%S %p", "%d/%m/%Y %I:%M %p")]).alias("_dt"))

    # ---- 12-HOUR CLOCK CORRECTION (added 2026-08-11) --------------------------
    # CICIDS2017 writes a 12-hour clock with NO AM/PM token (e.g. "5/7/2017 1:19").
    # "%H:%M" is tried first above, so 1 PM parses as 01:00 and NO row ever lands
    # in 13:00-23:59. Verified by diagnose_timestamps.py on all four candidate
    # files: the parsed hour histogram maxes out at 12, and hours 0, 6 and 7 are
    # EMPTY in every file -- so mapping 1-7 -> +12h is unambiguous.
    #
    #   Wednesday     32.8% of rows shift   01:00-12:59 -> 08:42-17:10
    #   Fri-PortScan  100%                  01:00-03:29 -> 13:00-15:29
    #   Fri-Morning   0%                    08:59-12:59 (genuinely a morning file)
    #   Tuesday       49.5%                 01:00-12:59 -> 08:53-17:00
    #
    # This matters because `_acc`, `_svc`, `_hour` and `dt_hours` are all computed
    # on the sorted order, so they feed B, C, decay and H. Leaving it unfixed
    # silently mis-orders a third to a half of every afternoon-spanning file.
    if FIX_12H_CLOCK:
        df = df.with_columns(
            pl.when(pl.col("_dt").dt.hour().is_between(1, 7))
              .then(pl.col("_dt") + pl.duration(hours=12))
              .otherwise(pl.col("_dt")).alias("_dt"))
        _shift = int(df.filter(pl.col("_dt").dt.hour() >= 13).height)
        print(f"[ts-fix] 12h clock corrected: {_shift:,} rows now in 13:00-23:59 "
              f"({_shift/max(df.height,1):.1%}); was 0 before the fix")
    print(f"[data] {df.height:,} flows  attack_rate={float(df['y'].mean()):.3f}  ts_parsed={float(df['_dt'].is_not_null().mean()):.2f}")
    for sp in ("train", "val", "test"):
        d = df.filter(pl.col("rsplit") == sp)
        g = d.group_by("tier").agg(pl.col("y").sum().alias("a"), pl.len().alias("n")).sort("tier")
        print(f"  {sp}: n={d.height:,} atk_rate={float(d['y'].mean()):.3f}  per-tier(atk/n)=" +
              " ".join(f"T{r['tier']}:{r['a']}/{r['n']}" for r in g.iter_rows(named=True)))

    # causal-within-source features (order by file then ts then event_id)
    df = df.sort(["source_ip", "_file", "_dt", "event_id"]).with_columns(
        pl.int_range(0, pl.len()).over("source_ip").alias("_acc"),
        pl.int_range(0, pl.len()).over(["source_ip", "destination_port"]).alias("_svc"),
        pl.col("_dt").dt.hour().fill_null(0).alias("_hour"),
        ((pl.col("_dt") - pl.col("_dt").shift(1).over("source_ip")).dt.total_seconds() / 3600.0)
        .fill_null(0.0).clip(0.0, None).alias("dt_hours"))
    ben = df.filter((pl.col("rsplit") == "train") & (pl.col("y") == 0))
    p95 = {c: (float(ben[c].quantile(0.95)) or 1.0) for c in ["_acc", "_svc"]}
    p95du = float(ben["down_up_ratio"].quantile(0.95)) or 1.0
    lf_dport, lf_hour = logfreq(ben, "destination_port"), logfreq(ben, "_hour")
    lf_win, lf_seg = logfreq(ben, "init_win_bytes_backward"), logfreq(ben, "min_seg_size_forward")
    # ---- Layer 1: named sub-factors --------------------------------------------------
    # PURE REFACTOR 2026-08-12: these expressions were previously inlined inside the
    # factor definitions below. The arithmetic is UNCHANGED (verified: 333/333 result
    # leaves identical). They are named so the factor -> sub-factor decomposition is
    # auditable and the thesis's instantiation table can be checked against the code.
    #
    # NAMING: after the trust-attribute taxonomy in UNIVERSAL_SUBFACTORS.md, NOT after
    # UNSW's AP/DA/DT/NR/VA/TR. Those are UNSW's own variables and the correspondence is
    # only attribute-level -- e.g. UNSW's AP is a packet RATE while `_acc` is a cumulative
    # COUNT. Claiming "CICIDS computes AP" would overstate it; claiming both instantiate
    # the ACCESS RATE attribute is exactly right. See FACTOR_CONSTRUCTION_ANALYSIS.md 4b.
    #
    # Every sub-factor is a [0,1] TRUST score against a benign-TRAIN baseline:
    # 1.0 = typical-benign, 0.0 = far outside it.
    df = df.with_columns(
        # --- Access Rate: cumulative requests by this source, P95-capped -> B
        (1 - (pl.col("_acc") / p95["_acc"]).clip(0, 1)).alias("sf_access_rate"),
        # --- Service Usage (per-entity): this source's requests to this port -> B
        (1 - (pl.col("_svc") / p95["_svc"]).clip(0, 1)).alias("sf_service_usage"),
        # --- Device Fingerprint: TCP-stack signature rarity vs benign -> D
        pl.col("init_win_bytes_backward").replace_strict(
            lf_win, default=0.0, return_dtype=pl.Float64).alias("sf_dev_initwin"),
        pl.col("min_seg_size_forward").replace_strict(
            lf_seg, default=0.0, return_dtype=pl.Float64).alias("sf_dev_minseg"),
        (1 - (pl.col("down_up_ratio") / p95du).clip(0, 1)).alias("sf_dev_downup"),
        # --- Destination Context: destination-port popularity vs benign -> C
        pl.col("destination_port").replace_strict(
            lf_dport, default=0.0, return_dtype=pl.Float64).alias("sf_dest_context"),
        # --- Access Time: hour-of-day typicality vs benign -> C
        pl.col("_hour").replace_strict(
            lf_hour, default=0.0, return_dtype=pl.Float64).alias("sf_access_time"),
    )
    # ---- Layer 1b: ADMISSION GATE (2026-08-13) ---------------------------------------
    # Report-only by default: the verdict is computed and written to admission.json but no
    # sub-factor is excluded, so the locked arithmetic below is untouched.
    _tr = df.filter(pl.col("rsplit") == "train")
    _va = df.filter(pl.col("rsplit") == "val")
    _te = df.filter(pl.col("rsplit") == "test")
    _yte = _te["y"].to_numpy()
    ADMISSION: dict[str, dict] = {}
    print(f"[gate] IQR floor {IQR_FLOOR}  stability max |mean_tr-mean_va| {STABILITY_MAX}  "
          f"mode={GATE_MODE}" + ("   (verdicts recorded, NOT enforced)" if GATE_MODE == "report" else ""))
    print(f"  {'sub-factor':<20}{'IQR':>9}{'|dmean|':>10}{'AUC_tr':>9}{'IQRok':>7}{'stable':>8}   ADMIT")
    print("  " + "-" * 70)
    for _slot, _cols in SUBFACTORS.items():
        for _c in _cols:
            _x = _tr[_c].to_numpy()
            _iqr = float(np.quantile(_x, 0.75) - np.quantile(_x, 0.25))
            _dm = abs(float(_tr[_c].mean()) - float(_va[_c].mean()))
            _auc = float(gov.roc(-_te[_c].to_numpy(), _yte))
            _iqr_ok, _stable = _iqr >= IQR_FLOOR, _dm <= STABILITY_MAX
            _verdict = _iqr_ok and _stable
            _adm = _verdict if GATE_MODE == "enforce" else True
            # keep the UNROUNDED auc as well: `subfactor_auc` in the metrics JSON is quoted to
            # 3 dp and rounding a 4-dp value again is not the same number (sf_access_rate went
            # .937 -> .938 on the first attempt at this refactor). Every consumer rounds ONCE,
            # from the raw value.
            ADMISSION[_c] = dict(slot=_slot, iqr=round(_iqr, 4), stability_delta=round(_dm, 4),
                                 test_auc_trust=round(_auc, 4), test_auc_trust_raw=_auc,
                                 iqr_ok=bool(_iqr_ok),
                                 stable=bool(_stable), gate_verdict=bool(_verdict),
                                 gate_mode=GATE_MODE, admitted=bool(_adm))
            print(f"  {_c:<20}{_iqr:>9.4f}{_dm:>10.4f}{_auc:>9.3f}"
                  f"{('Y' if _iqr_ok else 'n'):>7}{('Y' if _stable else 'n'):>8}   "
                  f"{'YES' if _adm else 'NO'}")
    _rej = [c for c, v in ADMISSION.items() if not v["gate_verdict"]]
    if _rej:
        print(f"[gate] gate REJECTS {_rej}"
              + ("  -> excluded from their factor's mean" if GATE_MODE == "enforce"
                 else "  -> NOT excluded (report mode); rerun with --gate-mode enforce to measure"))
    else:
        print("[gate] all sub-factors pass IQR + stability")
    _bad = {c: v["test_auc_trust"] for c, v in ADMISSION.items()
            if v["admitted"] and v["test_auc_trust"] < 0.5}
    if _bad:
        print(f"[gate] *** {len(_bad)} ADMITTED sub-factor(s) are oriented BACKWARDS "
              f"(AUC_trust < 0.5): {_bad}\n"
              f"       The gate tests SPREAD and STABILITY, not DIRECTION -- these pass it and "
              f"still drag their factor's mean. DISCLOSE, or measure --reorient flip|drop. ***")
    (HERE / f"admission{SUFFIX}.json").write_text(json.dumps(
        {"leg": "CICIDS", "gate_mode": GATE_MODE, "iqr_floor": IQR_FLOOR,
         "stability_max": STABILITY_MAX, "subfactor_map": SUBFACTORS,
         "note": "iqr on train; stability = |mean_train - mean_val|; test_auc_trust is a "
                 "DIAGNOSTIC and deliberately NOT a gate criterion (that would be test-set "
                 "selection). Direction is not gated -- see the BACKWARDS note.",
         "subfactors": ADMISSION}, indent=2), encoding="utf-8")

    # ---- Layer 1c: legacy R_init risk channel (2026-08-21, for --h-mode legacy_ema) --------
    # Vendored formula from LANL's phase7_risk_channels.py; CICIDS-native inputs (RINIT_* above).
    # Fit on TRAIN only (benign vs attack), applied to the whole pooled df -- same frozen-profile
    # pattern LANL uses. Computed unconditionally (cheap, 7 features) so legacy_ema never needs a
    # separate pre-pass; every other --h-mode simply ignores the resulting R_init column.
    _tr_benign = _tr.filter(pl.col("y") == 0)
    _tr_attack = _tr.filter(pl.col("y") == 1)
    _rinit_normal, _rinit_attack = {}, {}
    for _f in RINIT_PROFILE_FEATURES:
        _bn, _at = _tr_benign[_f], _tr_attack[_f]
        _rinit_normal[_f] = {"median": float(_bn.median()),
                             "iqr": float(_bn.quantile(0.75) - _bn.quantile(0.25))}
        _rinit_attack[_f] = {"median": float(_at.median()),
                             "iqr": float(_at.quantile(0.75) - _at.quantile(0.25))}
    print(f"[R_init] profile fit on train: {_tr_benign.height:,} benign / {_tr_attack.height:,} "
          f"attack rows, {len(RINIT_PROFILE_FEATURES)} features "
          f"({', '.join(f.replace('sf_', '') for f in RINIT_PROFILE_FEATURES)})")

    def _rinit_distance(profile):
        terms = [((pl.col(f) - profile[f]["median"]).abs()
                  / max(profile[f]["iqr"], RINIT_IQR_FLOOR)) for f in RINIT_PROFILE_FEATURES]
        total = terms[0]
        for t in terms[1:]:
            total = total + t
        return total / len(terms)

    df = df.with_columns(
        (_rinit_distance(_rinit_normal) / RINIT_SCALE).clip(0.0, 1.0).alias("R_normal_distance"),
        (1.0 - _rinit_distance(_rinit_attack) / RINIT_SCALE).clip(0.0, 1.0).alias("R_attack_match"),
    )
    df = df.with_columns(
        (RINIT_W_ATTACK * pl.col("R_attack_match")
         + RINIT_W_NORMAL * pl.col("R_normal_distance")).clip(0.0, 1.0).alias("R_init"))
    _ri_b = float(df.filter(pl.col("y") == 0)["R_init"].mean())
    _ri_a = float(df.filter(pl.col("y") == 1)["R_init"].mean())
    print(f"[R_init] mean benign={_ri_b:.4f}  mean attack={_ri_a:.4f}  delta={_ri_a - _ri_b:+.4f}"
          + ("" if H_MODE == "legacy_ema" else "   (computed but unused -- current --h-mode "
             f"{H_MODE!r} doesn't read R_init)"))

    # what was ACTUALLY averaged, for the metrics JSON (see subfactor_map_effective)
    global _USE_EFFECTIVE
    _USE_EFFECTIVE = {f: list(c) for f, c in SUBFACTORS.items() if c}

    # ---- Layer 2: factor = mean of its OBSERVED sub-factors ---------------------------
    if REORIENT is None and GATE_MODE != "enforce":
        # LOCKED PATH. Grouping written out explicitly (not sum()/len()) so the float
        # operation order is bit-identical to the pre-refactor inline version.
        df = df.with_columns(
            # Identity: CICFlowMeter exposes no auth telemetry -> renormalisation neutral
            pl.lit(0.5).alias("I"),
            ((pl.col("sf_access_rate") + pl.col("sf_service_usage")) / 2).alias("B"),
            ((pl.col("sf_dev_initwin") + pl.col("sf_dev_minseg")
              + pl.col("sf_dev_downup")) / 3).alias("D"),
            ((pl.col("sf_dest_context") + pl.col("sf_access_time")) / 2).alias("C"),
            (-LAMBDA * pl.col("dt_hours")).exp().alias("decay"))
    else:
        # ABLATION PATH -- REORIENT and/or --gate-mode enforce. Never used by a locked run.
        use = {f: list(cols) for f, cols in SUBFACTORS.items() if cols}
        if REORIENT == "flip":
            df = df.with_columns([(1.0 - pl.col(c)).alias(c) for c in BACKWARDS_SF])
        elif REORIENT == "drop":
            use = {f: [c for c in cols if c not in BACKWARDS_SF] for f, cols in use.items()}
        elif REORIENT is not None:
            raise ValueError(f"--reorient must be flip|drop, got {REORIENT!r}")
        if GATE_MODE == "enforce":
            # the gate's own verdict, independent of REORIENT
            print("[gate] *** WARNING: --gate-mode enforce is an ABLATION on this leg. The "
                  "2026-08-13 decision is REPORT-ONLY: the gate rejects sf_dest_context "
                  "(AUC .868, the strongest member of C) on stability while keeping the "
                  "backwards sf_access_time (.4805), so C collapses toward a single backwards "
                  "sub-factor. See the GATE_MODE block at the top. ***")
            use = {f: [c for c in cols if ADMISSION[c]["gate_verdict"]] for f, cols in use.items()}
        use = {f: cols for f, cols in use.items() if cols}       # renormalisation rule
        _USE_EFFECTIVE = {f: list(c) for f, c in use.items()}
        print(f"[ablation] REORIENT={REORIENT} GATE_MODE={GATE_MODE}; factor membership now "
              + " ".join(f"{f}<-{len(c)}" for f, c in use.items()))
        exprs = [pl.lit(0.5).alias("I")]
        for f, cols in use.items():
            e = pl.col(cols[0])
            for c in cols[1:]:
                e = e + pl.col(c)
            exprs.append((e / len(cols)).alias(f))
        # A factor whose every sub-factor was dropped falls back to the 0.5 neutral.
        for f in ("B", "D", "C"):
            if f not in use:
                exprs.append(pl.lit(0.5).alias(f))
        exprs.append((-LAMBDA * pl.col("dt_hours")).exp().alias("decay"))
        df = df.with_columns(exprs)
    # The declared decomposition must match what was actually built. Cheap, and it stops
    # SUBFACTORS drifting away from the arithmetic above (which is what the paper quotes).
    _declared = [c for cols in SUBFACTORS.values() for c in cols]
    _missing = [c for c in _declared if c not in df.columns]
    if _missing:
        raise RuntimeError(f"SUBFACTORS declares columns that were never built: {_missing}")
    print(f"[subfactors] {len(_declared)} named, mapped as "
          + " ".join(f"{f}<-{'+'.join(c.replace('sf_','') for c in cols)}"
                     for f, cols in SUBFACTORS.items() if cols))
    # ---- History factor H --------------------------------------------------------------
    # base_t = (B+C+D)/3 for the current flow;  mu_t = mean of this source's EARLIER base.
    # H_MODE (see the constant at the top) selects how History is expressed:
    #   level      H = mu_t                          <- the 2026-08-11 LOCK uses this
    #   signed_dev H = clip(0.5 + (base_t - mu_t))   <- deviation from the source's own norm
    #   abs_dev    H = 1 - |base_t - mu_t|
    #   legacy_ema H = alpha*T_init(t-1)+(1-alpha)*H(t-1)   <- LANL's H system, ported 2026-08-21
    # Cold start (no earlier flow for this source) = 0.5, the renormalisation neutral.
    if H_MODE == "legacy_ema":
        # Sequential self-referential recursion -- not vectorisable, needs the kernel.
        # df is ALREADY sorted (source_ip, _file, _dt, event_id) from line ~248 and nothing
        # since has reordered it (only .with_columns/.filter) -- re-sort explicitly anyway
        # (cheap insurance, not an assumption) rather than trust that implicitly, the same
        # discipline the LANL AC-join bug just taught: verify order, don't assume it.
        df = df.sort(["source_ip", "_file", "_dt", "event_id"])
        n = df.height
        time_arr = df["_dt"].dt.epoch("s").fill_null(0).to_numpy().astype(np.int64, copy=False)
        entity_arr = df["source_ip"].to_numpy()
        _, entity_code = np.unique(entity_arr, return_inverse=True)
        entity_code = entity_code.astype(np.int32, copy=False)
        n_groups = int(entity_code.max()) + 1 if n else 0
        print(f"[H] legacy_ema: {n_groups:,} unique source_ip groups, {n:,} rows")

        orig_idx = np.arange(n, dtype=np.int64)
        perm = np.lexsort((orig_idx, time_arr, entity_code))
        sorted_entity_code = entity_code[perm]
        g_starts, g_ends = trk.group_boundaries(sorted_entity_code)

        I_arr = np.ascontiguousarray(df["I"].to_numpy(), dtype=np.float32)
        B_arr = np.ascontiguousarray(df["B"].to_numpy(), dtype=np.float32)
        D_arr = np.ascontiguousarray(df["D"].to_numpy(), dtype=np.float32)
        C_arr = np.ascontiguousarray(df["C"].to_numpy(), dtype=np.float32)
        R_init_arr = np.ascontiguousarray(df["R_init"].to_numpy(), dtype=np.float32)

        # Single-shot run over the whole pooled df (train+val+test already concatenated,
        # unlike LANL's per-partition-with-JSON-carryover staging) -- every group starts cold.
        state_H_init      = np.full(n_groups, trk.H_COLD_START, dtype=np.float32)
        state_T_init_init = np.zeros(n_groups, dtype=np.float32)
        state_time_prev   = np.zeros(n_groups, dtype=np.int64)
        state_has_prev    = np.zeros(n_groups, dtype=np.bool_)
        state_sum_base    = np.zeros(n_groups, dtype=np.float64)
        state_cnt_base    = np.zeros(n_groups, dtype=np.int64)

        H_out_s      = np.empty(n, dtype=np.float32)
        T_base_out_s = np.empty(n, dtype=np.float32)
        T_init_out_s = np.empty(n, dtype=np.float32)
        dt_out_s     = np.empty(n, dtype=np.float32)
        state_H_final        = np.empty(n_groups, dtype=np.float32)
        state_T_init_final   = np.empty(n_groups, dtype=np.float32)
        state_time_final     = np.empty(n_groups, dtype=np.int64)
        state_sum_base_final = np.empty(n_groups, dtype=np.float64)
        state_cnt_base_final = np.empty(n_groups, dtype=np.int64)

        trk.trajectory_kernel(
            np.ascontiguousarray(time_arr[perm]), np.ascontiguousarray(entity_code[perm]),
            np.ascontiguousarray(I_arr[perm]), np.ascontiguousarray(B_arr[perm]),
            np.ascontiguousarray(D_arr[perm]), np.ascontiguousarray(C_arr[perm]),
            np.ascontiguousarray(R_init_arr[perm]),
            g_starts, g_ends,
            state_H_init, state_T_init_init, state_time_prev, state_has_prev,
            state_sum_base, state_cnt_base,
            np.float64(trk.ALPHA), np.float64(trk.GAMMA), np.float64(LAMBDA),
            np.float64(trk.H_COLD_START), np.int64(trk.H_MODE_CODES["legacy_ema"]),
            H_out_s, T_base_out_s, T_init_out_s, dt_out_s,
            state_H_final, state_T_init_final, state_time_final,
            state_sum_base_final, state_cnt_base_final,
        )
        H_out = np.empty(n, dtype=np.float32); H_out[perm] = H_out_s
        df = df.with_columns(pl.Series("H", H_out, dtype=pl.Float32))
        _b_ref = (pl.col("B") + pl.col("C") + pl.col("D")) / 3
        df = df.with_columns(_b_ref.alias("_b"))
        _hc = float(df.select(pl.corr("H", "_b")).item())
        print(f"[H] mode=legacy_ema  mean={float(df['H'].mean()):.4f}  "
              f"std={float(df['H'].std()):.4f}  corr(H, (B+C+D)/3)={_hc:+.4f}  "
              f"alpha={trk.ALPHA} gamma={trk.GAMMA} lambda={LAMBDA}")
    else:
        base = (pl.col("B") + pl.col("C") + pl.col("D")) / 3
        df = df.with_columns(base.alias("_b")).with_columns(pl.col("_b").shift(1).over("source_ip").alias("_pb"))
        df = df.with_columns(
            (pl.col("_pb").fill_null(0.0).cum_sum().over("source_ip")
             / pl.col("_pb").is_not_null().cast(pl.Float64).cum_sum().over("source_ip").clip(1, None)
             ).alias("_mu"))
        _has_past = pl.col("_pb").is_not_null().cum_sum().over("source_ip") > 0
        if H_MODE == "level":
            _h = pl.col("_mu")
        elif H_MODE == "signed_dev":
            _h = (0.5 + (pl.col("_b") - pl.col("_mu"))).clip(0.0, 1.0)
        elif H_MODE == "abs_dev":
            _h = (1.0 - (pl.col("_b") - pl.col("_mu")).abs()).clip(0.0, 1.0)
        else:
            raise ValueError(f"H_MODE must be level|signed_dev|abs_dev|legacy_ema, got {H_MODE!r}")
        df = df.with_columns(pl.when(_has_past).then(_h).otherwise(pl.lit(0.5)).alias("H"))
        _hc = float(df.select(pl.corr("H", "_b")).item())
        print(f"[H] mode={H_MODE}  mean={float(df['H'].mean()):.4f}  std={float(df['H'].std()):.4f}  "
              f"corr(H, (B+C+D)/3)={_hc:+.4f}")

    spd = {s: df.filter(pl.col("rsplit") == s) for s in ("train", "val", "test")}
    # external IDS R
    def Xl3(d): return np.nan_to_num(d.select(L3).to_numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    # R is an EXTERNAL signal: in deployment the IDS is trained once, frozen and served.
    # Two legitimate modes -- retrain answers "does this rebuild from raw?", --ids-model
    # answers "does this reproduce the locked artifact exactly?". Retraining is NOT
    # bit-portable across CPUs (subsample=0.8), and on this leg the weight argmax is a
    # ~4e-6 near-tie, so a ~1e-4 shift in R flips W* and moves every downstream threshold.
    _RPARAMS = dict(objective="binary:logistic", eval_metric="auc", max_depth=7, eta=0.1,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=50, reg_lambda=10.0,
                    tree_method="hist", seed=SEED, nthread=14)
    OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
    if IDS_MODEL:
        rbst = xgb.Booster(); rbst.load_model(str(Path(IDS_MODEL).resolve()))
        print(f"[R] LOADED frozen external-IDS model from {IDS_MODEL} (not retrained)")
        # Provenance, ALWAYS written (previously only written on a real train) so this
        # out/ dir is self-describing regardless of where the frozen model came from --
        # without this, a later audit has no way to tell "R frozen here" from "R never
        # even ran here" short of directory-mtime archaeology.
        (OUT / "model_external_ids_meta.json").write_text(json.dumps(
            {"mode": "frozen", "loaded_from": str(Path(IDS_MODEL).resolve()),
             "note": "R was NOT retrained for this run; a frozen booster was loaded "
                     "from loaded_from. Restore with --ids-model pointing at loaded_from."},
            indent=2), encoding="utf-8")
    else:
        rbst = xgb.train(_RPARAMS,
                         xgb.DMatrix(Xl3(spd["train"]), label=spd["train"]["y"].to_numpy(),
                                     feature_names=L3),
                         num_boost_round=300)
        # Persist it. Closes audit 17.8 on this leg: without this the exact R exists only
        # inside a run's memory, so no later run can be gated against it bit-for-bit.
        rbst.save_model(str(OUT / "model_external_ids.json"))
        (OUT / "model_external_ids_meta.json").write_text(json.dumps(
            {"mode": "trained", "features": L3, "params": _RPARAMS, "num_boost_round": 300,
             "note": "external IDS R. Load with --ids-model for a bit-identical rerun."},
            indent=2), encoding="utf-8")
        print(f"[R] trained + saved -> out/model_external_ids.json")
    for s in ("train", "val", "test"):
        spd[s] = spd[s].with_columns(pl.Series("R", rbst.predict(xgb.DMatrix(Xl3(spd[s]), feature_names=L3))))
    tr, va, te = spd["train"], spd["val"], spd["test"]
    print(f"[R] external-IDS test ROC={gov.roc(te['R'].to_numpy(), te['y'].to_numpy()):.4f}")
    print("[factor AUC_trust test]:", {c: round(float(gov.roc(-te[c].to_numpy(), te['y'].to_numpy())), 3) for c in gov.FACT})
    # Per-SUB-factor discrimination. A factor is a mean, so a dead or backwards member can be
    # masked by its siblings. Since 2026-08-13 this is computed once in the admission gate
    # (Layer 1b) and reused here, so the printed AUCs, admission.json and the metrics JSON
    # can never disagree -- the doc-drift failure mode that bit this leg on 2026-08-11.
    _sfauc = {c: round(v["test_auc_trust_raw"], 3) for c, v in ADMISSION.items()}
    print("[subfactor AUC_trust test]:", _sfauc)

    # CONFERENCE (2026-08-27): calibrateN reads len(gov.FACT) at call time, so --drop-identity
    # lowers the simplex arity 6 -> 5 and raises w_R's ceiling 0.50 -> 0.60. With I present
    # and W_MIN=0.10 this is candidate-for-candidate identical to calibrate6 (selftest asserts it).
    auc, W, wR = gov.calibrateN(va, W_MIN)
    Wd = {k: round(float(w), 2) for k, w in zip(gov.FACT, W)}
    print(f"[calib] W*={Wd} wR*={round(float(wR),2)} val-additive-ROC={auc:.4f}")
    _tk = gov.LAST_CALIB_TOPK
    if len(_tk) > 1:
        _gap = _tk[1]["gap_to_best"]
        print(f"[calib] runner-up gap = {_gap:.3e}  "
              f"(W={_tk[1]['W']} wR={_tk[1]['wR']})")
        if _gap < 1e-4:
            print(f"[calib] *** WARNING: the weight argmax is a NEAR-TIE (gap {_gap:.3e}). "
                  f"A tiny change in R can flip W*, which moves every downstream threshold. "
                  f"Freeze the IDS model (--ids-model) for a reproducible run. ***")
    Tadd_tr = gov.t_add(tr.select(gov.FACT).to_numpy(), tr["R"].to_numpy(), W, wR, tr["decay"].to_numpy())
    TGT = gov.make_target("linear", Tadd_tr.min(), Tadd_tr.max())
    tlo, thi = float(TGT(Tadd_tr).min()), float(TGT(Tadd_tr).max())
    def Xof(d): return np.nan_to_num(d.select(gov.FEATS).to_numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    def tgt(d): return TGT(gov.t_add(d.select(gov.FACT).to_numpy(), d["R"].to_numpy(), W, wR, d["decay"].to_numpy())).astype(np.float32)

    # --- FULL-DAY SCORE DUMP (2026-08-27) ------------------------------------------------
    # POPULATION-MATCHING ONLY. E3b scored the base paper's trust score on ALL rows of the day;
    # our own numbers are TEST-split. To put the two ROCs in one table they must be measured on
    # the same population, so this dumps T_add for EVERY row (train+val+test) with its split tag.
    #
    # READ THE CAVEAT: our W*/w_R* are FITTED ON VAL, so a full-day ROC includes rows that
    # informed the weights. It is a like-for-like comparison against THEIR formula, NOT a
    # generalization number -- the test-split ROC remains the honest one. Always report both,
    # each labelled with its population.
    if DUMP_FULLDAY:
        # NB: `R` is attached to the SPLIT frames (spd[s]), never to `df` -- so the full-day
        # frame must be rebuilt by concatenating tr/va/te, not taken from `df`.
        _cols = list(gov.FACT) + ["R", "decay", "event_id", "rsplit", "y", "label"]
        _all = pl.concat([tr.select(_cols), va.select(_cols), te.select(_cols)],
                         how="vertical_relaxed")
        _Tall = gov.t_add(_all.select(gov.FACT).to_numpy(), _all["R"].to_numpy(), W, wR,
                          _all["decay"].to_numpy())
        _Sall = np.clip(TGT(_Tall), 0.0, 1.0)
        _yall = _all["y"].to_numpy()
        _dest = OUT / f"fullday_scores{SUFFIX}.parquet"
        _all.select(["event_id", "rsplit", "y", "label"]).with_columns(
            pl.Series("T_add", _Tall), pl.Series("score", _Sall)).write_parquet(_dest)
        _roc_all = float(gov.roc(-_Sall, _yall))
        _m = _all["rsplit"].to_numpy()
        _per = {sp: round(float(gov.roc(-_Sall[_m == sp], _yall[_m == sp])), 4)
                for sp in ("train", "val", "test")}
        print(f"[fullday] n={len(_yall):,}  attacks={int(_yall.sum()):,} "
              f"({_yall.mean():.2%})  trust-ROC(full day)={_roc_all:.4f}")
        print(f"[fullday] per-split trust-ROC: {_per}   -> {_dest}")
        FULLDAY = dict(n=int(len(_yall)), n_attack=int(_yall.sum()),
                       prevalence=round(float(_yall.mean()), 6),
                       roc_full_day=round(_roc_all, 4), roc_by_split=_per,
                       caveat=("W*/w_R* are fitted on val, so the full-day ROC includes "
                               "weight-calibration rows. Population-matched comparison vs the "
                               "base paper only; the test-split ROC is the generalization number."))

    if POLICY_SCORE == "formula":
        # *** ABLATION 2026-08-21: NO f_theta at all. `tgt()` IS the closed-form additive
        # formula (T_add, min-max scaled) -- the exact same target f_theta is trained to
        # approximate, just used directly as the decision score instead of distilled first.
        # `R` (the external IDS XGBoost classifier) is untouched -- it already fed into
        # t_add() above via `d["R"]`; this branch only removes the SECOND, distillation
        # model. Answers "what does the proposed algorithm achieve with zero ML on the
        # policy layer itself?" -- see H_SPECIFICATION.md / memory for the UNSW precedent
        # of this same ablation (ROC byte-identical there; f_theta bought +.0051 F1 via
        # threshold interaction only, not better discrimination).
        print("[policy-score] formula -- f_theta NOT trained, scoring directly off T_add")
        Sv = np.clip(tgt(va), tlo, thi)
        St = np.clip(tgt(te), tlo, thi)
    elif POLICY_SCORE == "f_theta":
        bst = xgb.train(dict(objective="reg:squarederror", eval_metric="rmse", max_depth=8, eta=0.10,
                             subsample=FTHETA_SUBSAMPLE, colsample_bytree=FTHETA_COLSAMPLE_BYTREE,
                             min_child_weight=FTHETA_MIN_CHILD_WEIGHT, reg_lambda=FTHETA_REG_LAMBDA,
                             tree_method="hist", monotone_constraints=gov.MONO, seed=SEED, nthread=14),
                        xgb.DMatrix(Xof(tr), label=tgt(tr), feature_names=gov.FEATS), num_boost_round=600,
                        evals=[(xgb.DMatrix(Xof(va), label=tgt(va), feature_names=gov.FEATS), "val")],
                        early_stopping_rounds=40, verbose_eval=False)
        rng = (0, bst.best_iteration + 1)
        # Persist f_theta with everything needed to score a single new request end-to-end:
        # feature order, monotone constraints, the target scaling and clip, and the weights
        # its teacher was built from. Second half of audit 17.8 on this leg.
        bst.save_model(str(OUT / f"model_f_theta{SUFFIX}.json"))
        (OUT / f"model_f_theta{SUFFIX}_meta.json").write_text(json.dumps(
            {"features": gov.FEATS, "monotone_constraints": gov.MONO,
             "best_iteration": int(bst.best_iteration), "iteration_range": list(rng),
             "target_transform": "linear", "target_clip": [tlo, thi],
             "teacher_W": Wd, "teacher_wR": round(float(wR), 4), "lambda_per_hour": LAMBDA},
            indent=2), encoding="utf-8")
        Sv = np.clip(bst.predict(xgb.DMatrix(Xof(va), feature_names=gov.FEATS), iteration_range=rng), tlo, thi)
        St = np.clip(bst.predict(xgb.DMatrix(Xof(te), feature_names=gov.FEATS), iteration_range=rng), tlo, thi)
    else:
        raise ValueError(f"--policy-score must be f_theta|formula, got {POLICY_SCORE!r}")
    yv, tv = va["y"].to_numpy(), va["tier"].to_numpy()
    yt, tt = te["y"].to_numpy(), te["tier"].to_numpy()
    # Threshold-free rank-quality of the POLICY SCORE (Sv) on validation -- unlike realized
    # val FP/UAR, which are pinned close to aar_max/bdr_max by the quantile-based calibration
    # itself regardless of model quality, ROC-AUC is not subject to that circularity, so it is
    # the model-selection signal ftheta_regularization_sweep_cicids.py reads (never test).
    POLICY_VAL_ROC = round(float(gov.roc(-Sv, yv)), 4)

    tiers = sorted(set(int(x) for x in tt) | set(int(x) for x in tv))

    results = {}
    for mode in ("global", "monotonic", "independent"):
        aar = {t: v for t, v in zip(sorted(tiers), (list(AAR_MAX) + [AAR_MAX[-1]] * 4)[:len(tiers)])}
        bdr = {t: v for t, v in zip(sorted(tiers), (list(BDR_MAX) + [BDR_MAX[-1]] * 4)[:len(tiers)])}
        if mode == "global":
            allow, deny, allow_raw, deny_raw = pcal.calibrate_global(Sv, yv, tiers, AAR_GLOBAL, BDR_GLOBAL)
        else:
            am, dm = pcal.MODE_MONO[mode]
            allow, deny, allow_raw, deny_raw = pcal.calibrate_tiers_constrained(
                Sv, yv, tv, tiers, aar, bdr,
                allow_monotone=am, deny_monotone=dm, band_floor=BAND_FLOOR,
                band_floor_max_frac=BAND_FLOOR_MAX_FRAC)
        d_te = pcal.decide(St, tt, allow, deny)   # decisions on the TEST split (thresholds from val)
        if mode == "monotonic":
            # Per-row test decision dump (2026-08-21, report-only, changes no computed value) --
            # keys + label + y + tier + decision (0=Allow 1=Restrict 2=Deny) + the raw score and
            # its tier's thresholds, so any attack-TYPE subgroup (e.g. "did the PDP restrict/
            # deny the labels never seen in train?") OR score-level comparison between
            # --policy-score arms can be done after the fact without rerunning the pipeline.
            # Featured/monotonic mode only.
            _allow_arr = np.array([allow[int(x)] for x in tt])
            _deny_arr = np.array([deny[int(x)] for x in tt])
            te.select(["source_ip", "label", "y", "tier"]).with_columns(
                pl.Series("decision", d_te),
                pl.Series("score", St),
                pl.Series("allow_thr", _allow_arr),
                pl.Series("deny_thr", _deny_arr),
            ).write_parquet(OUT / f"decisions_test_monotonic{SUFFIX}.parquet")
        ben_m, atk_m = yt == 0, yt == 1
        nben, natk = int(ben_m.sum()), int(atk_m.sum())
        # detection confusion matrix: "flagged" = not-Allow (Restrict OR Deny)
        na = d_te != 0
        tp = int((atk_m & na).sum()); fp = int((ben_m & na).sum())
        fn = int((atk_m & ~na).sum()); tn = int((ben_m & ~na).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(natk, 1)
        # pooled Allow/Restrict/Deny distribution per population (graduated-response / universality evidence)
        def dist(mask):
            dd = d_te[mask]; n = int(mask.sum())
            return dict(n=n, allow=int((dd == 0).sum()), restrict=int((dd == 1).sum()), deny=int((dd == 2).sum()),
                        allow_rate=round(float((dd == 0).mean()), 4) if n else None,
                        restrict_rate=round(float((dd == 1).mean()), 4) if n else None,
                        deny_rate=round(float((dd == 2).mean()), 4) if n else None)
        results[mode] = dict(
            # --- headline TEST metrics ---
            ADR=round(rec, 4), UAR=round(tn / max(nben, 1), 4),
            precision=round(prec, 4), recall=round(rec, 4),
            F1=round(2 * prec * rec / max(prec + rec, 1e-12), 4),
            accuracy=round((tp + tn) / max(len(yt), 1), 4),
            roc=round(float(gov.roc(-St, yt)), 4),
            # benign-friction split: Restrict (step-up) vs hard-Deny
            BAR_restrict=round(int((ben_m & (d_te == 1)).sum()) / max(nben, 1), 4),
            BAR_deny=round(int((ben_m & (d_te == 2)).sum()) / max(nben, 1), 4),
            attack_deny_rate=round(int((atk_m & (d_te == 2)).sum()) / max(natk, 1), 4),
            # confusion matrix (flagged = not-Allow)
            confusion=dict(TP=tp, FP=fp, FN=fn, TN=tn, n=int(len(yt)), n_attack=natk, n_benign=nben),
            FN=fn, n_attack=natk,
            # graduated decision distribution (Allow/Restrict/Deny) — the universality scope
            decision_distribution=dict(benign=dist(ben_m), attack=dist(atk_m),
                                       attack_lowconf=dist(atk_m & (te["R"].to_numpy() < 0.5)),
                                       attack_highconf=dist(atk_m & (te["R"].to_numpy() >= 0.5))),
            ladder={str(t): [round(allow[t], 3), round(deny[t], 3), round(allow[t] - deny[t], 3)] for t in tiers},
            # raw (pre-isotonic) thresholds: shows WHERE the projection actually bound
            ladder_raw={str(t): [round(float(allow_raw[t]), 4), round(float(deny_raw[t]), 4)] for t in tiers},
            tier_breakdown={str(t): pcal.tier_action_rates(yt, d_te, tt, tiers)[t] for t in tiers})
        r = results[mode]; L = r['ladder']
        print(f"[{mode:11s}] ADR={r['ADR']} UAR={r['UAR']} P={r['precision']} R={r['recall']} "
              f"F1={r['F1']} Acc={r['accuracy']} ROC={r['roc']}  TP/FP/FN/TN={tp}/{fp}/{fn}/{tn}  "
              f"MFA-bands={ {t: L[str(t)][2] for t in tiers} }")
        print(f"             benign A/R/D={r['decision_distribution']['benign']['allow_rate']}/"
              f"{r['decision_distribution']['benign']['restrict_rate']}/{r['decision_distribution']['benign']['deny_rate']}  "
              f"attack A/R/D={r['decision_distribution']['attack']['allow_rate']}/"
              f"{r['decision_distribution']['attack']['restrict_rate']}/{r['decision_distribution']['attack']['deny_rate']}")
        # per-tier pooled Allow/Restrict/Deny (benign vs attack) — the resource-aware view
        for t in tiers:
            tb = r['tier_breakdown'][str(t)]
            print(f"               tier {t}: benign A/R/D={tb['benign']['allow']}/{tb['benign']['restrict']}/{tb['benign']['deny']}"
                  f"  attack A/R/D={tb['attack']['allow']}/{tb['attack']['restrict']}/{tb['attack']['deny']}  (n={tb['n']:,})")

    cfg = dict(files=FILES, split="per-file row-order 60/20/20", W=Wd, wR=round(float(wR), 4),
               # ICECE 2026 conference arm provenance (2026-08-27)
               drop_identity=DROP_IDENTITY, factors_used=list(gov.FACT),
               simplex_arity=len(gov.FACT) + 1, w_min=W_MIN,
               aar_global=AAR_GLOBAL, bdr_global=BDR_GLOBAL, fullday=FULLDAY,
               policy_score=POLICY_SCORE, policy_val_roc=POLICY_VAL_ROC,
               ftheta_hparams=(dict(subsample=FTHETA_SUBSAMPLE, colsample_bytree=FTHETA_COLSAMPLE_BYTREE,
                                    min_child_weight=FTHETA_MIN_CHILD_WEIGHT, reg_lambda=FTHETA_REG_LAMBDA)
                               if POLICY_SCORE == "f_theta" else None),
               aar_max=list(AAR_MAX), bdr_max=list(BDR_MAX), band_floor=BAND_FLOOR, band_floor_max_frac=BAND_FLOOR_MAX_FRAC,
               fix_12h_clock=FIX_12H_CLOCK,
               val_additive_roc=round(auc, 4), R_test_roc=round(float(gov.roc(te['R'].to_numpy(), yt)), 4),
               # `subfactor_map` is the DECLARED decomposition; `subfactor_map_effective` is
               # what the run ACTUALLY averaged. They differ under --reorient/--gate-mode
               # enforce. Added 2026-08-13: the three ablation arms run that day recorded only
               # the declared map, so their JSONs claim B<-2 D<-3 C<-2 while the logs show they
               # used B<-1 D<-2 C<-1. Read those three from the logs, not the JSON.
               subfactor_map=SUBFACTORS, subfactor_map_effective=_USE_EFFECTIVE,
               subfactor_auc=_sfauc, h_mode=H_MODE,
               gate_mode=GATE_MODE, iqr_floor=IQR_FLOOR, stability_max=STABILITY_MAX,
               subfactor_admission={c: {k: v[k] for k in
                                        ("iqr", "stability_delta", "gate_verdict", "admitted")}
                                    for c, v in ADMISSION.items()},
               calib_topk=gov.LAST_CALIB_TOPK,
               calib_runner_up_gap=(gov.LAST_CALIB_TOPK[1]["gap_to_best"]
                                    if len(gov.LAST_CALIB_TOPK) > 1 else None),
               factor_auc={c: round(float(gov.roc(-te[c].to_numpy(), yt)), 4) for c in gov.FACT},
               n_test=int(len(yt)), n_test_attack=int((yt == 1).sum()), n_test_benign=int((yt == 0).sum()))
    out = dict(config=cfg, results=results)
    (HERE / f"metrics_multiday{SUFFIX}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_governance_md(cfg, results, tiers)
    print(f"  -> {HERE}/metrics_multiday{SUFFIX}.json  +  GOVERNANCE_MULTIDAY{SUFFIX}.md")


TIER_NAME = {1: "Public", 2: "Internal", 3: "Sensitive", 4: "Critical"}


def write_governance_md(cfg, results, tiers):
    """Emit GOVERNANCE_MULTIDAY.md — all TEST tables (metrics, confusion, Allow/Restrict/Deny
    distribution pooled + per-tier) so the doc always stays in sync with metrics_multiday.json."""
    MODES = [("global", "B1 global (single gate)"), ("monotonic", "B2 monotonic (resource-aware ladder)"),
             ("independent", "B3 independent (no monotonicity)")]
    n, na, nb = cfg["n_test"], cfg["n_test_attack"], cfg["n_test_benign"]
    L = []
    L.append("# CICIDS2017 (3-day) — Governance / TEST evaluation\n")
    L.append("Auto-generated by `run_multiday_experiment.py`. **All numbers are TEST** "
             f"({n:,} flows: {na:,} attack / {nb:,} benign). Thresholds are calibrated on **val**, "
             "decisions/metrics computed on the held-out **test** split.\n")
    L.append(f"**Files:** {', '.join(cfg['files'])}  ·  **Split:** {cfg['split']}\n")
    L.append(f"**Fusion weights** `W*={cfg['W']}`, `wR*={cfg['wR']}` (factors {round((1-cfg['wR'])*100)}% / R "
             f"{round(cfg['wR']*100)}%) — grid-searched on val (max additive-ROC {cfg['val_additive_roc']}).\n")
    L.append(f"**Per-factor trust-AUC (test):** {cfg['factor_auc']}  ·  **External IDS R ROC (test):** {cfg['R_test_roc']}\n")
    L.append("> Decision rule: `T ≥ aₜ → Allow` · `dₜ ≤ T < aₜ → Restrict (step-up MFA)` · `T < dₜ → Deny`, "
             "with per-tier thresholds `(aₜ, dₜ)`.\n")

    L.append("## 1. Headline test metrics by policy mode\n")
    L.append("| metric | " + " | ".join(d for _, d in MODES) + " |")
    L.append("|---|" + "|".join("---" for _ in MODES) + "|")
    rows = [("ADR (attack not-Allow)", "ADR"), ("UAR (benign Allow)", "UAR"), ("Precision", "precision"),
            ("Recall (=ADR)", "recall"), ("F1", "F1"), ("Accuracy", "accuracy"),
            ("ROC-AUC (threshold-free)", "roc"), ("BAR-restrict (benign step-up)", "BAR_restrict"),
            ("BAR-deny (benign hard-block)", "BAR_deny"), ("attack hard-Deny rate", "attack_deny_rate")]
    for label, key in rows:
        L.append(f"| {label} | " + " | ".join(str(results[m][key]) for m, _ in MODES) + " |")

    L.append("\n## 2. Confusion matrix (flagged = not-Allow = Restrict ∪ Deny)\n")
    L.append("| cell | " + " | ".join(d for _, d in MODES) + " |")
    L.append("|---|" + "|".join("---" for _ in MODES) + "|")
    for cell in ("TP", "FP", "FN", "TN"):
        L.append(f"| {cell} | " + " | ".join(f"{results[m]['confusion'][cell]:,}" for m, _ in MODES) + " |")
    L.append("\n*TP = attack flagged · FP = benign flagged · FN = attack Allowed (missed) · TN = benign Allowed.*\n")

    L.append("## 3. Graduated response — pooled Allow / Restrict / Deny distribution\n")
    L.append("Evidence the PDP gives a **3-way graduated** decision, not a binary verdict "
             "(the universality scope). Rows are % of each population.\n")
    for m, desc in MODES:
        dd = results[m]["decision_distribution"]
        L.append(f"\n**{desc}**\n")
        L.append("| population | n | Allow | Restrict (step-up MFA) | Deny |")
        L.append("|---|---|---|---|---|")
        for pop, key in [("benign", "benign"), ("attack (all)", "attack"),
                         ("attack R<0.5 (low-conf)", "attack_lowconf"), ("attack R≥0.5 (high-conf)", "attack_highconf")]:
            p = dd[key]
            ar, rr, dr = p["allow_rate"], p["restrict_rate"], p["deny_rate"]
            fmt = lambda x: f"{x:.2%}" if x is not None else "—"
            L.append(f"| {pop} | {p['n']:,} | {fmt(ar)} | {fmt(rr)} | {fmt(dr)} |")
    L.append("\n*B1 global has **zero Restrict** (single gate → binary). Only the resource-aware ladder "
             "(B2/B3) opens the step-up/MFA band — the governance contribution.*\n")

    L.append("## 4. Per-tier threshold ladder (calibrated on val)\n")
    for m, desc in MODES:
        L.append(f"\n**{desc}**\n")
        L.append("| tier | resource | allow ≥ | deny < | Restrict band |")
        L.append("|---|---|---|---|---|")
        for t in tiers:
            a, d_, band = results[m]["ladder"][str(t)]
            L.append(f"| {t} | {TIER_NAME.get(t,'?')} | {a:.3f} | {d_:.3f} | {band:.3f} |")

    L.append("\n## 5. Per-tier decision distribution (test counts, benign vs attack)\n")
    for m, desc in MODES:
        L.append(f"\n**{desc}**\n")
        L.append("| tier | resource | n | benign Allow/Restrict/Deny | attack Allow/Restrict/Deny |")
        L.append("|---|---|---|---|---|")
        for t in tiers:
            tb = results[m]["tier_breakdown"][str(t)]
            b, a = tb["benign"], tb["attack"]

            # 2026-08-27: a tier with ZERO rows of one class gets None rates, not floats.
            # The locked 3-day set has attacks in every tier so this never fired there; a
            # SINGLE-DAY run does hit it -- Wednesday has no Tier-3/Tier-4 attacks at all.
            # Formatting crashed AFTER the metrics JSON was already written, so it only ever
            # cost the .md. Render the empty cell as "--" instead of dying.
            def _ard(d):
                if d is None or any(d.get(k) is None for k in ("allow", "restrict", "deny")):
                    return "--"
                return f"{d['allow']:.3f}/{d['restrict']:.3f}/{d['deny']:.3f}"

            L.append(f"| {t} | {TIER_NAME.get(t,'?')} | {tb['n']:,} | {_ard(b)} | {_ard(a)} |")
    L.append("\n*Rates are within-population (benign rows sum to 1 across A/R/D; attack rows likewise).*\n")
    (HERE / f"GOVERNANCE_MULTIDAY{SUFFIX}.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description="CICIDS2017 FSL-PDP experiment")
    _ap.add_argument("--aar", help="comma-separated per-tier attack-Allow budget, e.g. 0.01,0.007,0.005,0.003")
    _ap.add_argument("--bdr", help="comma-separated per-tier benign-Deny budget")
    _ap.add_argument("--band-floor", type=float, help="minimum step-up band width")
    _ap.add_argument("--band-floor-frac", type=float, help="cap the band floor at this fraction of allow[t]")
    _ap.add_argument("--suffix", default="", help="tag the output files")
    _ap.add_argument("--ids-model", dest="ids_model", default=None,
                     help="path to a frozen out/model_external_ids.json. The external IDS is "
                          "LOADED instead of retrained, which makes R bit-identical on any "
                          "machine. Without it R is retrained (subsample=0.8) and can differ "
                          "slightly across CPUs -- and because the weight argmax on this leg is "
                          "a ~4e-6 near-tie, that is enough to flip W* and move every threshold.")
    _ap.add_argument("--h-mode", choices=["level", "signed_dev", "abs_dev", "legacy_ema"], default="legacy_ema",
                     help="how the History factor is expressed; see H_MODE. `legacy_ema` "
                          "(promoted to default 2026-08-21) ports LANL's self-referential EMA "
                          "H system, incl. a new CICIDS-native R_init risk channel -- see "
                          "trajectory_kernel.py. `level` was the pre-2026-08-21 lock.")
    _ap.add_argument("--gate-mode", choices=["report", "enforce"], default="report",
                     help="admission gate (IQR floor + train/val stability). `report` "
                          "(DEFAULT) computes the verdict and writes admission.json but "
                          "changes NOTHING = the lock. `enforce` excludes rejected "
                          "sub-factors from their factor's mean.")
    _ap.add_argument("--reorient", choices=["flip", "drop"], default=None,
                     help="ABLATION: 'flip' replaces the backwards sub-factors with 1-x; "
                          "'drop' excludes them from their factor's mean (what UNSW's "
                          "admission gate does). Default None = the locked path.")
    _ap.add_argument("--policy-score", choices=["f_theta", "formula"], default="f_theta",
                     help="ABLATION (2026-08-21): 'formula' bypasses the f_theta XGBoost "
                          "regressor entirely, scoring/deciding directly off the closed-form "
                          "additive T_add formula. 'f_theta' (default) is the lock. `R` (the "
                          "external IDS) is unaffected either way.")
    _ap.add_argument("--ftheta-subsample", type=float, default=None,
                     help="ABLATION (2026-08-24): f_theta's own row subsample fraction. "
                          "Default None reproduces the locked value (1.0).")
    _ap.add_argument("--ftheta-colsample-bytree", type=float, default=None,
                     help="ABLATION (2026-08-24): f_theta's own column subsample fraction. "
                          "Default None reproduces the locked value (1.0).")
    _ap.add_argument("--ftheta-min-child-weight", type=float, default=None,
                     help="ABLATION (2026-08-24): f_theta's min_child_weight. "
                          "Default None reproduces the locked value (10).")
    _ap.add_argument("--ftheta-reg-lambda", type=float, default=None,
                     help="ABLATION (2026-08-24): f_theta's L2 leaf-weight regularization. "
                          "Default None reproduces the locked value (1.0).")

    # --- ICECE 2026 CONFERENCE ARM (2026-08-27) ------------------------------------
    _ap.add_argument("--day", default=None,
                     help="CONFERENCE: restrict the day set. Comma-separated substrings matched "
                          "against FILES, e.g. 'Wednesday'. Default None = the locked 3-day set "
                          "(Tuesday + Wednesday + Friday-PortScan).")
    _ap.add_argument("--drop-identity", dest="drop_identity", action="store_true",
                     help="CONFERENCE: drop the Identity factor I entirely. On this leg I is a "
                          "HARDCODED CONSTANT 0.5 (SUBFACTORS['I'] == [] -- CICFlowMeter exposes "
                          "no identity telemetry), so W_I*0.5*decay is a constant offset the "
                          "threshold calibration absorbs. Dropping it lowers the simplex arity "
                          "6 -> 5 and frees w_R's ceiling 0.50 -> 0.60. The I column is still "
                          "BUILT; it simply never enters FACT/FEATS/MONO.")
    _ap.add_argument("--aar-global", dest="aar_global", type=float, default=None,
                     help="CONFERENCE: B1 global-gate attack-Allow budget (default 0.02).")
    _ap.add_argument("--bdr-global", dest="bdr_global", type=float, default=None,
                     help="CONFERENCE: B1 global-gate benign-Deny budget (default 0.08).")
    _ap.add_argument("--dump-fullday", dest="dump_fullday", action="store_true",
                     help="CONFERENCE: also score EVERY row of the day (train+val+test) with the "
                          "calibrated W*/w_R* and dump out/fullday_scores<suffix>.parquet, for a "
                          "population-matched ROC comparison against the base paper's whole-day "
                          "trust score (E3b). NOT a generalization number -- see the code comment.")
    _ap.add_argument("--w-min", dest="w_min", type=float, default=None,
                     help="CONFERENCE: weight-simplex floor (default 0.10).")
    _a = _ap.parse_args()
    REORIENT = _a.reorient
    H_MODE = _a.h_mode
    GATE_MODE = _a.gate_mode
    POLICY_SCORE = _a.policy_score
    if _a.aar: AAR_MAX = [float(x) for x in _a.aar.split(",")]
    if _a.bdr: BDR_MAX = [float(x) for x in _a.bdr.split(",")]
    if _a.band_floor is not None: BAND_FLOOR = _a.band_floor
    if _a.band_floor_frac is not None: BAND_FLOOR_MAX_FRAC = _a.band_floor_frac
    if _a.ftheta_subsample is not None: FTHETA_SUBSAMPLE = _a.ftheta_subsample
    if _a.ftheta_colsample_bytree is not None: FTHETA_COLSAMPLE_BYTREE = _a.ftheta_colsample_bytree
    if _a.ftheta_min_child_weight is not None: FTHETA_MIN_CHILD_WEIGHT = _a.ftheta_min_child_weight
    if _a.ftheta_reg_lambda is not None: FTHETA_REG_LAMBDA = _a.ftheta_reg_lambda
    SUFFIX = _a.suffix
    IDS_MODEL = _a.ids_model

    # --- ICECE 2026 CONFERENCE ARM wiring (2026-08-27) ---------------------------------
    if _a.aar_global is not None: AAR_GLOBAL = _a.aar_global
    if _a.bdr_global is not None: BDR_GLOBAL = _a.bdr_global
    if _a.w_min is not None: W_MIN = _a.w_min
    DUMP_FULLDAY = _a.dump_fullday
    if _a.day:
        _want = [d.strip().lower() for d in _a.day.split(",") if d.strip()]
        _sel = [f for f in FILES if any(w in f.lower() for w in _want)]
        if not _sel:
            raise SystemExit(f"--day {_a.day!r} matched none of {FILES}")
        FILES = _sel
        print(f"[day] restricted to {len(FILES)} file(s): {FILES}")
    if _a.drop_identity:
        # Must happen BEFORE main(): every consumer reads gov.FACT/FEATS/MONO at call time.
        DROP_IDENTITY = True
        gov.FACT = [f for f in gov.FACT if f != "I"]
        gov.FEATS = gov.FACT + ["R", "decay"]
        gov.MONO = "(" + ",".join(["1"] * len(gov.FACT) + ["-1", "1"]) + ")"
        print(f"[drop-identity] FACT={gov.FACT}  (simplex arity {len(gov.FACT)+1}, "
              f"wR ceiling {1 - len(gov.FACT)*W_MIN:.2f})")
    main()
