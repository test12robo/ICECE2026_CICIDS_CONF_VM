# ICECE 2026 -- CICIDS2017 conference leg, VM runbook

Built **2026-08-27** from `ICECE2026_CONFERENCE_PAPER/pipeline_cicids/`.
Target: a clean environment to run (and re-run/extend) the CICIDS2017 Wednesday
experiments the conference paper's second results table depends on (E9).

---

## What this bundle is, and is NOT

CICIDS is **not yet a single locked operating point** the way the UNSW bundle is.
Several arms have been run and are all documented below; the one currently
recommended for the paper is **`--gate-mode enforce`** (see step 3), but nothing
here forces that as a hidden default -- every command is explicit, on purpose,
because this leg is still an open sweep. `run_pipeline.py` ships byte-identical
to the working copy; **no argparse defaults were changed.**

    T_add = ( W_B*B + W_D*D + W_C*C + W_H*H ) * e^(-lambda*dt)  -  w_R * R

---

## 0. Prerequisites

    python -m pip install -r requirements.txt

Pinned: `polars==1.40.1`, `numpy==2.4.3`, `xgboost==3.2.0`, `scikit-learn==1.8.0`,
`pandas==3.0.1`, `pyarrow==24.0.0`. See the file for the full rationale.

Raw data goes in `dataset/` (see `dataset/PUT_RAW_CSVS_HERE.txt`). On the VM these
already exist under the locked leg -- junction instead of copying:

    mklink /J dataset C:\Rafiea\CICIDS_PDP_CLEAN_VM\dataset

**Run every command from the bundle root.**

---

## 1. Pin the thread counts BEFORE anything else

`R` (the external IDS) is XGBoost trained with `subsample=0.8`/`colsample_bytree=0.8`.
A fixed seed alone does not make that bit-portable across machines with different
core counts -- and unlike the UNSW bundle, a Wednesday-only run RETRAINS `R` (the
frozen 3-day model would leak Tuesday/Friday into a Wednesday-only split), so this
matters here even for the "no flags" runs.

    set OMP_NUM_THREADS=14
    set MKL_NUM_THREADS=14
    set OPENBLAS_NUM_THREADS=14
    set POLARS_MAX_THREADS=14
    set NUMBA_NUM_THREADS=14

---

## 2. Optional self-test (seconds, synthetic data)

    python selftest_conference_cicids.py

Checks `simplexN`/`calibrateN` against `simplex6`/`calibrate6`, the `--drop-identity`
rebinding, and the `--day` filter -- before spending real time on ~700k real rows.

---

## 3. THE RECOMMENDED ARM -- run this first

    python run_pipeline.py --day Wednesday --drop-identity --policy-score formula ^
           --aar-global 0.01 --bdr-global 0.005 --gate-mode enforce --suffix _wed_gateenf

    python verify_conference.py

Expected (2026-08-27 run, `_REFERENCE/metrics_wed_gateenf_CONFERENCE_ANCHOR.json`):

    W* = {B .20, D .10, C .10, H .25}   wR* = 0.35   (runner-up gap 2.386e-05 -- a near-tie)
    ADR .9434 | UAR .9828 | precision .7496 | F1 .8354 | ROC .9954
    TP 6,767 | FP 2,261 | FN 406 | TN 129,107
    global gate: allow >= 0.554 | deny < 0.459   (ONE band -- B1 prints one row per tier
    key, it is not four separate thresholds; see E9 write-up section 11)

`--gate-mode enforce` is the recommended arm because it is a clean Pareto win over
the `report`-mode baseline (below) AND repairs the two inverted factors (`D`, `C`)
at the source -- see `experiments/results/E9_cicids_wednesday.md` sections 10a-10f
for the full diagnosis. It is selected on **validation** ROC (.9919 vs .9767), the
gate itself is label-free, and UNSW's pipeline already enforces this same gate --
CICIDS running in `report` mode was a leg inconsistency, not a considered choice.

**Caveat to carry forward**: the weight argmax gap here (2.386e-05) is close enough
that `R`'s retrain noise CAN flip it to the runner-up (`H .20`, `wR .40`) on a
different machine. `verify_conference.py` treats a `W`/`wR` mismatch as a flag to
re-read, not an automatic failure.

---

## 4. Other arms already run locally, reproducible the same way

| arm | command (append to the step-3 base flags, replace --gate-mode) | ADR / UAR / P / F1 | FP / FN | wR* |
|---|---|---|---|---|
| baseline (report mode, weak -- do not report this one) | `--gate-mode report --suffix _wed_dropI` | .8922 / .9632 / .5700 / .6956 | 4,828 / 773 | 0.45 |
| aar .02 (aar_global is inert, FP bit-identical) | `--gate-mode report --aar-global 0.02 --suffix _wed_aar02` | .8844 / .9632 / .5678 / .6916 | 4,828 / 829 | 0.45 |
| aar .05 | `--gate-mode report --aar-global 0.05 --suffix _wed_aar05` | .7638 / .9637 / .5344 / .6289 | 4,773 / 1,694 | 0.45 |
| w-min 0.05 (rejected -- drives wR to 0.75, argues against the thesis) | `--gate-mode report --w-min 0.05 --suffix _wed_wmin05` | .8889 / .9938 / .8874 / .8881 | 809 / 797 | **0.75** |

## 5. Optional -- full 692,703-row Wednesday day (not yet run as of 2026-08-27 14:34)

    python run_pipeline.py --day Wednesday --drop-identity --policy-score formula ^
           --aar-global 0.01 --bdr-global 0.005 --gate-mode enforce --dump-fullday ^
           --suffix _wed_gateenf_fullday

Needed only if the base-paper comparison (E3b, which scores their trust score on
the full day) goes into a shared table with our numbers -- E9's test-split ROC .9898
(baseline) / .9954 (gate-enforce) is NOT measured on the same population as E3b's
trust-ROC range, and that mismatch must not be papered over in a joint table.

## 6. Optional -- 3-day byte-safety regression (needs Tuesday + Friday-PortScan too)

Proves the refactor (`simplexN`, `--day`, `--drop-identity`, `--gate-mode`, `--w-min`)
is inert when none of those flags are passed -- i.e. this copy still reproduces the
LOCKED 3-day leg exactly:

    python run_pipeline.py --ids-model _REFERENCE/model_external_ids.json --suffix _regress3day
    python verify.py metrics_multiday_regress3day.json

`verify.py`'s bundled default reference is `_REFERENCE/metrics_cicids_locked_20260821_legacyema.json`
(ROC .9972 / ADR .9741 / UAR .9978 / F1 .9777 / FN 790 / FP 562, monotonic mode).
Expect **0 differences in `results.*`**; **11 report-only `factor_auc`/`subfactor_auc`
diagnostics WILL differ** -- that lock JSON pre-dates the leg's `roc()` tie-handling
fix (2026-08-22). Never quote `factor_auc`/`subfactor_auc` out of a lock file; quote
them from a fresh run instead. Full explanation: E9 write-up section 8.

---

## 7. What was removed from the working pipeline_cicids/ folder, and why

| removed | reason |
|---|---|
| `__pycache__/`, `out/`, `*.bak_preconf` | caches and backups |
| every already-produced `admission_*.json`/`metrics_multiday_*.json`/`GOVERNANCE_MULTIDAY_*.md`/`*.log` | regenerate on the VM instead of shipping stale local runs |
| `_REFERENCE/metrics_cicids_locked_20260811.json`, `metrics_multiday_20260618.json`, `metrics_PRE_TIERALIGN.json`, `metrics_STEP{0,1,2,3}_*.json`, `_gate_20260813/` | superseded locked-leg history, not on the conference path |
| `summarize_wed_sweep.py`, `sweep_conf_wed.sh` | local scratch drivers for the sweep in step 4, not needed to re-run any single arm |

**Nothing was edited.** Check `MANIFEST.sha256.json` -- every shipped file hashes
identically to `pipeline_cicids/`. `verify_conference.py` and this RUNBOOK are new
files, not modifications.

---

## 8. Scope rules that outlive this bundle

The conference paper must never contain: the name **FSL-PDP** / FCH-PDP /
Formula-Supervised; `f_theta` or distillation; per-tier tables or resource tiers
(this leg's tier ladder is internal diagnostics only -- report a single global
allow/deny pair, see step 3); or the sentence *"identity is not a trust factor."*
Whether CICIDS enters the paper at all, and in which form, was an open user
decision as of 2026-08-27 -- check `PROGRESS.md` before assuming it is settled.
