#!/usr/bin/env python
"""
Compare a fresh CICIDS run against a reference metrics file, field by field.

    python verify.py                          # metrics_multiday.json vs the 2026-06-18 anchor
    python verify.py new.json --ref old.json

Exit code 0 = identical, 1 = differences found.

DEFAULT REFERENCE = `_REFERENCE/metrics_cicids_locked_20260821_legacyema.json` — the CURRENT
lock (monotonic: ROC .9972 / ADR .9741 / UAR .9978 / F1 .9777 / FN 790 / FP 562).
A fresh `python run_pipeline.py` must reproduce it EXACTLY.

  RETARGETED 2026-08-21. `--h-mode` default flipped `level` -> `legacy_ema` (LANL's
  self-referential EMA H system, ported this same day; needed a new CICIDS-native
  R_init risk channel, see RINIT_* in run_pipeline.py / trajectory_kernel.py).
  Measured with frozen R then reconfirmed flagless (fresh R retrain landed on the
  identical W*/wR despite this leg's documented near-tie instability): ADR
  .9521->.9741, F1 .9663->.9777, FN 1460->790, UAR/ROC/FP essentially unchanged.
  CAVEAT (disclosed, not hidden): factor_auc.H is IDENTICAL to the old `level` lock's
  (.969 both) -- CICIDS's attack traffic is 100% from one source_ip, so this is a
  calibration/threshold-interaction gain, not new information from H. Both `level`
  and `legacy_ema` inherit the same leak; user decision was to accept it (inherent
  to this lab-generated dataset, same as UNSW) and use the better-performing formula.

  The previous lock (`level`, 2026-08-11) is preserved at
  `_REFERENCE/metrics_cicids_locked_20260811.json`. Reproduce ITS equivalence check
  (now expected to show the `legacy_ema` vs `level` H-mode difference, not identity)
  with:
      python run_pipeline.py --h-mode level --suffix _levellegacy
      python verify.py metrics_multiday_levellegacy.json --ref _REFERENCE/metrics_cicids_locked_20260811.json

  Older anchor (06-18, pre-refactor) still available the same way:
      python verify.py --ref _REFERENCE/metrics_multiday_20260618.json --expect-diff

NOTE: `--expect-diff` inverts the exit code, for when differences ARE the outcome
you are testing for.
"""
from pathlib import Path
import argparse
import json
import sys

HERE = Path(__file__).resolve().parent
DEFAULT_NEW = HERE / "metrics_multiday.json"
DEFAULT_REF = HERE / "_REFERENCE" / "metrics_cicids_locked_20260821_legacyema.json"

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = YELLOW = RESET = ""

MODES = ("global", "monotonic", "independent")
HEADLINE = ("ADR", "UAR", "precision", "recall", "F1", "accuracy", "roc",
            "BAR_restrict", "BAR_deny", "attack_deny_rate", "FN", "n_attack")


def walk(o, prefix=""):
    """Flatten a nested dict/list into {dotted.path: scalar}."""
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(walk(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out.update(walk(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = o
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="CICIDS metrics equivalence check")
    ap.add_argument("new", nargs="?", default=str(DEFAULT_NEW))
    ap.add_argument("--ref", default=str(DEFAULT_REF))
    ap.add_argument("--expect-diff", action="store_true",
                    help="invert the exit code: differences are the expected outcome")
    args = ap.parse_args()

    pn, pr = Path(args.new), Path(args.ref)
    for p in (pn, pr):
        if not p.is_file():
            print(f"not found: {p}")
            return 2
    new, ref = json.loads(pn.read_text()), json.loads(pr.read_text())
    print(f"new : {pn}")
    print(f"ref : {pr}\n")

    # ---- configuration first: a mismatch here explains every metric below ----
    print("--- configuration ---")
    cn, cr = new.get("config", {}), ref.get("config", {})
    cfg_diff = 0
    for k in sorted(set(cn) | set(cr)):
        a, b = cr.get(k), cn.get(k)
        if a == b:
            print(f"  {GREEN}OK{RESET}      {k}")
        else:
            cfg_diff += 1
            print(f"  {RED}DIFF{RESET}    {k}\n"
                  f"            ref = {a}\n"
                  f"            new = {b}")

    # ---- headline metrics per policy mode ----
    print("\n--- headline metrics by policy mode ---")
    print(f"{'metric':<22}{'mode':<13}{'ref':>12}{'new':>12}{'delta':>12}   verdict")
    print("-" * 86)
    n_diff = 0
    for mode in MODES:
        rm, nm = ref["results"].get(mode, {}), new["results"].get(mode, {})
        for k in HEADLINE:
            a, b = rm.get(k), nm.get(k)
            if a == b:
                continue
            n_diff += 1
            try:
                delta = f"{b - a:+.4f}" if isinstance(a, float) else f"{b - a:+,}"
            except Exception:
                delta = "n/a"
            print(f"{k:<22}{mode:<13}{str(a):>12}{str(b):>12}{delta:>12}   {RED}DIFF{RESET}")
    if n_diff == 0:
        for mode in MODES:
            rm = ref["results"].get(mode, {})
            print(f"{'(all identical)':<22}{mode:<13}"
                  f"{'ADR ' + str(rm.get('ADR')):>12}{'F1 ' + str(rm.get('F1')):>12}"
                  f"{'FN ' + str(rm.get('FN')):>12}   {GREEN}OK{RESET}")

    # ---- exhaustive: every leaf in results ----
    fr, fn_ = walk(ref.get("results", {})), walk(new.get("results", {}))
    keys = sorted(set(fr) | set(fn_))
    deep = [k for k in keys if fr.get(k) != fn_.get(k)]
    print(f"\n--- exhaustive leaf comparison ---")
    print(f"  {len(keys):,} leaves compared, {len(deep):,} differ")
    for k in deep[:25]:
        print(f"    {RED}DIFF{RESET} {k}: ref={fr.get(k)}  new={fn_.get(k)}")
    if len(deep) > 25:
        print(f"    ... and {len(deep) - 25:,} more")

    total = cfg_diff + len(deep)
    print("\n" + "=" * 86)
    if total == 0:
        print(f"{GREEN}IDENTICAL{RESET} — the clean package reproduces the reference exactly "
              f"({len(keys):,} leaves + {len(cn)} config fields).")
    else:
        print(f"{YELLOW if args.expect_diff else RED}{total:,} DIFFERENCE(S){RESET} "
              f"— {cfg_diff} config, {len(deep):,} result leaves.")
    print("=" * 86)

    if args.expect_diff:
        return 0 if total else 1
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
