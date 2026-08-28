#!/usr/bin/env python
"""
Conference reproduction gate -- ICECE 2026, CICIDS2017 Wednesday, recommended arm.

Compares a fresh `metrics_multiday_*.json` (results.global block) against the
2026-08-27 run this bundle ships (`_REFERENCE/metrics_wed_gateenf_CONFERENCE_ANCHOR.json`).

Unlike the UNSW bundle's gate, this anchor has NOT yet been independently
reproduced from a second from-scratch run -- treat a mismatch as a signal to
investigate (R retraining is the noisiest part of this leg), not necessarily
as an error in your run. A near-tie caveat already applies to the weight
argmax here (runner-up gap 2.386e-05) -- expect W*/wR* to occasionally land
on the runner-up (H .20 / wR .40) on a different machine's R retrain.

    python verify_conference.py
    python verify_conference.py metrics_multiday_wed_gateenf.json

With no argument it checks metrics_multiday_wed_gateenf.json (written next to run_pipeline.py,
not under out/ -- only the trained IDS model goes there), which is what
    python run_pipeline.py --day Wednesday --drop-identity --policy-score formula \
           --aar-global 0.01 --bdr-global 0.005 --gate-mode enforce --suffix _wed_gateenf
writes. Exit 0 = every gated field matches. Exit 1 = drift -- read the note above
before assuming the run is wrong.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANCHOR = HERE / "_REFERENCE" / "metrics_wed_gateenf_CONFERENCE_ANCHOR.json"
# run_pipeline.py writes metrics_multiday*.json into its OWN directory (HERE), not out/ --
# only the trained IDS model (model_external_ids.json) goes under out/. Confirmed against
# run_pipeline.py's actual write call after this default pointed at the wrong path on a real
# VM run (2026-08-28).
DEFAULT_RUN = HERE / "metrics_multiday_wed_gateenf.json"

GATED_METRICS = ["ADR", "UAR", "precision", "recall", "F1", "accuracy", "roc",
                 "BAR_restrict", "BAR_deny", "attack_deny_rate"]
GATED_CONFUSION = ["TP", "FP", "FN", "TN", "n", "n_attack", "n_benign"]
GATED_CONFIG = ["W", "wR", "w_min", "drop_identity", "policy_score", "gate_mode",
                "aar_global", "bdr_global", "h_mode", "R_test_roc", "val_additive_roc"]


def main() -> int:
    if len(sys.argv) > 2:
        print(__doc__)
        return 2
    new_p = Path(sys.argv[1]) if len(sys.argv) == 2 else DEFAULT_RUN
    if not new_p.is_file():
        print(f"no such file: {new_p}")
        if len(sys.argv) == 1:
            print("Run the RUNBOOK.md step-3 command first.")
        return 2
    if not ANCHOR.is_file():
        print(f"anchor missing: {ANCHOR}")
        return 2

    new = json.loads(new_p.read_text(encoding="utf-8"))
    ref = json.loads(ANCHOR.read_text(encoding="utf-8"))
    new_g = new["results"]["global"]
    ref_g = ref["results"]["global"]

    bad = []
    for k in GATED_METRICS:
        a, b = ref_g.get(k), new_g.get(k)
        if a != b:
            bad.append(("results.global." + k, a, b))
    for k in GATED_CONFUSION:
        a, b = ref_g.get("confusion", {}).get(k), new_g.get("confusion", {}).get(k)
        if a != b:
            bad.append(("results.global.confusion." + k, a, b))
    for k in GATED_CONFIG:
        a, b = ref["config"].get(k), new["config"].get(k)
        if a != b:
            bad.append(("config." + k, a, b))

    print(f"anchor : {ANCHOR.name}")
    print(f"run    : {new_p.name}")
    print()
    if bad:
        print(f"*** {len(bad)} FIELD(S) DIFFER ***")
        for k, a, b in bad:
            print(f"  {k:<32} anchor={a!r}  run={b!r}")
        print()
        print("Read the module docstring before treating this as a bug: the")
        print("weight argmax here is a near-tie (runner-up gap 2.386e-05) and R is")
        print("retrained (not frozen) for a Wednesday-only run, so W*/wR* CAN")
        print("legitimately move on different hardware. Everything else should not.")
        return 1

    print("*** REPRODUCED - every gated field matches the conference anchor ***")
    print(f"  W* = {ref['config']['W']}   wR* = {ref['config']['wR']}")
    print(f"  ADR {ref_g['ADR']} | UAR {ref_g['UAR']} | P {ref_g['precision']} | "
          f"F1 {ref_g['F1']} | ROC {ref_g['roc']}")
    c = ref_g["confusion"]
    print(f"  TP {c['TP']} | FP {c['FP']} | FN {c['FN']} | TN {c['TN']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
