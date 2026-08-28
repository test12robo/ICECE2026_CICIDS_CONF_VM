"""Sequential per-identity trust-trajectory kernel -- VENDORED from
`LANL_PDP_CLEAN/lanl_workspace/pipeline/phase8_trust_trajectory.py` (`_trajectory_kernel` +
`_group_boundaries`), by exact extraction, 2026-08-21 -- NOT retyped from memory, same
provenance discipline as this leg's own `pdp_core.py` ("vendored trust-equation primitives").

Why this is reusable as-is: the kernel takes plain numpy arrays (I/B/D/C/R_init/time_sec +
group boundaries) and has NO LANL-specific column names or logic inside it. LANL's own
docstring anticipated exactly this reuse: "Identity for the EMA: `src_user` on LANL. On
UNSW/CSE-CIC-IDS2018 the identity will be `src_ip` via the same kernel, different column name
(configurable via --identity-col)." This file ships that reuse for CICIDS (`source_ip`).

    H(t)      = alpha * T_init(t-1) + (1 - alpha) * H(t-1)       (legacy_ema; eq. 12)
    T_base(t) = 0.2 * (I + B + D + C + H(t))
    T_init(t) = T_base(t) * (1 - R_init(t))^gamma * exp(-lambda * dt_hours)

Other h_mode values (level/signed_dev/abs_dev) are carried over from the source for parity but
CICIDS's `run_pipeline.py` already has its own, simpler, non-sequential implementation of those
three (pure Polars expanding-mean expressions) -- this kernel is wired up ONLY for `legacy_ema`,
which is the one mode that genuinely needs the sequential self-referential recursion.

Only ONE thing is CICIDS-specific here, and it lives in `run_pipeline.py`, not in this file:
`R_init`'s PROFILE_FEATURES are CICIDS's own 7 sub-factors (`sf_access_rate` etc.), not LANL's
11 -- the formula and constants (IQR floor, scale, attack/normal weights) are shared, the raw
inputs are dataset-native, matching every other cross-leg factor in this project.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

ALPHA = 0.10
GAMMA = 0.35
H_COLD_START = 0.5

H_MODES = ("legacy_ema", "level", "signed_dev", "abs_dev")
H_MODE_CODES = {name: i for i, name in enumerate(H_MODES)}


@njit(parallel=True, cache=True, fastmath=False)
def trajectory_kernel(
    time_sec,            # int64   [n]   per-row (epoch seconds)
    entity_code,         # int32   [n]   factorised identity (sorted by entity, time)
    I_arr,               # float32 [n]
    B_arr,               # float32 [n]
    D_arr,               # float32 [n]
    C_arr,               # float32 [n]
    R_init_arr,          # float32 [n]
    group_starts,        # int64   [G]   start index per identity group
    group_ends,          # int64   [G]   end index per identity group (exclusive)
    state_H_init,        # float32 [G]   initial H (h_cold_start for a single-shot run)
    state_T_init_init,   # float32 [G]   initial T_init
    state_time_prev,     # int64   [G]   initial last time_sec
    state_has_prev,      # bool    [G]   True if this entity has prior state (False here)
    state_sum_base,      # float64 [G]   initial running SUM of base
    state_cnt_base,      # int64   [G]   initial running COUNT of base
    alpha,               # float
    gamma,               # float
    lambda_per_hour,     # float
    h_cold_start,        # float
    h_mode,              # int64   0=legacy_ema 1=level 2=signed_dev 3=abs_dev
    # outputs
    H_out,               # float32 [n]
    T_base_out,          # float32 [n]
    T_init_out,          # float32 [n]
    dt_out,              # float32 [n]
    state_H_final,       # float32 [G]   out: H after last event
    state_T_init_final,  # float32 [G]   out: T_init after last event
    state_time_final,    # int64   [G]   out: time_sec of last event
    state_sum_base_final,# float64 [G]   out: running sum of base after last event
    state_cnt_base_final,# int64   [G]   out: running count of base after last event
):
    """Identical to LANL's `_trajectory_kernel` -- see that file's docstring for the full
    per-row math and causality argument. Reproduced verbatim; only parameter names generalised
    (`user_code`->`entity_code`)."""
    n_groups = group_starts.shape[0]
    for g in prange(n_groups):
        s = group_starts[g]
        e = group_ends[g]
        H            = state_H_init[g]
        T_init_prev  = state_T_init_init[g]
        time_prev    = state_time_prev[g]
        has_prev     = state_has_prev[g]
        sum_base     = state_sum_base[g]
        cnt_base     = state_cnt_base[g]

        for i in range(s, e):
            t = time_sec[i]

            if has_prev:
                dt_hours = (t - time_prev) / 3600.0
            else:
                dt_hours = 0.0

            base_now = 0.25 * (I_arr[i] + B_arr[i] + C_arr[i] + D_arr[i])

            if h_mode == 0:
                if has_prev:
                    H_new = alpha * T_init_prev + (1.0 - alpha) * H
                else:
                    H_new = h_cold_start
            else:
                if cnt_base > 0:
                    mu = sum_base / cnt_base
                    if h_mode == 1:
                        H_new = mu
                    elif h_mode == 2:
                        H_new = 0.5 + (base_now - mu)
                    else:
                        d = base_now - mu
                        if d < 0.0:
                            d = -d
                        H_new = 1.0 - d
                    if H_new < 0.0:
                        H_new = 0.0
                    elif H_new > 1.0:
                        H_new = 1.0
                else:
                    H_new = h_cold_start

            T_base = 0.2 * (I_arr[i] + B_arr[i] + D_arr[i] + C_arr[i] + H_new)

            one_minus_r = 1.0 - R_init_arr[i]
            if one_minus_r < 1e-9:
                one_minus_r = 1e-9
            risk_factor = one_minus_r ** gamma
            decay = np.exp(-lambda_per_hour * dt_hours)
            T_init = T_base * risk_factor * decay

            H_out[i]      = np.float32(H_new)
            T_base_out[i] = np.float32(T_base)
            T_init_out[i] = np.float32(T_init)
            dt_out[i]     = np.float32(dt_hours)

            H           = H_new
            T_init_prev = T_init
            time_prev   = t
            has_prev    = True
            sum_base   += base_now
            cnt_base   += 1

        state_H_final[g]        = np.float32(H)
        state_T_init_final[g]   = np.float32(T_init_prev)
        state_time_final[g]     = time_prev
        state_sum_base_final[g] = sum_base
        state_cnt_base_final[g] = cnt_base


def group_boundaries(sorted_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-group [start, end) index arrays for an int array sorted by group id.
    Identical to LANL's `_group_boundaries`, reproduced verbatim."""
    n = sorted_ids.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    diff = np.empty(n, dtype=bool)
    diff[0] = True
    diff[1:] = sorted_ids[1:] != sorted_ids[:-1]
    starts = np.flatnonzero(diff).astype(np.int64)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = n
    return starts, ends
