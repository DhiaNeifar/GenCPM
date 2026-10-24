"""Plausibility and cross-sender consistency checks: each feature is calibrated on benign training objects, and a
check is violated above the 99th percentile of benign values (ResilienCPM rule families)."""

from __future__ import annotations
import numpy as np
from gencpm.defenses.features import CPM_FEATS, OBJ_FEATS


class TailCalibrator:
    """Maps a feature to its empirical CDF over benign *training* values (NaN -> 0 = no evidence)."""

    def fit(self, x):
        x = x[np.isfinite(x)]
        self.q = np.sort(x) if len(x) else np.zeros(1)
        return self

    def __call__(self, x):
        out = np.searchsorted(self.q, np.nan_to_num(x, nan=-np.inf), side="right") / len(self.q)
        return np.where(np.isfinite(x), out, 0.0)


VIOL_Q = 0.99  # a check is violated above the 99th percentile of benign training values


def rule_scores(F, C, obj_row, n_cpm, cal, feats_obj, feats_cpm, checkable_only=False):
    """Object score: max over its checks of the benign tail CDF (+ small mean term to break ties).
    CPM score: max(fraction of objects violating a check, CPM-level check excess above VIOL_Q)."""
    tails = [cal[f](F[:, OBJ_FEATS.index(f)]) for f in feats_obj]
    if "is_new" in cal:  # new objects appearing close to the sender
        new_near = np.where(F[:, OBJ_FEATS.index("is_new")] > 0, -F[:, OBJ_FEATS.index("dist_sender")], np.nan)
        tails.append(cal["is_new"](new_near))
    T = np.stack(tails, 1)
    so = T.max(1) + 1e-3 * T.mean(1)
    # Consistency: only objects that another CAV could check carry evidence (denominator).
    w = (F[:, OBJ_FEATS.index("n_check")] > 0).astype(float) if checkable_only else np.ones(len(F))
    cnt = np.bincount(obj_row, weights=w, minlength=n_cpm)
    viol = np.bincount(obj_row, weights=w * (T.max(1) > VIOL_Q), minlength=n_cpm)
    frac = viol / np.maximum(cnt, 1)
    cpm_ex = np.zeros(n_cpm)
    for f in feats_cpm:
        cpm_ex = np.maximum(cpm_ex, np.clip((cal[f](C[:, CPM_FEATS.index(f)]) - VIOL_Q) / (1 - VIOL_Q), 0, 1))
    mean_so = np.bincount(obj_row, weights=so, minlength=n_cpm) / np.maximum(cnt, 1)
    sc = np.maximum(frac, cpm_ex) + 1e-3 * mean_so
    return so, sc
