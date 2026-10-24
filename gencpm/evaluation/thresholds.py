"""Operating points chosen on validation data: max-F1, false-positive rate <= 1% and <= 5%."""

from __future__ import annotations
from typing import Dict
import numpy as np


def _rates(y, p, thr):
    pred = p >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "fpr": fpr, "fnr": 1 - rec,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def select_thresholds(y_val, p_val) -> Dict[str, float]:
    from sklearn.metrics import precision_recall_curve, roc_curve
    pr, rc, th = precision_recall_curve(y_val, p_val)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    thr = {"maxF1": float(th[np.argmax(f1)])}
    fpr, tpr, th2 = roc_curve(y_val, p_val)
    for target in (0.01, 0.05):
        ok = np.where(fpr <= target)[0]
        thr[f"FPR<={int(target * 100)}%"] = float(th2[ok[np.argmax(tpr[ok])]]) if len(ok) else float("inf")
    return thr
