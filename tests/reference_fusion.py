"""
Python/shapely reference of the late-fusion primitives, used to test the C++ core of gencpm.evaluation:
rotated NMS and greedy TP/FP matching on BEV IoUs computed with shapely, IoUs cast to float32 before any comparison.
"""

import numpy as np
import shapely


def _polygons(corners: np.ndarray) -> np.ndarray:
    """(N, 8, 3) or (N, 4, 2) corners -> shapely polygons of the first four (x, y) corners."""
    return shapely.polygons(np.ascontiguousarray(corners[:, :4, :2], dtype=np.float64))


def _iou_dense(pa, pb):
    inter = shapely.area(shapely.intersection(pa[:, None], pb[None, :]))
    union = shapely.area(shapely.union(pa[:, None], pb[None, :]))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (inter / union).astype(np.float32)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """BEV IoU of every box in a with every box in b. Polygon geometry is computed only for pairs whose bounding
    circles overlap; every other pair cannot intersect and gets exactly the dense result (0), so the output is
    identical to the dense computation."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    qa, qb = np.asarray(a)[:, :4, :2].astype(np.float64), np.asarray(b)[:, :4, :2].astype(np.float64)
    ca, cb = qa.mean(1), qb.mean(1)
    ra = np.sqrt(((qa - ca[:, None]) ** 2).sum(-1)).max(1)
    rb = np.sqrt(((qb - cb[:, None]) ** 2).sum(-1)).max(1)
    d = np.sqrt(((ca[:, None, :] - cb[None, :, :]) ** 2).sum(-1))
    near = d <= (ra[:, None] + rb[None, :]) * (1 + 1e-9) + 1e-9
    pa, pb = _polygons(a), _polygons(b)
    degenerate = (shapely.area(pa)[:, None] == 0) | (shapely.area(pb)[None, :] == 0)
    near |= degenerate  # keep the dense result (possibly NaN) for degenerate boxes
    out = np.zeros((len(a), len(b)), np.float32)
    i, j = np.nonzero(near)
    if len(i):
        inter = shapely.area(shapely.intersection(pa[i], pb[j]))
        union = shapely.area(shapely.union(pa[i], pb[j]))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i, j] = (inter / union).astype(np.float32)
    return out


def nms_rotated_fast(corners: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    if corners.shape[0] == 0:
        return np.array([], dtype=np.int32)
    iou = iou_matrix(corners, corners)
    ixs = scores.argsort()[::-1][:1000]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        remove = np.where(iou[i, ixs[1:]] > threshold)[0] + 1
        ixs = np.delete(ixs, remove)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def tp_fp_fast(det_corners, det_scores, gt_corners, thresholds=(0.3, 0.5, 0.7)):
    """Returns (scores sorted descending, tp array (len(thresholds), n_det), n_gt)."""
    n_gt = 0 if gt_corners is None else int(gt_corners.shape[0])
    if det_corners is None or det_corners.shape[0] == 0:
        return np.zeros(0, np.float32), np.zeros((len(thresholds), 0), np.int8), n_gt
    order = np.argsort(-det_scores)
    scores_sorted = det_scores[order].astype(np.float32)
    tp = np.zeros((len(thresholds), len(order)), np.int8)
    if n_gt == 0:
        return scores_sorted, tp, n_gt
    iou = iou_matrix(det_corners, gt_corners)
    for t, thr in enumerate(thresholds):
        alive = np.ones(n_gt, bool)
        for k, d in enumerate(order):
            if not alive.any():
                continue
            row = np.where(alive, iou[d], -1.0)
            j = int(np.argmax(row))
            if row[j] < thr:
                continue
            tp[t, k] = 1
            alive[j] = False
    return scores_sorted, tp, n_gt
