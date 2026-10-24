"""The C++ late-fusion core against the Python/shapely reference: box filters, rotated NMS, range mask, TP/FP."""
import numpy as np
import pytest

pytest.importorskip("shapely")
from gencpm.evaluation import fusion_core  # noqa: E402
from gencpm.evaluation.geometry import pose_to_matrix  # noqa: E402
from reference_fusion import nms_rotated_fast, tp_fp_fast  # noqa: E402

IOUS = np.array([0.3, 0.5, 0.7])


def random_boxes(rng, n):
    """Boxes in the released format: location, center, half extent, angle (roll, yaw, pitch in degrees)."""
    loc = np.c_[rng.uniform(-150, 150, n), rng.uniform(-45, 45, n), rng.uniform(-2.5, -1.0, n)]
    ext = np.c_[rng.uniform(1.5, 3.5, n), rng.uniform(0.7, 1.3, n), rng.uniform(0.6, 1.2, n)]
    ang = np.c_[np.zeros(n), rng.uniform(-180, 180, n), np.zeros(n)]
    return np.c_[loc, np.zeros((n, 2)), ext[:, 2:], ext, ang].astype(np.float64)


def reference(corners, scores, gt, nms=0.15, rx=140.0, ry=40.0, z0=-3.0, z1=1.0, ext=6.0):
    mn, mx = corners.min(1), corners.max(1)
    keep = (mx[:, 0] - mn[:, 0] <= ext) & (mx[:, 1] - mn[:, 1] <= ext) & (mx[:, 2] - mn[:, 2] != 0) \
        & (mn[:, 2] >= z0) & (mx[:, 2] <= z1)
    c, s = corners[keep], scores[keep]
    k = nms_rotated_fast(c, s, nms)
    c, s = c[k], s[k]
    inr = np.all((c[:, :, 0] >= -rx) & (c[:, :, 0] <= rx) & (c[:, :, 1] >= -ry) & (c[:, :, 1] <= ry), axis=1)
    sc, tp, _ = tp_fp_fast(c[inr], s[inr], gt, IOUS)
    return sc, tp


@pytest.mark.parametrize("seed", range(20))
def test_fuse_and_score_matches_reference(seed):
    rng = np.random.default_rng(seed)
    pose = [rng.uniform(-50, 50), rng.uniform(-50, 50), 0.0, 0.0, rng.uniform(-180, 180), 0.0]
    world_to_ego = np.linalg.inv(pose_to_matrix(pose))
    gt_boxes = random_boxes(rng, 30)
    det = np.r_[gt_boxes + rng.normal(0, 0.3, gt_boxes.shape) * np.r_[[1, 1, 0] + [0] * 9], random_boxes(rng, 40)]
    corners = fusion_core.project(det, world_to_ego, True)
    gt = fusion_core.project(gt_boxes, world_to_ego, True)
    scores = rng.uniform(0.2, 1.0, len(corners)).astype(np.float32)
    fusion_core.set_filters(140, 40, -3, 1, 6)
    sc, tp = fusion_core.fuse_and_score(corners, scores, gt, 0.15, IOUS)
    rsc, rtp = reference(corners, scores, gt)
    np.testing.assert_array_equal(sc, rsc)
    np.testing.assert_array_equal(tp, rtp)


def test_filters_are_configurable_and_reset():
    fusion_core.set_filters(102.4, 51.2, -4, 2, 6)
    assert fusion_core.get_filters() == pytest.approx((102.4, 51.2, -4.0, 2.0, 6.0))
    fusion_core.set_filters(140, 40, -3, 1, 6)
    assert fusion_core.get_filters() == (140.0, 40.0, -3.0, 1.0, 6.0)
