"""Geometry shared by the late-fusion evaluation: sender poses as homogeneous transforms and the evaluation range.

Poses follow the CARLA convention used by the released frames, [x, y, z, roll, yaw, pitch] with angles in degrees
(the convention of OPV2V-style datasets). The range mask keeps boxes whose eight corners all lie within |x| <= X and
|y| <= Y in the ego frame, as in OpenCOOD late-fusion evaluation.
"""

import numpy as np

COMM_RANGE_M = 70.0         # a CAV takes part in the ego's late fusion if it is within this distance
EVAL_RANGE_XY = (140.0, 40.0)


def pose_to_matrix(pose):
    """4x4 transform from a sender frame to the world, CARLA rotation convention (yaw, then pitch, then roll)."""
    x, y, z, roll, yaw, pitch = pose[:]
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))
    m = np.identity(4)
    m[0, 3], m[1, 3], m[2, 3] = x, y, z
    m[0, 0] = c_p * c_y
    m[0, 1] = c_y * s_p * s_r - s_y * c_r
    m[0, 2] = -c_y * s_p * c_r - s_y * s_r
    m[1, 0] = s_y * c_p
    m[1, 1] = s_y * s_p * s_r + c_y * c_r
    m[1, 2] = -s_y * s_p * c_r + c_y * s_r
    m[2, 0] = s_p
    m[2, 1] = -c_p * s_r
    m[2, 2] = c_p * c_r
    return m


def boxes_in_range(corners, range_xy=EVAL_RANGE_XY):
    """Mask of the boxes (torch tensor (N, 8, 3)) whose corners all lie within the evaluation range."""
    import torch
    rx, ry = range_xy
    lo = torch.tensor([-rx, -ry], dtype=torch.float32).reshape(1, 1, -1).to(corners.device)
    hi = torch.tensor([rx, ry], dtype=torch.float32).reshape(1, 1, -1).to(corners.device)
    return torch.all(torch.all(corners[:, :, :2] >= lo, dim=-1) & torch.all(corners[:, :, :2] <= hi, dim=-1), dim=-1)
