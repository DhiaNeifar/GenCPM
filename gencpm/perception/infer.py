"""
Fast single-CAV inference for several OpenCOOD late-fusion detectors in one pass over the data.

Outputs per frame, next to the input or in a mirror tree: <ts>_preds.yaml, <ts>_pred.npy, <ts>_score.npy, <ts>_gt.npy),
but:
  * every point cloud and annotation is read once and fed to all detectors (the data sit on a slow disk),
  * preprocessing (YAML parsing, PCD reading, voxelization, target generation) runs in DataLoader worker processes,
  * YAML is parsed and written with the libyaml C bindings, outputs are written by a thread pool.
Detection ids restart at 1000000 for every episode and follow the original (CAV, timestamp) order.

Model k writes its outputs next to the frames under --out-root[k]; for an out-root different from the data root,
the frame's .yaml and .pcd are symlinked there first (mirror layout used for the second detector).

Usage:
  python -m gencpm infer --data-root DATA --episodes Town01/ep1 ... --model-dir DETECTOR_DIR --out-root OUT
Requires OpenCOOD (pip install -e <OpenCOOD>) and a detector folder with config.yaml and net_epoch*.pth.
"""

import argparse
from collections import OrderedDict
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import yaml

from gencpm.evaluation.geometry import pose_to_matrix

FLOAT_RE = re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X)


class FastLoader(getattr(yaml, "CLoader", yaml.Loader)):
    """libyaml loader with the same float resolver as opencood.hypes_yaml.yaml_utils.load_yaml."""


FastLoader.add_implicit_resolver(u'tag:yaml.org,2002:float', FLOAT_RE, list(u'-+0123456789.'))
DUMPER = getattr(yaml, "CSafeDumper", yaml.SafeDumper)


def build_model_and_dataset(model_dir, data_root_one_episode):
    from opencood.hypes_yaml import yaml_utils
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils

    class Opt:  # yaml_utils.load_yaml reads <model_dir>/config.yaml when given an opt with model_dir
        pass
    opt = Opt(); opt.model_dir = model_dir
    hypes = yaml_utils.load_yaml(None, opt)
    hypes['root_dir'] = hypes['validate_dir'] = data_root_one_episode
    hypes['scenario'], hypes['cav_id'] = 'probe', None
    dataset = build_dataset(hypes, visualize=False, train=False)
    model = train_utils.create_model(hypes)
    model.cuda()
    _, model = train_utils.load_saved_model(model_dir, model)
    model.eval()
    return hypes, dataset, model



# ---- helpers: detections to CPM-style object dicts, single-CAV batches, point clouds ----

def _rotation_z(yaw_rad):
    c = np.cos(yaw_rad)
    s = np.sin(yaw_rad)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def _yaw_from_rot(rot):
    return np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))


def _build_detected_objects(pred_box_tensor,
                            pred_score,
                            cav_pose,
                            order,
                            score_threshold,
                            id_start):
    detected_objects = OrderedDict()

    if pred_box_tensor is None or pred_box_tensor.shape[0] == 0:
        return detected_objects, id_start

    pred_np = pred_box_tensor.detach().cpu().numpy()
    pred_scores = pred_score.detach().cpu().numpy() \
        if pred_score is not None else None

    # (N, 7): x, y, z, h, w, l, yaw (for hwl order)
    from opencood.utils import box_utils
    box_centers = box_utils.corner_to_center(pred_np, order=order)

    cav_to_world = pose_to_matrix(cav_pose)
    cav_rot = cav_to_world[:3, :3]

    for i in range(box_centers.shape[0]):
        if pred_scores is not None and score_threshold is not None:
            if pred_scores[i] < score_threshold:
                continue

        x, y, z, h, w, l, yaw = box_centers[i].tolist()

        center_world = cav_to_world @ np.array([x, y, z, 1.0])

        box_rot = _rotation_z(yaw)
        world_rot = cav_rot @ box_rot
        yaw_world_deg = float(_yaw_from_rot(world_rot))

        extent = [float(abs(l) / 2.0),
                  float(abs(w) / 2.0),
                  float(abs(h) / 2.0)]

        # Actor-local bbox center is unknown for predictions.
        center = [0.0, 0.0, float(extent[2])]
        location = [float(center_world[0] - center[0]),
                    float(center_world[1] - center[1]),
                    float(center_world[2] - center[2])]

        detected_objects[id_start] = {
            'angle': [0.0, yaw_world_deg, 0.0],
            'center': center,
            'extent': extent,
            'location': location,
        }
        id_start += 1

    return detected_objects, id_start


def _to_native_types(obj):
    if isinstance(obj, dict):
        return {_to_native_types(k): _to_native_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native_types(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return _to_native_types(obj.tolist())
    return obj


def _make_single_cav_batch(dataset, selected_cav_processed):
    anchor_box = selected_cav_processed['anchor_box']
    if anchor_box is not None:
        anchor_box = torch.from_numpy(np.array(anchor_box))
    object_bbx_center = torch.from_numpy(
        np.array([selected_cav_processed['object_bbx_center']]))
    object_bbx_mask = torch.from_numpy(
        np.array([selected_cav_processed['object_bbx_mask']]))
    processed_lidar = dataset.pre_processor.collate_batch(
        [selected_cav_processed['processed_lidar']])
    label_dict = dataset.post_processor.collate_batch(
        [selected_cav_processed['label_dict']])
    return {
        'ego': {
            'anchor_box': anchor_box,
            'object_bbx_center': object_bbx_center,
            'object_bbx_mask': object_bbx_mask,
            'processed_lidar': processed_lidar,
            'label_dict': label_dict,
            'object_ids': selected_cav_processed['object_ids'],
            'transformation_matrix': torch.eye(4).float(),
        }
    }


def _pcd_has_intensity_field(pcd_file):
    with open(pcd_file, 'rb') as f:
        for _ in range(12):
            line = f.readline().decode('ascii', 'ignore').split()
            if line and line[0].upper() == 'FIELDS':
                fields = [x.lower() for x in line[1:]]
                return 'intensity' in fields and 'rgb' not in fields and 'rgba' not in fields
    return False


def pcd_fields_to_np(pcd_file, intensity_clip=255.0):
    """
    Read x, y, z, intensity directly from the fields of an ASCII or binary pcd.
    Raw integer intensity is clipped at `intensity_clip` and scaled to [0, 1],
    the range of the colour-channel intensity of OPV2V.
    """
    header, n_header, n_lines = {}, 0, 0
    with open(pcd_file, 'rb') as f:
        for line in f:
            n_header += len(line); n_lines += 1
            parts = line.decode('ascii', 'ignore').split()
            if parts and not parts[0].startswith('#'):
                header[parts[0].upper()] = parts[1:]
            if parts and parts[0].upper() == 'DATA':
                break
    fields = [x.lower() for x in header['FIELDS']]
    n_points = int(header['POINTS'][0])
    if header['DATA'][0] == 'ascii':
        import pandas as pd
        data = pd.read_csv(pcd_file, sep=' ', skiprows=n_lines, header=None, usecols=range(len(fields)),
                           nrows=n_points, dtype=np.float32, comment='#').to_numpy()
        data = data[:, :len(fields)]
    elif header['DATA'][0] == 'binary':
        kinds = {('F', '4'): '<f4', ('F', '8'): '<f8', ('I', '1'): 'i1', ('I', '2'): '<i2', ('I', '4'): '<i4',
                 ('U', '1'): 'u1', ('U', '2'): '<u2', ('U', '4'): '<u4'}
        dtype = np.dtype([(n, kinds[(t, s)]) for n, t, s in zip(fields, header['TYPE'], header['SIZE'])])
        with open(pcd_file, 'rb') as f:
            f.seek(n_header)
            rec = np.frombuffer(f.read(n_points * dtype.itemsize), dtype=dtype, count=n_points)
        data = np.stack([rec[n].astype(np.float32) for n in fields], axis=1)
    else:
        raise ValueError(f'unsupported pcd DATA type {header["DATA"]} in {pcd_file}')
    xyz = data[:, [fields.index('x'), fields.index('y'), fields.index('z')]]
    if 'intensity' in fields:
        intensity = np.clip(data[:, fields.index('intensity')], 0, intensity_clip) / intensity_clip
    else:
        intensity = np.zeros(len(xyz), dtype=np.float32)
    keep = np.isfinite(xyz).all(axis=1)
    return np.hstack((xyz[keep], intensity[keep, None])).astype(np.float32)


def read_point_cloud(path):
    """(N, 4) float32 x, y, z, intensity. OPV2V/CARLA files keep the intensity in the colour channel (read with open3d);
    files with an `intensity` field (e.g. CooperScene) are read directly, intensity clipped at 255 and scaled to [0, 1]."""
    if _pcd_has_intensity_field(path):
        return pcd_fields_to_np(path)
    import open3d as o3d
    pc = o3d.io.read_point_cloud(path)
    xyz, colors = np.asarray(pc.points), np.asarray(pc.colors)
    return np.hstack((xyz, colors[:, :1])).astype(np.float32)

class Frames(torch.utils.data.Dataset):
    def __init__(self, items, datasets):
        self.items, self.datasets = items, datasets

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ep, cav, ts, yaml_path, pcd_path = self.items[i]
        with open(yaml_path) as f:
            params = yaml.load(f, Loader=FastLoader)
        lidar_np = read_point_cloud(pcd_path)
        batches = []
        for ds in self.datasets:
            proc = ds.get_item_single_car({'lidar_np': lidar_np, 'params': params})
            batches.append(_make_single_cav_batch(ds, proc))
        return i, params, batches


def write_outputs(out_dir, ts, params, detected_objects, pred_box, scores, gt_boxes, link_from=None):
    os.makedirs(out_dir, exist_ok=True)
    if link_from is not None:  # mirror layout: symlink the source frame
        for ext in (".yaml", ".pcd"):
            dst = os.path.join(out_dir, ts + ext)
            if not os.path.lexists(dst):
                os.symlink(link_from + ext, dst)
    out = dict(params)
    out['detected_objects'] = detected_objects
    out = _to_native_types(out)
    with open(os.path.join(out_dir, f"{ts}_preds.yaml"), "w") as f:
        yaml.dump(out, f, Dumper=DUMPER, sort_keys=False, default_flow_style=False)
    if pred_box is not None:
        np.save(os.path.join(out_dir, f"{ts}_pred.npy"), pred_box)
    if scores is not None:
        np.save(os.path.join(out_dir, f"{ts}_score.npy"), scores)
    if gt_boxes is not None:
        np.save(os.path.join(out_dir, f"{ts}_gt.npy"), gt_boxes)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--episodes", nargs="+", required=True, help="episode paths relative to --data-root")
    ap.add_argument("--model-dir", nargs="+", required=True)
    ap.add_argument("--out-root", nargs="+", required=True, help="one per model")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--skip-done", action="store_true", help="skip frames whose outputs exist for every model")
    ap.add_argument("--max-frames", type=int, default=None, help="testing: only the first N frames")
    ap.add_argument("--link-root", default=None,
                    help="mirror symlinks point to <link-root>/<episode>/... instead of the read location "
                         "(use when --data-root is a temporary staged copy)")
    a = ap.parse_args()
    assert len(a.model_dir) == len(a.out_root)
    from opencood.tools import train_utils
    from opencood.utils import box_utils
    from tqdm import tqdm

    # build_dataset scans its whole root, so give it a one-frame probe (symlinks) instead of the data root.
    import tempfile
    probe_root = tempfile.mkdtemp(prefix="infer_probe_")
    ep0 = os.path.join(a.data_root, a.episodes[0])
    cav0 = sorted(x for x in os.listdir(ep0) if os.path.isdir(os.path.join(ep0, x)))[0]
    ts0 = sorted(x for x in os.listdir(os.path.join(ep0, cav0)) if re.fullmatch(r"\d{6}\.yaml", x))[0][:6]
    os.makedirs(os.path.join(probe_root, "probe", cav0))
    for ext in (".yaml", ".pcd"):
        os.symlink(os.path.join(ep0, cav0, ts0 + ext), os.path.join(probe_root, "probe", cav0, ts0 + ext))
    models = [build_model_and_dataset(m, probe_root) for m in a.model_dir]
    datasets = [m[1] for m in models]

    items = []
    for ep in a.episodes:
        ep_dir = os.path.join(a.data_root, ep)
        for cav in sorted(x for x in os.listdir(ep_dir) if os.path.isdir(os.path.join(ep_dir, x))):
            cdir = os.path.join(ep_dir, cav)
            for name in sorted(x for x in os.listdir(cdir)
                               if x.endswith(".yaml") and "additional" not in x and "_preds" not in x):
                ts = name[:-5]
                pcd = os.path.join(cdir, ts + ".pcd")
                if not os.path.exists(pcd):
                    continue
                if a.skip_done and all(os.path.exists(os.path.join(o, ep, cav, f"{ts}_preds.yaml")) for o in a.out_root):
                    continue
                items.append((ep, cav, ts, os.path.join(cdir, name), pcd))
    if a.max_frames:
        items = items[:a.max_frames]
    print(f"{len(items):,} frames in {len(a.episodes)} episodes, {len(models)} models, {a.workers} workers", flush=True)

    # Read-ahead hints for the slow disk: tell the kernel which files come next so it can queue and order the reads.
    import threading
    progress = {"i": 0}

    def prefetch(ahead=400):
        j = 0
        while j < len(items):
            if j > progress["i"] + ahead:
                time.sleep(0.05); continue
            for path in (items[j][3], items[j][4]):
                try:
                    fd = os.open(path, os.O_RDONLY)
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
                    os.close(fd)
                except OSError:
                    pass
            j += 1
    threading.Thread(target=prefetch, daemon=True).start()

    loader = torch.utils.data.DataLoader(Frames(items, datasets), batch_size=None, shuffle=False,
                                         num_workers=a.workers, prefetch_factor=4, persistent_workers=False)
    id_start = {k: {} for k in range(len(models))}
    pool = ThreadPoolExecutor(8)
    futures = []
    t0 = time.time()
    for i, params, batches in tqdm(loader, total=len(items), unit="frame", mininterval=30, dynamic_ncols=True):
        ep, cav, ts, yaml_path, _ = items[i]
        progress["i"] = i
        pose = params['lidar_pose']
        for k, (hypes, ds, model) in enumerate(models):
            with torch.no_grad():
                b = train_utils.to_device(batches[k], torch.device("cuda"))
                out = {'ego': model(b['ego'])}
                pred_box, pred_score = ds.post_processor.post_process(b, out)
            start = id_start[k].get(ep, 1000000)
            detected, id_start[k][ep] = _build_detected_objects(pred_box, pred_score, pose,
                                                                hypes['postprocess']['order'], None, start)
            gt = None
            c, m = b['ego']['object_bbx_center'], b['ego']['object_bbx_mask']
            if c is not None and m is not None:
                c = c[0][m[0] == 1].detach().cpu().numpy()
                if c.size:
                    gt = box_utils.boxes_to_corners_3d(c, hypes['postprocess']['order'])
            pb = pred_box.detach().cpu().numpy() if pred_box is not None else None
            sc = pred_score.detach().cpu().numpy() if pred_score is not None else None
            out_dir = os.path.join(a.out_root[k], ep, cav)
            if os.path.abspath(a.out_root[k]) == os.path.abspath(a.data_root):
                link = None
            elif a.link_root:
                link = os.path.join(a.link_root, ep, cav, ts)
            else:
                link = yaml_path[:-5]
            futures.append(pool.submit(write_outputs, out_dir, ts, params, detected, pb, sc, gt, link))
        if len(futures) > 2000:
            for f in futures:
                f.result()
            futures = []
    for f in futures:
        f.result()
    pool.shutdown()
    dt = time.time() - t0
    print(f"done: {len(items):,} frames in {dt / 60:.1f} min ({1000 * dt / max(len(items), 1):.1f} ms/frame)", flush=True)


if __name__ == "__main__":
    main()
