# Data format

## Perception frames (input)

```text
<frames root>/Town*/<episode>/<vehicle>/<frame>_preds.yaml
                                        <frame>_score.npy
```

One folder per connected vehicle and episode; frame numbers are simulator ticks of 50 ms, every second tick (10 Hz).
Each `_preds.yaml` holds what one vehicle perceived in one frame:

| Field | Content |
|---|---|
| `lidar_pose` | sender pose `[x, y, z, roll, yaw, pitch]`, world frame, degrees |
| `ego_speed` | sender speed (m/s) |
| `detected_objects` | the vehicle's LiDAR detections: `{id: {location, center, extent, angle}}`, world frame |
| `vehicles` | ground-truth vehicles in sensor range, same box format |

Boxes use the OPV2V convention: `location` is the box bottom, `center` the offset to the box center, `extent` the half
sizes `(length, width, height)`, and `angle` `[roll, yaw, pitch]` in degrees. `_score.npy` holds the detection scores
in the order of `detected_objects`. Other datasets can be used once they are in this layout (see
[cooperscene.md](cooperscene.md)).

## Generated CPMs

`python -m gencpm generate FRAMES OUT` writes, per episode, one YAML file per vehicle, frame, and attack realization:
`OUT/Town*/<episode>/r<realization>_<vehicle>_<frame>.yaml`, plus `simulation_config_r<realization>.yaml` with the
generation settings.

| Field | Content |
|---|---|
| `vehicle_id`, `timestamp` | sender and frame |
| `global_position`, `vehicle_speed` | reported sender position and speed |
| `detected_vehicles` | transmitted objects: `id`, `track_id`, `location`, `center`, `extent`, `orientation`, `speed` |
| `detected_objects` | the same objects in the frame format above |
| `cpm_type` | `benign` or `malicious` as scheduled |
| `attacks` | for a malicious CPM: `attack_type`, `attack_parameters`, `target_ids`, `altered_fields`, `ground_truth_reference` |

The label used by every experiment is the *content* label computed by `python -m gencpm cache`: a CPM is malicious
only if what it transmits, its object list or its reported position, differs from what its sender perceived.

## Work directories

Each collection gets one work directory (set `work` in `configs/paths.yaml`):

| File or folder | Written by | Content |
|---|---|---|
| `frames_meta.pkl`, `frames_arrays.npz` | `frames` | detections, scores, ground truth, poses of every frame |
| `cpm_meta.pkl`, `cpm_objects.npz`, `cpm_boxes.npz`, `cache_summary.json` | `cache` | transmitted CPMs and content labels |
| `defense_features.npz` | `features` | receiver-side object and CPM features |
| `defenses_W20/` | `fit`, `transfer` | rule, PILD and gradient-boosted-tree scores per fold, object gates |
| `baselines_W20/`, `temporal_W20_global/` | `baselines`, `train` | rule baselines and temporal models per fold |
| `detection_summary.csv` | `summarize` | AUC, precision, recall, F1, FPR, FNR of every detector |

End-to-end results go to `<work>/results/<experiment>/fusion_report.csv` (AP@0.3/0.5/0.7 per condition and subset,
with episode-bootstrap confidence intervals).
