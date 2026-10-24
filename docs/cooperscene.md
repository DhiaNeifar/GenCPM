# Real data: CooperScene

GenCPM runs on any dataset once its frames are in the layout of [data_format.md](data_format.md). For CooperScene
(three connected vehicles and one roadside unit, 10 Hz), the frames are produced in two steps, then the experiment is
the same as for CARLA.

## 1. Detections

The released CARLA frames already contain detections; CooperScene provides LiDAR only, so a detector has to run first.
This optional step needs [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) installed in the environment, `open3d`
(`pip install -e ".[perception]"`), and the PointPillars late-fusion detector trained on CooperScene, released in
`models/detectors/cooperscene_pointpillars` (its `config.yaml` is `configs/detectors/cooperscene_pointpillars.yaml`;
set `root_dir` and `validate_dir` there to train one yourself with OpenCOOD).

```bash
python -m gencpm infer --data-root /path/to/cooperscene --episodes validate/13 validate/14 test/2 test/8 \
    --model-dir /path/to/AdverCPM/models/detectors/cooperscene_pointpillars \
    --out-root /path/to/cooperscene_detections --workers 8
```

CooperScene point clouds store the intensity as a separate field; GenCPM reads it directly, clipped at 255 and scaled
to [0, 1].

## 2. Layout

```bash
python -m gencpm cooperscene-layout --src /path/to/cooperscene_detections --dst /path/to/cooperscene_frames --mode copy
```

This cuts each scene into episodes of 20 to 25 s (the 5-fold protocol needs at least five episodes) and numbers the
frames like the CARLA frames, two ticks of 50 ms per frame, so every time constant of the attacks and defenses is in
seconds as in CARLA.

## 3. Experiment

Set `cooperscene_frames: /path/to/cooperscene_frames` in `configs/paths.yaml`, run `reproduce/carla.sh` up to
`temporal` (the CARLA-trained defenses are applied without retraining), then

```bash
bash reproduce/cooperscene.sh all
bash reproduce/report.sh          # Table 10
```

The evaluation range is that of the CooperScene detector, 102.4 m by 51.2 m around the ego.
