# GenCPM

Code of **AdverCPM**, a benchmark of benign and adversarial Cooperative Perception Messages (CPMs) for connected
vehicles. From per-frame LiDAR detections of each connected vehicle, GenCPM builds the CPMs it would transmit, applies
insider attacks, trains and runs receiver-side defenses, and measures their effect end to end through late fusion.

| Package | Paper | What it does |
|---|---|---|
| `gencpm/cpm` | GenCPM, Sec. IV-B | builds one CPM per vehicle and frame from its detections; schedules and applies attacks |
| `gencpm/attacks` | MalCPM, Sec. IV-C | eight single-attacker attacks (CF, CS, CSD, ISP, GT, SD, RP, PF) and four coordinated attacks (CG, CSUP, CCSD, CISP) |
| `gencpm/defenses` | ResilienCPM, Sec. IV-D | plausibility and consistency rules, temporal models, gradient-boosted trees, PILD; transfer to other collections |
| `gencpm/data` | Sec. V | caches of frames and CPMs, content labels, episode folds |
| `gencpm/evaluation` | Sec. VI | late fusion and AP with a C++ core, detection metrics, attack sweeps |
| `gencpm/report` | Sec. VI, appendix | the tables and figures of the paper |
| `gencpm/perception` | optional | LiDAR detection for new datasets (needs OpenCOOD); not needed to reproduce the paper |

## Install

```bash
git clone https://github.com/DhiaNeifar/GenCPM.git && cd GenCPM
conda env create -f environment.yml && conda activate gencpm
pip install -e ".[test]"          # also compiles the C++ fusion core
pytest tests                      # about one second
```

A GPU is used for the temporal models and PILD when available; everything else runs on the CPU.

## Data

The AdverCPM release on Hugging Face ([DhiaNeifar/AdverCPM](https://huggingface.co/datasets/DhiaNeifar/AdverCPM))
holds the perception frames (per-frame detections of every connected vehicle, with PointPillars and SECOND). Download it and tell GenCPM where it is:

```bash
python -m gencpm download --to /path/to/AdverCPM       # about 2 GB; --parts frames/pointpillars for the minimum
cp configs/paths.example.yaml configs/paths.yaml       # configs/paths.yaml is git-ignored
# edit configs/paths.yaml: carla_frames, carla_frames_second, work
python -m gencpm paths                                 # shows what was configured
```

The frames are organized as `<carla_frames>/Town*/<episode>/<vehicle>/<frame>_preds.yaml` (detections, ground truth,
vehicle pose) with `<frame>_score.npy` (detection scores). [docs/data_format.md](docs/data_format.md) describes the
frames, the generated CPMs, and the work directories. Every path comes from `configs/paths.yaml` or from the command
line; the repository contains no machine-specific path.

## Reproduce the paper

Each script runs one experiment in steps, with progress bars and a log in `<work>/logs/`. Run a script without
arguments to see its steps, or with `all` to run them in order.

| Result | Command | Approximate time (one workstation, one GPU) |
|---|---|---|
| Quick check of the installation | `bash reproduce/smoke.sh` | ~10 min |
| Single-attacker collection: Tables 6 and 8 (top), Figs. 6 and 7 | `bash reproduce/carla.sh all` | ~5 h |
| Coordinated attacks: Table 8 (bottom) | `bash reproduce/coordinated.sh all` | ~3 h |
| Second perception model, SECOND: Table 9 | `bash reproduce/second.sh all` | ~2 h |
| Real data, CooperScene: Table 10 ([docs/cooperscene.md](docs/cooperscene.md)) | `bash reproduce/cooperscene.sh all` | ~20 min |
| Number of attackers and parameter sensitivity: Figs. 5 and 8 | `bash reproduce/appendix.sh all` | ~6 h |
| All tables and figures from the results above | `bash reproduce/report.sh` | ~5 min |
| Everything, in order | `bash reproduce/all.sh` | ~1 day |

`coordinated.sh`, `second.sh`, and `cooperscene.sh` use the defenses trained by `carla.sh`, so run `carla.sh` first.
The tables are written as LaTeX and the figures as PNG to `<work>/paper/`.

Every training step saves its models in `<work>` (temporal models, PILD, gradient-boosted trees, per fold and run);
`coordinated.sh`, `second.sh` and `cooperscene.sh` load the ones `carla.sh` saved instead of retraining them.

## Command line

Every stage is one command of `python -m gencpm` (`python -m gencpm --help` lists them):

```text
download    get the release from Hugging Face      paths        show configs/paths.yaml
frames      cache the released frames              generate     benign and attacked CPMs
cache       CPM cache with content labels          coordinated  coordinated-attack collection
features    receiver-side features                 fit          rules, PILD, gradient-boosted trees per fold
train       temporal models per fold               baselines    rule baselines, gradient-boosted trees
transfer    apply PILD and trees trained elsewhere apply        apply trained temporal models
summarize   detection metrics                      fusion       end-to-end late fusion: run | report | robosac-calib
sweep       attack sweeps: run | report            table-*, figures*   tables and figures
```

## Repository layout

```text
gencpm/        the library (see the table above); gencpm/evaluation/csrc holds the C++ fusion core
configs/       paths.example.yaml, the attack set, the sweep configurations, the CooperScene detector config
reproduce/     one script per experiment of the paper
tests/         unit tests: attacks, C++ fusion core against a Python reference, geometry, no machine paths
docs/          data format, CooperScene
```

## License

MIT, see [LICENSE](LICENSE).
