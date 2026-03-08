# GenCPM for AdverCPM

This repository contains the **GenCPM implementation** for the paper:

**AdverCPM: A Cooperative Perception Messages Adversarial Dataset in Connected Autonomous Vehicle Systems**

GenCPM takes raw CARLA simulation outputs and generates CPMs (Cooperative Perception Messages) with benign/malicious labels and attack metadata for downstream analysis and model training.

## 1) Source Dataset (Hugging Face)

Raw data is hosted at:

- https://huggingface.co/datasets/DhiaNeifar/AdverCPM

Inside the dataset, use the folder **`Carla Dataset`**. It contains multiple compressed archives (`Town0x.rar`).

### Required extraction outcome

After downloading and extracting all `Town0x.rar` files, you should have a structure like:

```text
dataset/
  Town01/
    2026_03_02_17_35_12/
      2026_03_02_17_35_12.txt
      <CAV_ID_1>/
        000060.yaml
        000060_preds.yaml
        000060_pred.npy
        000060_score.npy
        000060_gt.npy
        000060_front.png
        000060_left.png
        000060_right.png
        000060_rear.png
        000060.pcd
        ...
      <CAV_ID_2>/
        ...
  Town02/
    ...
```

Notes:

- Each scenario folder is timestamp-named (for example `2026_03_02_17_35_12`).
- Each scenario contains one scenario-level `.txt` file and multiple CAV folders.
- In many scenarios, frame timestamps start around `000060` and continue with fixed time steps.
- `t.yaml` stores the ego-vehicle and surrounding-scene state for that timestamp.
- `t_preds.yaml` stores the same context plus detection output (`detected_objects`) used by GenCPM.
- `t_pred.npy`, `t_score.npy`, and `t_gt.npy` are inference artifacts from 3D object detection.

## 2) Generate CPM (GenCPM)

Run CPM generation in parallel from the extracted dataset:

```bash
python run_simulations_parallel.py <RAW_DATASET_ROOT> <CPM_OUTPUT_ROOT> \
  --workers 20 \
  --repeats 1 \
  --dt 0.05 \
  --lambda-attacks 0.2 \
  --mean-duration 5.0 \
  --prob-drift 0.25 \
  --prob-add 0.25 \
  --prob-remove 0.25 \
  --prob-whitenoise 0.25 \
  --seed 42
```

Example:

```bash
python run_simulations_parallel.py ./dataset ./CPM --workers 20 --repeats 1 --dt 0.05 --lambda-attacks 0.2 --mean-duration 5.0 --prob-drift 0.25 --prob-add 0.25 --prob-remove 0.25 --prob-whitenoise 0.25 --seed 42
```

### Expected CPM output

```text
CPM/
  Town01/
    2026_03_02_17_35_12/
      simulation_config_r000.yaml
      r000_<cav_id>_<timestamp>.yaml
      r000_<cav_id>_<timestamp>.yaml
      ...
  Town02/
    ...
```

If `--repeats > 1`, additional run prefixes appear (`r001_...`, `r002_...`, etc.), all inside the same scenario folder.

The exact CPM schema is described in the scientific article. In generated files, `cpm_type` is the main label (`benign` or `malicious`) and `attacks` carries applied attack metadata.

## 3) Attack Implementations

Attack logic is implemented in:

- `attacks/drift_attack.py (Instantaneous Spatial Perturbation (ISP))`
- `attacks/white_noise_attack.py (Cumulative Spatial Drift (CSD))`
- `attacks/add_object_attack.py (Content Fabrication (CF))`
- `attacks/remove_object_attack.py (Content Suppression (CS)`
- `attacks/registry.py`

Generation/scheduling flow is in:

- `run_simulation.py`
- `run_simulations_parallel.py`

## 4) Analysis and Training (Notebooks)

### Main notebook

- `Notebooks/CPM.ipynb`

This is the primary workflow notebook. It is used to:

- Load raw CPM files into `df_cpm` and `df_veh`.
- Perform exploratory CPM statistics.
- Build sequence datasets for learning.
- Run window-size analysis (W sweep).
- Train/evaluate deep models.
- Save trained models and generate figures.

### Table-generation scripts

- `Notebooks/TownStats.py`
  - Generates **per-map scenario statistics** (mean ± std over runs).
  - Example:

```bash
python ./Notebooks/TownStats.py ./CPM --non-connected-mode vehicles
```

- `Notebooks/RoadTypeStats.py` (RoadStats / road-type statistics)
  - Generates **road-type statistics** table.
  - Example:

```bash
python ./Notebooks/RoadTypeStats.py --path ./CPM
```

## 5) Models and Architectures

Model builders are defined in `Notebooks/CPM.ipynb` and include:

- LSTM
- GRU
- BiLSTM
- CNN-LSTM
- (plus transformer variants available in the notebook)

Grid training saves legacy-compatible artifacts per window/model:

```text
models/window_grid/
  W_03/
    LSTM.h5
    LSTM.json
    LSTM.weights.h5
    ...
  W_05/
    ...
```

`.h5` + JSON/weights are provided for compatibility with older TensorFlow/Keras environments.

## 6) Figures

Window/model comparison figures are generated under:

- `figures/window_grid/window_model_summary.png`
- `figures/window_grid/roc_grid_by_window.png`

![Window/Model Summary](/Notebooks/figures/window_grid/auc_vs_window_percent.png)
![ROC Grid by Window](/Notebooks/figures/window_grid/roc_window_3.png)
![ROC Grid by Window](/Notebooks/figures/window_grid/roc_window_5.png)
![ROC Grid by Window](/Notebooks/figures/window_grid/roc_window_10.png)
![ROC Grid by Window](/Notebooks/figures/window_grid/roc_window_20.png)

## 7) Environment

At minimum, install:

- `tensorflow`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `pyyaml`
- `tqdm`
- `jupyter`

Example:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib pyyaml tqdm jupyter
```

## 8) Citation

If you use this code or dataset, please cite the AdverCPM paper and dataset release.
