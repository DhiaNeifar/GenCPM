# Shared settings of the reproduce/ scripts. Paths come from configs/paths.yaml (see configs/paths.example.yaml).
set -eo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
PY=${PY:-python}
GC="$PY -m gencpm"
eval "$($GC paths --shell)"
[ -n "$WORK" ] || { echo "set 'work' in configs/paths.yaml"; exit 1; }
mkdir -p "$WORK/logs" "$WORK/results" "$WORK/paper"
WORKERS=${WORKERS:-$(( $(nproc) > 4 ? $(nproc) - 2 : 2 ))}
ATTACKS_SINGLE=$(cat "$REPO/configs/attacks_single.json")
# work directories (caches, defenses) and generated CPMs, one per collection
W_PP=$WORK/carla_pp;          CPM_PP=$WORK/carla_pp_cpm
W_COORD=$WORK/carla_coordinated
W_SECOND=$WORK/carla_second;  CPM_SECOND=$WORK/carla_second_cpm
W_COOP=$WORK/cooperscene;     CPM_COOP=$WORK/cooperscene_cpm
RES=$WORK/results

log_step() {  # log_step NAME: everything below goes to the terminal and to $WORK/logs/NAME.log
  exec > >(tee -a "$WORK/logs/$1.log") 2>&1
  echo "[$(date +%F' '%H:%M)] start $1"
}

# object gates written by `fit` / `transfer` (object-level filtering in late fusion)
gates() {  # gates WORKDIR [with_transfer]
  local D=$1/defenses_W20
  echo "$D/objgate_plausibility_seed0.npz $D/objgate_consistency_seed0.npz $D/objgate_PILD_seed0.npz $D/objgate_PILD_seed1.npz $D/objgate_PILD_seed2.npz"
  [ -n "$2" ] && echo "$D/objgate_PILD-transfer_seed0.npz $D/objgate_PILD-transfer_seed1.npz $D/objgate_PILD-transfer_seed2.npz"
  true
}

defend() {  # defend WORKDIR: receiver features, rules + PILD + GBDT per fold (models saved), rule baselines
  OMP_NUM_THREADS=6 $GC features --work-dir "$1"
  OMP_NUM_THREADS=6 $GC fit --work-dir "$1" --seeds 0 1 2
  OMP_NUM_THREADS=6 $GC baselines --work-dir "$1" --seq-len 20 --seeds 0 1 2
}

transfer() {  # transfer WORKDIR [OUT_NAME TAG]: defenses saved by carla.sh (CARLA, PointPillars), applied without retraining
  OMP_NUM_THREADS=6 $GC transfer --work-dir "$1" --pretrained "$W_PP" --seeds 0 1 2
  OMP_NUM_THREADS=6 $GC apply --work-dir "$1" --source-dir "$W_PP/temporal_W20_global" --seq-len 20 --seeds 0 1 2 \
    ${2:+--out-name "$2"} ${3:+--tag-suffix=$3}
}

train_temporal() {  # train_temporal WORKDIR [SEQ_LEN]: one process per fold, PAR folds at a time on the GPU
  local W=$1 SEQ=${2:-20} PAR=${PAR:-4}
  $PY - "$W" "$SEQ" <<'PYEOF' & local T=$!
import glob, sys, time
from tqdm import tqdm
w, seq = sys.argv[1], sys.argv[2]
bar = tqdm(total=60, desc=f"temporal models W={seq}", unit="model")
while True:
    bar.n = len(glob.glob(f"{w}/temporal_W{seq}_global/*_probs.npz")); bar.refresh()
    if bar.n >= 60:
        break
    time.sleep(5)
PYEOF
  local pids=() k
  for k in 0 1 2 3 4; do
    while [ "$(jobs -rp | grep -vx "$T" | wc -l)" -ge "$PAR" ]; do sleep 2; done
    OMP_NUM_THREADS=4 $GC train --work-dir "$W" --seq-len "$SEQ" --seeds 0 1 2 --only-folds $k > "$W/train_W${SEQ}_fold$k.log" 2>&1 &
    pids+=($!)
  done
  local rc=0 p; for p in "${pids[@]}"; do wait "$p" || rc=1; done
  kill "$T" 2>/dev/null || true; echo
  [ $rc -eq 0 ] || { for k in 0 1 2 3 4; do tail -3 "$W/train_W${SEQ}_fold$k.log"; done; return 1; }
}

fusion() {  # fusion OUT WORKDIR "DETECTORS" "TEMPORAL_DIRS" "GATES" [extra fusion-run options]
  local OUT=$1 W=$2 DET=$3 TD=$4 GT=$5; shift 5
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $GC fusion run --work-dir "$W" --temporal-dir $TD \
    --detectors $DET --seeds 0 1 2 --threshold-rule 'FPR<=5%' --object-gates $GT \
    --defense-features "$W/defense_features.npz" --runs r000 r001 --out-dir "$RES/$OUT" --workers "$WORKERS" \
    --chunks 4 --overwrite "$@"
  $GC fusion report --out-dir "$RES/$OUT" --bootstrap 1000 --report-workers 8 | tail -3
}

robosac() {  # robosac OUT WORKDIR: consensus selection adapted from ROBOSAC (calibrated on benign traffic)
  [ -d "$RES/robosac_calib" ] || $GC fusion robosac-calib --work-dir "$W_PP" --out-dir "$RES/robosac_calib" --workers "$WORKERS"
  OMP_NUM_THREADS=1 $GC fusion run --work-dir "$2" --temporal-dir none --object-gates --detectors none \
    --robosac-calib "$RES/robosac_calib" --threshold-rule 'FPR<=5%' --runs r000 --ts-stride 2 --robosac-samples 8 \
    --out-dir "$RES/$1" --workers "$WORKERS" --chunks 4 --overwrite
  $GC fusion report --out-dir "$RES/$1" --bootstrap 1000 --report-workers 8 | tail -3
}
