# Controlled attack studies: number of attackers (Fig. 5) and attack-parameter sensitivity (Fig. 8).
# Needs carla.sh frames. Usage: bash reproduce/appendix.sh STEP   steps: single | coordinated | all
source "$(dirname "$0")/common.sh"
STEP=$1; [ -n "$STEP" ] || { sed -n 2,3p "$0"; exit 1; }
sweep() {  # sweep NAME CONFIGS CHUNK
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $GC sweep run --frames-work "$W_PP" --configs "$2" \
    --workers "$WORKERS" --chunk "$3" --out-dir "$RES/$1"
  $GC sweep report --out-dir "$RES/$1" --bootstrap 1000 --workers 4 | tail -5
}
run() {
  case $1 in
    single)      sweep sweeps_single "$REPO/configs/sweeps/single_attacker.json" 6 ;;
    coordinated) sweep sweeps_coordinated "$REPO/configs/sweeps/coordinated.json" 4 ;;
    *) echo "unknown step $1"; exit 1 ;;
  esac
}
log_step "appendix_$STEP"
if [ "$STEP" = all ]; then for s in single coordinated; do run $s; done; else run "$STEP"; fi
echo "[$(date +%F' '%H:%M)] done  appendix $STEP"
