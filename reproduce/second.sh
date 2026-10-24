# Second perception model, SECOND: Table 9. Needs carla_frames_second in configs/paths.yaml and carla.sh up to temporal.
# Usage: bash reproduce/second.sh STEP   steps: frames | generate | cache | defend | transfer | fusion | all
source "$(dirname "$0")/common.sh"
STEP=$1; [ -n "$STEP" ] || { sed -n 2,3p "$0"; exit 1; }
[ -n "$CARLA_FRAMES_SECOND" ] || { echo "set carla_frames_second in configs/paths.yaml"; exit 1; }
run() {
  case $1 in
    frames)   $GC frames --work-dir "$W_SECOND" --data-root "$CARLA_FRAMES_SECOND" --workers "$WORKERS" ;;
    generate) $GC generate "$CARLA_FRAMES_SECOND" "$CPM_SECOND" --repeats 2 --seed 777 --workers "$WORKERS" --attack-probs "$ATTACKS_SINGLE" ;;
    cache)    $GC cache --work-dir "$W_SECOND" --cpm-root "$CPM_SECOND" --workers "$WORKERS" --label-rule content ;;
    defend)   defend "$W_SECOND" ;;
    transfer) transfer "$W_SECOND"
              $GC summarize --work-dir "$W_SECOND" | tail -15 ;;
    fusion)   fusion carla_second "$W_SECOND" 'CNN-LSTM GBDT-rules GBDT-transfer plausibility consistency PILD PILD-transfer' \
                'temporal_W20_global baselines_W20 defenses_W20' "$(gates "$W_SECOND" transfer)" ;;
    *) echo "unknown step $1"; exit 1 ;;
  esac
}
log_step "second_$STEP"
if [ "$STEP" = all ]; then for s in frames generate cache defend transfer fusion; do run $s; done; else run "$STEP"; fi
echo "[$(date +%F' '%H:%M)] done  second $STEP"
