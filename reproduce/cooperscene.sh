# Real data, CooperScene: Table 10. Needs cooperscene_frames in configs/paths.yaml (see docs/cooperscene.md) and
# carla.sh up to temporal (the CARLA-trained defenses are applied without retraining).
# Usage: bash reproduce/cooperscene.sh STEP   steps: frames | generate | cache | defend | transfer | fusion | all
source "$(dirname "$0")/common.sh"
STEP=$1; [ -n "$STEP" ] || { sed -n 2,4p "$0"; exit 1; }
[ -n "$COOPERSCENE_FRAMES" ] || { echo "set cooperscene_frames in configs/paths.yaml"; exit 1; }
run() {
  case $1 in
    frames)   $GC frames --work-dir "$W_COOP" --data-root "$COOPERSCENE_FRAMES" --workers "$WORKERS" ;;
    generate) $GC generate "$COOPERSCENE_FRAMES" "$CPM_COOP" --repeats 2 --seed 777 --workers "$WORKERS" --attack-probs "$ATTACKS_SINGLE" ;;
    cache)    $GC cache --work-dir "$W_COOP" --cpm-root "$CPM_COOP" --workers "$WORKERS" --label-rule content ;;
    defend)   defend "$W_COOP" ;;
    transfer) transfer "$W_COOP" temporal_W20_transfer -transfer
              $GC summarize --work-dir "$W_COOP" --dirs temporal_W20_transfer baselines_W20 defenses_W20 | tail -22 ;;
    # evaluation range of the CooperScene detector (+/-102.4 x +/-51.2 m) and a height window of -4..2 m
    fusion)   fusion cooperscene "$W_COOP" 'LSTM-transfer GRU-transfer BiLSTM-transfer CNN-LSTM-transfer GBDT-transfer PILD-transfer GBDT-rules PILD plausibility consistency' \
                'temporal_W20_transfer baselines_W20 defenses_W20' "$(gates "$W_COOP" transfer)" --range-xy 102.4 51.2 --z-window -4 2 ;;
    *) echo "unknown step $1"; exit 1 ;;
  esac
}
log_step "cooperscene_$STEP"
if [ "$STEP" = all ]; then for s in frames generate cache defend transfer fusion; do run $s; done; else run "$STEP"; fi
echo "[$(date +%F' '%H:%M)] done  cooperscene $STEP"
