# Coordinated-attack collection (groups of three colluders): Table 8 (bottom block). Needs carla.sh frames, cache,
# defend and temporal (detectors trained on single-attacker attacks are transferred).
# Usage: bash reproduce/coordinated.sh STEP   steps: generate | defend | temporal | transfer | fusion | all
source "$(dirname "$0")/common.sh"
STEP=$1; [ -n "$STEP" ] || { sed -n 2,4p "$0"; exit 1; }
run() {
  case $1 in
    generate) $GC coordinated --frames-work "$W_PP" --out "$W_COORD" --workers "$WORKERS" --common-targets \
                --label-rule content --duty 1.0 | tail -12 ;;
    defend)   defend "$W_COORD" ;;
    temporal) train_temporal "$W_COORD" 20 ;;
    transfer) transfer "$W_COORD" temporal_W20_transfer -transfer
              $GC summarize --work-dir "$W_COORD" --dirs temporal_W20_global temporal_W20_transfer baselines_W20 defenses_W20 | tail -18 ;;
    fusion)   fusion carla_coordinated "$W_COORD" 'LSTM GRU BiLSTM CNN-LSTM LSTM-transfer GRU-transfer BiLSTM-transfer CNN-LSTM-transfer GBDT-rules GBDT-transfer plausibility consistency PILD PILD-transfer' \
                'temporal_W20_global temporal_W20_transfer baselines_W20 defenses_W20' "$(gates "$W_COORD" transfer)"
              robosac carla_coordinated_robosac "$W_COORD" ;;
    *) echo "unknown step $1"; exit 1 ;;
  esac
}
log_step "coordinated_$STEP"
if [ "$STEP" = all ]; then for s in generate defend temporal transfer fusion; do run $s; done; else run "$STEP"; fi
echo "[$(date +%F' '%H:%M)] done  coordinated $STEP"
