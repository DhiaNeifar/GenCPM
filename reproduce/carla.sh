# Single-attacker collection on the released CARLA frames (PointPillars): Tables 6 and 8 (top block), Figs. 6 and 7.
# Usage: bash reproduce/carla.sh STEP        steps in order:
#   frames     cache the released frames                                     ~10 min
#   generate   benign + attacked CPMs, eight attacks, two realizations       ~30 min
#   cache      CPM cache with content labels                                 ~10 min
#   defend     receiver features, rules, PILD, GBDT, rule baselines           ~30 min
#   temporal   LSTM, GRU, BiLSTM, CNN-LSTM, 5 folds x 3 runs (GPU)            ~30 min
#   fusion     end-to-end late fusion of every detector + ROBOSAC consensus   ~1-2 h
#   windows    temporal models with window lengths 3, 5, 10 (Fig. 7, GPU)    ~1.5 h
#   all        every step above
source "$(dirname "$0")/common.sh"
STEP=$1; [ -n "$STEP" ] || { sed -n 2,12p "$0"; exit 1; }
[ -n "$CARLA_FRAMES" ] || { echo "set carla_frames in configs/paths.yaml"; exit 1; }
run() {
  case $1 in
    frames)   $GC frames --work-dir "$W_PP" --data-root "$CARLA_FRAMES" --workers "$WORKERS" ;;
    generate) $GC generate "$CARLA_FRAMES" "$CPM_PP" --repeats 2 --seed 777 --workers "$WORKERS" --attack-probs "$ATTACKS_SINGLE" ;;
    cache)    $GC cache --work-dir "$W_PP" --cpm-root "$CPM_PP" --workers "$WORKERS" --label-rule content ;;
    defend)   defend "$W_PP" ;;
    temporal) train_temporal "$W_PP" 20 ;;
    fusion)   $GC summarize --work-dir "$W_PP" | tail -12
              fusion carla_pp "$W_PP" 'LSTM GRU BiLSTM CNN-LSTM GBDT-rules plausibility consistency PILD' \
                'temporal_W20_global baselines_W20 defenses_W20' "$(gates "$W_PP")"
              robosac carla_pp_robosac "$W_PP" ;;
    windows)  for L in 3 5 10; do train_temporal "$W_PP" $L; done ;;
    *) echo "unknown step $1"; exit 1 ;;
  esac
}
log_step "carla_$STEP"
if [ "$STEP" = all ]; then for s in frames generate cache defend temporal fusion windows; do run $s; done; else run "$STEP"; fi
echo "[$(date +%F' '%H:%M)] done  carla $STEP"
