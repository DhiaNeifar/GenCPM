# Quick end-to-end check of the installation on 5 episodes (one per town): frames, CPMs with all eight attacks,
# labels, defenses, a short temporal training, late fusion, and a report. About 10 minutes; the numbers are not the
# paper's (five episodes, one run, two training epochs), only a check that every stage runs.
source "$(dirname "$0")/common.sh"
[ -n "$CARLA_FRAMES" ] || { echo "set carla_frames in configs/paths.yaml"; exit 1; }
S=$WORK/smoke; rm -rf "$S"; mkdir -p "$S/frames"
for town in "$CARLA_FRAMES"/Town*; do
  ep=$(ls "$town" | head -1); mkdir -p "$S/frames/$(basename "$town")"; ln -s "$town/$ep" "$S/frames/$(basename "$town")/$ep"
done
log_step smoke
W=$S/work
$GC frames --work-dir "$W" --data-root "$S/frames" --workers "$WORKERS"
$GC generate "$S/frames" "$S/cpm" --repeats 2 --seed 777 --workers "$WORKERS" --attack-probs "$ATTACKS_SINGLE" | tail -1
$GC cache --work-dir "$W" --cpm-root "$S/cpm" --workers "$WORKERS" --label-rule content | tail -1
OMP_NUM_THREADS=6 $GC features --work-dir "$W"
OMP_NUM_THREADS=6 $GC fit --work-dir "$W" --seeds 0
OMP_NUM_THREADS=6 $GC baselines --work-dir "$W" --seq-len 20 --seeds 0
OMP_NUM_THREADS=4 $GC train --work-dir "$W" --seq-len 20 --seeds 0 --models LSTM --epochs 2 --patience 1
$GC summarize --work-dir "$W" | tail -8
D=$W/defenses_W20
OMP_NUM_THREADS=1 $GC fusion run --work-dir "$W" --temporal-dir temporal_W20_global baselines_W20 defenses_W20 \
  --detectors LSTM GBDT-rules PILD plausibility consistency --seeds 0 --threshold-rule 'FPR<=5%' \
  --object-gates "$D/objgate_plausibility_seed0.npz" "$D/objgate_consistency_seed0.npz" "$D/objgate_PILD_seed0.npz" \
  --defense-features "$W/defense_features.npz" --runs r000 r001 --out-dir "$S/fusion" --workers "$WORKERS" --overwrite | tail -1
$GC fusion report --out-dir "$S/fusion" --boot-device cpu --bootstrap 100 | tail -2
echo "smoke test passed: every stage ran (results in $S)"
