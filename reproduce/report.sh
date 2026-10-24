# Tables and figures of the paper from the results of the other scripts, written to <work>/paper.
# Parts whose results are missing are skipped with a note.
source "$(dirname "$0")/common.sh"
OUT=$WORK/paper; mkdir -p "$OUT"
have() { [ -f "$1" ] || { echo "skip: $1 missing"; return 1; }; }
$GC summarize --work-dir "$W_PP" > /dev/null
have "$W_COORD/cache_summary.json" && $GC summarize --work-dir "$W_COORD" --dirs temporal_W20_global temporal_W20_transfer baselines_W20 defenses_W20 > /dev/null
have "$W_COORD/cache_summary.json" && $GC table-distribution --work "$W_PP" --coord-work "$W_COORD" --out "$OUT/attack_statistics.tex"
have "$RES/carla_coordinated/fusion_report.csv" && $GC table-main --work "$W_PP" --coord-work "$W_COORD" \
  --fusion "$RES/carla_pp" "$RES/carla_pp_robosac" --coord-fusion "$RES/carla_coordinated" "$RES/carla_coordinated_robosac" \
  --out "$OUT/results_main.tex"
if have "$RES/carla_second/fusion_report.csv"; then
  $GC summarize --work-dir "$W_SECOND" > /dev/null
  $GC table-second --pp-work "$W_PP" --second-work "$W_SECOND" --pp-fusion "$RES/carla_pp,$RES/carla_pp_robosac" \
    --second-fusion "$RES/carla_second" --out "$OUT/revision_second.tex"
fi
if have "$RES/cooperscene/fusion_report.csv"; then
  $GC summarize --work-dir "$W_COOP" --dirs temporal_W20_transfer baselines_W20 defenses_W20 > /dev/null
  $GC table-cooperscene --bare --work "$W_COOP" --fusion "$RES/cooperscene" --out "$OUT/cooperscene.tex"
fi
have "$RES/carla_coordinated/fusion_report.csv" && $GC figures --work "$W_PP" --coord-work "$W_COORD" \
  --fusion "$RES/carla_pp/fusion_report.csv" --coord-fusion "$RES/carla_coordinated/fusion_report.csv" --out-dir "$OUT"
have "$RES/sweeps_single/sweep_report.csv" && $GC figures-appendix --report "$RES/sweeps_single/sweep_report.csv" \
  --configs configs/sweeps/single_attacker.json --coord-report "$RES/sweeps_coordinated/sweep_report.csv" \
  --coord-configs configs/sweeps/coordinated.json --out-dir "$OUT"
echo "tables and figures in $OUT"
