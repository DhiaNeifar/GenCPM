# Everything in the paper from the released frames, in order. Takes about a day on one GPU workstation.
set -e
D=$(dirname "$0")
bash "$D/carla.sh" all
bash "$D/coordinated.sh" all
[ -z "$(python -m gencpm paths --shell | sed -n 's/^CARLA_FRAMES_SECOND=//p' | tr -d "'")" ] || bash "$D/second.sh" all
[ -z "$(python -m gencpm paths --shell | sed -n 's/^COOPERSCENE_FRAMES=//p' | tr -d "'")" ] || bash "$D/cooperscene.sh" all
bash "$D/appendix.sh" all
bash "$D/report.sh"
