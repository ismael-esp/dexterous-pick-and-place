#!/usr/bin/env bash
# Isaac Lab pick-and-place benchmark: N episodes per object per arm (+ optional one video each).
# Resumable: objects whose log already has a result, and videos that exist, are skipped.
#
#   ISAAC_PY=~/miniconda3/envs/env_isaaclab/bin/python bash scripts/run_sim_benchmark.sh
#   ARMS="la" OBJS="mustard milk" NEP=5 VIDEO=0 GPU=0 bash scripts/run_sim_benchmark.sh
#
# Env vars: ISAAC_PY (Isaac Lab python), BODEX_SERVICE_PYTHON (bodex env python), GPU (index),
#           ARMS, OBJS, NEP (episodes per object), VIDEO (0/1), OUT (output dir).
set -o pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ISAAC_PY="${ISAAC_PY:?set ISAAC_PY to the python of your Isaac Lab env}"
BODEX_PY="${BODEX_SERVICE_PYTHON:?set BODEX_SERVICE_PYTHON to the python of the bodex env}"
GPU="${GPU:-0}"
OUT="${OUT:-$ROOT/runs/sim_benchmark}"
LOGS="$OUT/logs"; VIDS="$OUT/videos"; mkdir -p "$LOGS" "$VIDS"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$BODEX_PY")")/lib:${LD_LIBRARY_PATH:-}"
ARMS=(${ARMS:-ra la})
OBJS=(${OBJS:-green_cube red_cube pikachu whale snorlax tiger milk mustard green_can tea_peach})
NEP=${NEP:-10}
VIDEO=${VIDEO:-1}
# pin Isaac's renderer to the same GPU as CUDA (multi-GPU machines)
KIT="--kit_args=--/renderer/activeGpu=$GPU --/renderer/multiGpu/enabled=false"

rm -f /tmp/bodex_service.sock
CUDA_VISIBLE_DEVICES=$GPU "$BODEX_PY" -m bodex_service.server > "$LOGS/worker.log" 2>&1 &
SVC=$!
trap 'kill $SVC 2>/dev/null; sleep 2; kill -9 $SVC 2>/dev/null' EXIT
for i in $(seq 1 90); do grep -q "ready on" "$LOGS/worker.log" && break; sleep 2; done
echo "BODex worker ready (pid $SVC)"

# side-grasp objects get a larger seed budget
seeds_for() { case "$1" in mustard|milk|green_can|tea_peach) echo 400;; *) echo 300;; esac; }

for ARM in "${ARMS[@]}"; do
  for OBJ in "${OBJS[@]}"; do
    S=$(seeds_for "$OBJ")
    if [ "$NEP" -gt 0 ] && ! grep -q "full success" "$LOGS/${ARM}_${OBJ}.log" 2>/dev/null; then
      t0=$(date +%s)
      CUDA_VISIBLE_DEVICES=$GPU "$ISAAC_PY" -u isaaclab_tasks/run_bodex_pick_place.py \
        --active_arm "$ARM" --object_id "$OBJ" --approach auto \
        --num_episodes "$NEP" --seed_num "$S" --headless --no_record --device cuda:0 "$KIT" \
        > "$LOGS/${ARM}_${OBJ}.log" 2>&1
      echo "$ARM $OBJ: $(grep -E 'full success' "$LOGS/${ARM}_${OBJ}.log" | tail -1 | sed 's/.*: //') ($(( $(date +%s)-t0 ))s)"
    fi
    if [ "$VIDEO" = 1 ] && [ ! -s "$VIDS/${ARM}_${OBJ}.mp4" ]; then
      CUDA_VISIBLE_DEVICES=$GPU "$ISAAC_PY" -u isaaclab_tasks/run_bodex_pick_place.py \
        --active_arm "$ARM" --object_id "$OBJ" --approach auto \
        --num_episodes 1 --seed_num "$S" --headless --enable_cameras --device cuda:0 "$KIT" \
        --video_path "$VIDS/${ARM}_${OBJ}.mp4" > "$LOGS/${ARM}_${OBJ}_vid.log" 2>&1
      echo "$ARM $OBJ video: $(grep -E 'full success' "$LOGS/${ARM}_${OBJ}_vid.log" | tail -1 | sed 's/.*: //')"
    fi
  done
done
echo "ALL_DONE -> $OUT   (summarise: python scripts/summarize_sim_benchmark.py $OUT)"
