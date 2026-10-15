#!/bin/bash
# Record 3 episodes per object into mp4s (ra arm). Videos -> recordings/episodes/.
cd /mnt/ssd2/ws/Dualarm_ws/BODex
set -u
PY=/home/robot/anaconda3/envs/env_isaaclab/bin/python
LOGS=/tmp/record_eps
VID=recordings/episodes
mkdir -p "$LOGS" "$VID"
export LD_LIBRARY_PATH="/home/robot/anaconda3/envs/bodex/lib:${LD_LIBRARY_PATH:-}"
OBJECTS=(peach red_cube tomato mouse tumbler airplane bottle flat_plate \
         plate core_can core_mug red_cube_v2 green_cube pikachu mustard whale tiger)
ARM=ra
NEP=3
for OBJ in "${OBJECTS[@]}"; do
    echo "=========== recording $ARM/$OBJ ($NEP ep) ==========="
    pkill -9 -f "bodex_service.server" 2>/dev/null || true
    sleep 2; rm -f /tmp/bodex_service.sock
    "$PY" -u isaaclab_tasks/run_bodex_pick_place.py \
        --active_arm "$ARM" --object_id "$OBJ" --approach auto \
        --num_episodes "$NEP" --headless --enable_cameras --fast_regrasp \
        --video_path "$VID/${OBJ}_${ARM}.mp4" \
        > "$LOGS/${OBJ}.log" 2>&1
    pkill -9 -f "bodex_service.server" 2>/dev/null || true
    F=$(grep -E "full success" "$LOGS/${OBJ}.log" | tail -1 | sed 's/.*: //')
    OK=$([ -f "$VID/${OBJ}_${ARM}.mp4" ] && echo "video✓" || echo "NOVID")
    echo ">>> $OBJ : full=$F  $OK"
done
echo ""
echo "=========== RECORDING COMPLETE — $VID ==========="
ls -la "$VID"/*.mp4 2>/dev/null
