"""BODex grasp → physical dual-arm executor.

Loads a BODex grasp_and_mogen output `.npy`, selects the best candidate, and
commands the specified arm (ra=right UR3+Allegro, la=left UR3e+Allegro) through
the pick sequence:

    1. Go to pregrasp pose (UR arm JointTrajectory)
    2. Close hand smoothly to grasp_qpos   (SetDesPos service)
    3. Close hand further to squeeze_qpos  (SetDesPos service)
    4. Lift 15 cm along world +Z            (UR arm JointTrajectory)
    5. (Optional) Return to home or place

Modeled on `ros2_ws_jazzy/src/my_dual_robot_cell/my_dual_robot_cell_control/
scripts/dexgrasp_pap.py` which is a DexGraspNet-driven reference.

Usage (from a ROS2-overlay Python, e.g. `source install/setup.bash`):
    python grasp_executor.py \\
        --arm ra \\
        --grasp_npy /path/to/grasp_data/peach/tabletop_*/*.npy \\
        --speed 0.4

Requires: geometry_msgs, trajectory_msgs, sensor_msgs, allegro_hand_interfaces.
"""
# Standard Library
import argparse
import math
import sys
import time

# Third Party
import numpy as np

# Order the *combined* URDFs use (6 arm + 16 hand). Matches
# ra_ur3_allegro_isaac_.urdf / la_ur3e_allegro_isaac_.urdf.
ARM_JOINTS_RA = [
    "ra_shoulder_pan_joint", "ra_shoulder_lift_joint", "ra_elbow_joint",
    "ra_wrist_1_joint", "ra_wrist_2_joint", "ra_wrist_3_joint",
]
ARM_JOINTS_LA = [j.replace("ra_", "la_") for j in ARM_JOINTS_RA]

# Allegro joint order as commanded by SetDesPos (16 entries, index→thumb).
# BODex saves 16 hand joints in the URDF order `joint_0.0 … joint_15.0` (floating
# hand) or `ra_/la_joint_0_0 … _15_0` (combined URDF). We map 1-to-1.
HAND_JOINTS_ORDER = list(range(16))


def load_best_grasp(npy_path: str, arm: str):
    """Return (arm_trajectory, pregrasp_hand, grasp_hand, squeeze_hand) as numpy arrays.

    Accepts BOTH floating-hand `*_grasp.npy` output (no arm trajectory, only
    the 3-stage hand pose) AND mogen `*_mogen.npy` output (full arm trajectory
    baked in as robot_pose shape [1, N, T, 22]).
    """
    d = np.load(npy_path, allow_pickle=True).item()
    print(f"Loaded {npy_path}  keys={sorted(d.keys())}")

    rp = d["robot_pose"]
    ge = d["grasp_error"]
    max_err_per_seed = ge.max(axis=-1)[0]

    best_seed = int(np.argmin(max_err_per_seed))
    print(f"  best seed = {best_seed}  error = {max_err_per_seed[best_seed]:.4f}")
    print(f"  robot_pose shape: {rp.shape}  (1, N_seeds, T_stages, DoF)")

    arm_traj = None
    # mogen output: robot_pose is [1, N, T_waypoints, 22] — 6 arm + 16 hand.
    # T typically 33 (30 approach waypoints + grasp + squeeze + lift).
    # grasp-only output: robot_pose is [1, N, 3, 23] — 3 stages (pregrasp/
    # grasp/squeeze) of floating-hand root pose (7) + 16 hand joints.
    if rp.shape[-1] == 22:
        # mogen output: first 6 = arm, last 16 = hand at each waypoint
        arm_traj = rp[0, best_seed, :, :6]          # [T, 6]
        # Hand configs appear near the end of the trajectory: last waypoint
        # holds the "grasp closed" config; waypoints earlier in the trajectory
        # approach with an open hand. The mogen file stacks: [approach_traj
        # (30 wp, hand open), grasp_wp (1 wp, grasp closed), squeeze_wp
        # (1 wp, squeeze), lift_wp (1 wp, squeeze)] — so:
        if rp.shape[2] >= 33:
            pregrasp_hand = rp[0, best_seed, 29, 6:22]  # last approach wp's hand
            grasp_hand    = rp[0, best_seed, 30, 6:22]  # grasp waypoint
            squeeze_hand  = rp[0, best_seed, 31, 6:22]  # squeeze waypoint
        else:
            # fallback: pull from the last few waypoints
            pregrasp_hand = rp[0, best_seed, -3, 6:22]
            grasp_hand    = rp[0, best_seed, -2, 6:22]
            squeeze_hand  = rp[0, best_seed, -1, 6:22]
    elif rp.shape[-1] == 23:
        # floating-hand grasp-only output
        pregrasp_hand = rp[0, best_seed, 0, 7:23]
        grasp_hand    = rp[0, best_seed, 1, 7:23]
        squeeze_hand  = rp[0, best_seed, 2, 7:23]
    else:
        raise ValueError(f"Unexpected robot_pose trailing dim {rp.shape[-1]}; expected 22 or 23.")

    return arm_traj, pregrasp_hand, grasp_hand, squeeze_hand


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["ra", "la"], required=True)
    p.add_argument("--grasp_npy", required=True,
                   help="Path to BODex grasp .npy (output of plan_mogen_batch.py)")
    p.add_argument("--speed", type=float, default=0.4,
                   help="Arm trajectory duration scaling (seconds per 1 rad of largest joint delta)")
    p.add_argument("--hand_close_duration", type=float, default=2.0,
                   help="Time (s) to smoothly close hand from pregrasp → grasp → squeeze")
    p.add_argument("--lift_m", type=float, default=0.15,
                   help="Post-grasp lift distance along world +Z (meters)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without publishing (requires ROS2 for topic types)")
    args = p.parse_args()

    arm_joints = ARM_JOINTS_RA if args.arm == "ra" else ARM_JOINTS_LA
    arm_cmd_topic = f"/{args.arm}_scaled_joint_trajectory_controller/joint_trajectory"
    hand_svc_name = f"/{args.arm}_/setDesiredConfiguration"

    arm_traj, pregrasp, grasp, squeeze = load_best_grasp(args.grasp_npy, args.arm)
    print(f"  pregrasp hand: {pregrasp}")
    print(f"  grasp hand:    {grasp}")
    print(f"  squeeze hand:  {squeeze}")
    if arm_traj is None:
        print("  [warn] arm_trajectory not in .npy — caller must compute IK from current robot state.")
        print("         (Only hand commands will be issued; arm stays in place.)")
    else:
        print(f"  arm trajectory: {arm_traj.shape}  first: {arm_traj[0]}  last: {arm_traj[-1]}")

    if args.dry_run:
        print("[dry_run] exit")
        return 0

    try:
        import rclpy
        from rclpy.node import Node
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from builtin_interfaces.msg import Duration
        from allegro_hand_interfaces.srv import SetDesPos
    except ImportError as e:
        print(f"ROS2 Python not available ({e}). Source your workspace overlay and retry.")
        return 1

    rclpy.init()
    node = rclpy.create_node(f"bodex_grasp_executor_{args.arm}")

    arm_pub = node.create_publisher(JointTrajectory, arm_cmd_topic, 10)
    hand_cli = node.create_client(SetDesPos, hand_svc_name)
    while not hand_cli.wait_for_service(timeout_sec=2.0):
        node.get_logger().warn(f"Still waiting for {hand_svc_name} ...")

    def publish_arm_traj(positions_list, duration_s):
        """positions_list: list of 6-vectors. Generates JointTrajectory with
        evenly-spaced time_from_start."""
        msg = JointTrajectory()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.joint_names = arm_joints
        n = len(positions_list)
        for i, q in enumerate(positions_list):
            pt = JointTrajectoryPoint()
            pt.positions = list(float(x) for x in q)
            tfs = duration_s * (i + 1) / n
            sec = int(tfs)
            nsec = int((tfs - sec) * 1e9)
            pt.time_from_start = Duration(sec=sec, nanosec=nsec)
            msg.points.append(pt)
        arm_pub.publish(msg)
        return duration_s

    def send_hand(pos_16):
        req = SetDesPos.Request()
        req.pos = [float(x) for x in pos_16]
        future = hand_cli.call_async(req)
        # Don't block — fire-and-forget; the commander smooths the transition.

    def interpolate_hand(start, end, duration_s, steps=40):
        """Linearly interpolate hand joints over `steps` points."""
        for k in range(1, steps + 1):
            alpha = k / steps
            q = (np.asarray(start) + alpha * (np.asarray(end) - np.asarray(start))).tolist()
            send_hand(q)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(duration_s / steps)

    # ========== sequence ==========

    # Stage 0: open hand to pregrasp BEFORE moving arm (avoid hitting anything)
    node.get_logger().info("stage 0: preshape hand")
    send_hand(pregrasp.tolist())
    time.sleep(0.5)

    # Stage 1: move arm to pregrasp pose (last waypoint of arm_traj)
    if arm_traj is not None:
        # use the whole trajectory for a smooth motion
        # estimate duration from max joint delta (rough heuristic)
        start_q = arm_traj[0]
        end_q = arm_traj[-1]
        max_delta = float(np.max(np.abs(np.asarray(end_q) - np.asarray(start_q))))
        dur = max(2.0, max_delta / max(args.speed, 0.05))
        node.get_logger().info(f"stage 1: arm → pregrasp ({dur:.1f}s, {len(arm_traj)} waypoints)")
        publish_arm_traj([q.tolist() for q in arm_traj], dur)
        time.sleep(dur + 0.5)

    # Stage 2: smoothly close hand pregrasp → grasp
    node.get_logger().info("stage 2: close to grasp")
    interpolate_hand(pregrasp, grasp, args.hand_close_duration * 0.6, steps=30)

    # Stage 3: final squeeze
    node.get_logger().info("stage 3: squeeze")
    interpolate_hand(grasp, squeeze, args.hand_close_duration * 0.4, steps=20)

    # Stage 4: latch hand for 2 s (Allegro driver treats no-command as release)
    t_end = time.time() + 2.0
    while time.time() < t_end:
        send_hand(squeeze.tolist())
        rclpy.spin_once(node, timeout_sec=0.05)

    # Stage 5: lift. Requires current joint state + cuRobo IK. For this
    # reference implementation, we simply re-send the last arm waypoint with
    # joint 1 (shoulder_lift) raised by 0.1 rad — a VERY crude vertical lift.
    # A production version should do Cartesian IK; see the TODO below.
    if arm_traj is not None:
        lift_q = list(arm_traj[-1])
        lift_q[1] += 0.1   # TODO: replace with cuRobo Cartesian +Z IK
        node.get_logger().info(f"stage 5: lift (crude +0.1 rad on shoulder_lift)")
        publish_arm_traj([lift_q], 2.0)
        time.sleep(2.5)

    node.get_logger().info("done.")
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
