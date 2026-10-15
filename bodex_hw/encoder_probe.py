#!/usr/bin/env python3
"""Hands-free Allegro encoder probe.

Run it, then physically move the suspect joint(s) through their range for the
whole duration — no need to watch the screen:
  - it BEEPS (terminal bell) every time a watched joint's value actually moves,
    so silence = dead encoder, beeping = alive;
  - at the end (or on Ctrl-C) it prints min/max/range per watched joint and a
    DEAD / ALIVE verdict.

Usage (in a ROS-sourced terminal):
    python3 bodex_hw/encoder_probe.py                      # right hand, joints 7 & 14, 30 s
    python3 bodex_hw/encoder_probe.py --topic /la_/joint_states --joints 7 14 --duration 60
"""
import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

MOVE_THRESH = 0.01   # rad — beep when a watched joint moves more than this


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/ra_/joint_states")
    ap.add_argument("--joints", type=int, nargs="+", default=[7, 14],
                    help="joint indices to watch (default: 7=middle tip, 14=thumb)")
    ap.add_argument("--duration", type=float, default=30.0,
                    help="seconds to record (Ctrl-C also ends and prints the verdict)")
    args = ap.parse_args()

    rclpy.init()
    node = Node("encoder_probe")
    stats = {j: {"min": None, "max": None, "last": None, "n": 0} for j in args.joints}
    count = [0]

    def cb(msg):
        count[0] += 1
        for j in args.joints:
            if j >= len(msg.position):
                continue
            v = float(msg.position[j])
            s = stats[j]
            s["n"] += 1
            if s["min"] is None or v < s["min"]:
                s["min"] = v
            if s["max"] is None or v > s["max"]:
                s["max"] = v
            if s["last"] is not None and abs(v - s["last"]) > MOVE_THRESH:
                sys.stdout.write("\a")          # audible: the joint moved!
                sys.stdout.flush()
            s["last"] = v

    node.create_subscription(JointState, args.topic, cb, 50)
    print(f"[probe] recording {args.topic} joints {args.joints} for "
          f"{args.duration:.0f}s — MOVE THE FINGERTIP NOW (beep = joint alive)")
    t_end = time.time() + args.duration
    try:
        while time.time() < t_end:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass

    print(f"\n[probe] {count[0]} messages received")
    if count[0] == 0:
        print("[probe] NO DATA — is the driver running and the terminal sourced?")
    for j in args.joints:
        s = stats[j]
        if s["n"] == 0:
            print(f"  joint {j:2d}: no samples")
            continue
        rng = s["max"] - s["min"]
        verdict = "ALIVE (tracks motion)" if rng > MOVE_THRESH else \
                  "DEAD (never moved — frozen reading)"
        print(f"  joint {j:2d}: min={s['min']:+.4f} max={s['max']:+.4f} "
              f"range={rng:.4f} rad -> {verdict}")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
