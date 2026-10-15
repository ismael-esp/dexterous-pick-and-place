#!/usr/bin/env python3
"""Reviewer DEMO driver for the BODex pick-and-place.

Same robot pipeline as ``benchmark_pick_place.py`` (identical synthesis, safety
gates, replay, recording and CSV format) — but structured for a live demo in
front of a reviewer:

    1. Pick an object from a numbered menu.
    2. Choose N — how many pick-and-places to run on it (you type it; the
       --iters value is just the default you can accept with Enter).
    3. Run N iterations.  BEFORE each one the robot waits at home for a keypress
       (default 'a') so you can PLACE/RE-POSITION the object in the workspace.
       Press the key -> one real pick-and-place runs.  While it moves, press
       's' (success) or 'f' (fail) — either stops the motion, homes the arm and
       opens the hand, and records the row.
    4. After N iterations you're back at the menu: pick the next object and
       repeat, or 'q' to finish.

Every iteration is appended to the SAME CSV schema as the benchmark (and can be
video-recorded with --record), so the existing hw_10_object_test report and all
analysis work on the output unchanged.

    # real demo, right arm, record every iteration, 10 runs/object by default:
    python bodex_hw/demo_pick_place.py --active_arm ra --table_z 0.91 --record \
        --csv ~/Dualarm_ws/BODex/benchmark_results/demo_review.csv --iters 10

    # pre-pick the first object and its count, then drop into the menu:
    python bodex_hw/demo_pick_place.py --object mustard --iters 20 ...

    # dry test, NO robot / NO ROS (exercise the menu + gating + CSV):
    python bodex_hw/demo_pick_place.py --mock --csv /tmp/demo_test.csv --iters 3

Keys:  digits+Enter select object · '<gate>' (default 'a') run an iteration ·
       's' success · 'f' fail · 'q' stop the current object / quit the menu.

SAFETY: same as the benchmark — the post-abort/return-home is a straight joint
lerp from wherever the motion froze (NOT collision-checked); it opens the hand
first, lifts for table clearance and moves slowly. Keep a hand on the e-stop and
press s/f only when it's safe to drop the object.
"""
from __future__ import annotations

import argparse
import os
import sys

# Run as a bare script from any directory: put the repo root on sys.path so the
# `bodex_hw.*` package imports resolve (same shim as benchmark_pick_place.py).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import bodex_hw.benchmark_pick_place as B
from bodex_hw.benchmark_pick_place import (
    BENCH_OBJECTS, build_args, run_one_trial, write_row, open_csv, load_counts,
    choose_results_file, read_one_key, show_overrides, apply_set_command,
    seed_overrides_from_cli, print_menu, mock_single_trial, MockIO,
)
from bodex_hw.object_meta import resolve_object_entry
from bodex_hw.run_bodex_pick_place_hw import _home_rad, _traj_cfg, run_single_trial


# --------------------------------------------------------------------------
# CLI: the full benchmark flag set + three demo-specific flags.
# --------------------------------------------------------------------------
def parse_demo_cli(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    d = argparse.ArgumentParser(add_help=False)
    d.add_argument("--object", default=None,
                   help="pre-select the FIRST object by name; the menu still "
                        "runs after so you can pick more.")
    d.add_argument("--iters", "--n", dest="iters", type=int, default=10,
                   help="DEFAULT iterations per object (you're still prompted "
                        "each object; Enter accepts this). Default 10.")
    d.add_argument("--gate_key", default="a",
                   help="key to press to RUN each iteration after you've placed "
                        "the object (default 'a').")
    demo_ns, rest = d.parse_known_args(argv)
    cli = B.parse_cli(rest)          # all the shared benchmark flags + --help
    cli.demo_object = demo_ns.object
    cli.demo_iters = max(1, demo_ns.iters)
    cli.demo_gate_key = (demo_ns.gate_key or "a").strip().lower()[:1] or "a"
    return cli


def _ask_int(prompt, default):
    try:
        s = input(f"{prompt} [{default}] > ").strip()
    except EOFError:
        return default
    if not s:
        return default
    try:
        v = int(s)
        return v if v > 0 else default
    except ValueError:
        print(f"[demo] not a number; using {default}.")
        return default


def _select_object(cli, counts, disabled, overrides, pending):
    """Return an object name to run, or None to quit. Consumes a pending
    --object on the first call; otherwise drives the numbered menu (with the
    same `set`/`show` live-tuning commands as the benchmark)."""
    if pending[0] is not None:
        obj, pending[0] = pending[0], None
        if obj in BENCH_OBJECTS and obj not in disabled:
            print(f"[demo] starting with --object {obj}")
            return obj
        print(f"[demo] --object '{obj}' unavailable; using the menu instead.")
    while True:
        print_menu(cli.active_arm, counts, disabled)
        print("    (number = pick object · 'set <obj> <key> <val>' tune · "
              "'show' overrides · q = finish)")
        try:
            sel = input("select object> ").strip().lower()
        except EOFError:
            return None
        if sel in ("q", "quit", "exit"):
            return None
        if sel in ("show", "?"):
            show_overrides(overrides)
            continue
        if sel.startswith("set"):
            print(apply_set_command(sel.split(), overrides))
            continue
        if sel.isdigit() and 1 <= int(sel) <= len(BENCH_OBJECTS):
            obj = BENCH_OBJECTS[int(sel) - 1]
            if obj in disabled:
                print(f"[demo] {obj} is unavailable ({disabled[obj]}).")
                continue
            return obj
        print("[demo] enter a listed number, 'set ...', 'show', or q.")


def _run_object_session(cli, io, obj, n, home_arm, counts, overrides, fh, writer):
    """Run up to N keypress-gated pick-and-places on one object. Executed
    (success/fail) iterations count toward N; a no-grasp lets you reposition and
    retry without burning an iteration. 'q' at the gate stops this object."""
    entry = resolve_object_entry(obj, cli.active_arm)
    gate = cli.demo_gate_key
    done = 0
    print(f"\n=== {obj}: demonstrating {n} pick-and-place(s) "
          f"(gate key '{gate}', s/f to judge, q to stop) ===")
    while done < n:
        c = counts[obj]
        trial_idx = c["att"] + c["nofail"] + 1
        k = read_one_key(
            f"\n[{obj}] run {done + 1}/{n} — POSITION the object in the "
            f"workspace, then press '{gate}' to RUN  (q = stop {obj})")
        if k in ("q", "quit"):
            print(f"[demo] stopping {obj} after {done}/{n}.")
            break
        if k != gate:
            print(f"[demo] (press '{gate}' to run, or 'q' to stop {obj})")
            continue

        # Build per-iteration args + recording dir (same scheme as the benchmark
        # so videos land at <record_dir>/<obj>/trialNNN and the CSV matches).
        args = build_args(obj, cli.active_arm, cli, overrides)
        video = ""
        if args.record and not cli.mock:
            video = os.path.join(cli.record_dir, obj, f"trial{trial_idx:03d}")
            os.makedirs(video, exist_ok=True)
            args.record_dir = video
        if not cli.mock:
            io.set_object_topic(args.object_pose_topic)

        print(f"[demo] running {obj} run {done + 1}/{n} "
              f"(trial {trial_idx}; s=success, f=fail)...")
        if cli.mock:
            worker_factory = (lambda se, o=obj, a=cli.active_arm:
                              mock_single_trial(se, o, a))
        else:
            worker_factory = (lambda se, a=args, e=entry:
                              run_single_trial(io, a, e, home_arm,
                                               _traj_cfg(a), stop_evt=se))

        result, key, dur, aborted, no_motion = run_one_trial(
            worker_factory, io, home_arm, cli.mock)

        if no_motion:
            outcome = result.get("outcome", "synth_fail")
            c["nofail"] += 1
            print(f"[demo] NO GRASP ({outcome}): {result.get('note', '')} — "
                  f"re-position and it'll retry (this run is NOT counted).")
        else:
            outcome = "success" if key == "s" else "fail"
            c["att"] += 1
            if key == "s":
                c["succ"] += 1
            done += 1
            print(f"[demo] {obj} run {done}/{n}: {outcome.upper()}  "
                  f"(object total {c['succ']}/{c['att']})")

        write_row(fh, writer, args=args, obj=obj, trial_idx=trial_idx,
                  outcome=outcome, result=result, key=key, dur=dur,
                  aborted=aborted, video=video)
    if done >= n:
        print(f"[demo] finished {obj}: {n} run(s) done "
              f"({counts[obj]['succ']}/{counts[obj]['att']} success).")


def main(argv=None):
    cli = parse_demo_cli(argv)
    active_arm = cli.active_arm
    home_arm = _home_rad(active_arm)

    # Validate object geometry up front (a missing asset never crashes mid-demo).
    disabled = {}
    for obj in BENCH_OBJECTS:
        try:
            resolve_object_entry(obj, active_arm)
        except Exception as e:
            disabled[obj] = str(e).splitlines()[0][:40]

    write_path, load_paths = choose_results_file(cli)
    counts = load_counts(load_paths if load_paths else write_path)
    fh, writer = open_csv(write_path)
    print(f"[demo] logging to {write_path}")
    if cli.record and not cli.mock:
        print(f"[demo] recording video under {cli.record_dir}")
    if disabled:
        print(f"[demo] unavailable (no geometry): {', '.join(disabled)}")

    overrides = seed_overrides_from_cli(cli)
    if any(overrides.values()):
        show_overrides(overrides)

    # ROS / IO setup (skipped entirely in --mock).
    io = None
    if cli.mock:
        io = MockIO(home_arm)
    else:
        import rclpy
        from bodex_hw.ros_io import HwIO
        rclpy.init()
        io = HwIO(active_arm=active_arm,
                  object_pose_topic=f"/{BENCH_OBJECTS[0]}_pose",
                  arm_iface="forward_position")
        io.start()

    pending = [cli.demo_object]
    try:
        while True:
            obj = _select_object(cli, counts, disabled, overrides, pending)
            if obj is None:
                break
            n = _ask_int(f"[{obj}] how many pick-and-place runs?", cli.demo_iters)
            _run_object_session(cli, io, obj, n, home_arm, counts, overrides,
                                fh, writer)
    except KeyboardInterrupt:
        print("\n[demo] interrupted — results saved.")
    finally:
        try:
            fh.close()
        except Exception:
            pass
        try:
            if cli.mock:
                io.shutdown()
            else:
                io.shutdown()
                import rclpy
                rclpy.shutdown()
        except Exception:
            pass
        # Final per-object summary for the reviewer.
        print("\n[demo] session summary:")
        any_done = False
        for o in BENCH_OBJECTS:
            c = counts[o]
            if c["att"] or c["nofail"]:
                any_done = True
                print(f"   {o:12s} {c['succ']}/{c['att']} success"
                      + (f"  ({c['nofail']} no-grasp)" if c['nofail'] else ""))
        if not any_done:
            print("   (no runs recorded)")
        print(f"[demo] CSV: {write_path}")


if __name__ == "__main__":
    main()
