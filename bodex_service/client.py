"""BODex grasp client — importable from any Python environment.

Requires only the Python stdlib + numpy in the caller's env. It talks to a
long-running worker (server.py) that lives inside the bodex conda env, so
the caller never needs torch/cuRobo/etc. installed.

Typical usage:

    from bodex_service import BODexClient
    import numpy as np

    # Context-managed: starts the worker if not running, stops it at exit
    with BODexClient() as client:
        result = client.synthesize_pick_and_place(
            active_arm="ra",
            object_id="red_cube",
            object_pose=np.array([0.05, 0.47, 1.00, 1, 0, 0, 0]),
            place_xyz=np.array([0.12, 0.50, 1.01]),
            approach="top",
        )
        trajectory = result["trajectory"]      # [T, 22] float32
        pick_end  = result["pick_end_idx"]     # int

Or as a long-lived singleton:

    client = BODexClient(auto_start=True)
    ...many calls...
    client.close()
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from multiprocessing.connection import Client
from typing import Any

import numpy as np


# Resolve the repo root so we can find server.py, bodex python, and
# isaaclab_tasks in a stable way regardless of how this module is imported.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BODEX_REPO = os.path.dirname(_HERE)                           # /.../BODex
_ISAACLAB_TASKS = os.path.join(_BODEX_REPO, "isaaclab_tasks")
_SERVER_SCRIPT = os.path.join(_HERE, "server.py")

# The bodex conda env's python. Can be overridden via env var.
DEFAULT_BODEX_PYTHON = os.environ.get(
    "BODEX_SERVICE_PYTHON", "/home/robot/anaconda3/envs/bodex/bin/python"
)

# Default socket + auth
DEFAULT_SOCKET = "/tmp/bodex_service.sock"
DEFAULT_AUTHKEY = "bodex-service"


class BODexClient:
    """Python client for the BODex grasp synthesis service.

    Starts a persistent worker (in the bodex conda env) on first use if one
    isn't already running; subsequent grasp requests are dispatched over a
    Unix domain socket (multiprocessing.connection).

    Parameters
    ----------
    socket_path : str
        Unix socket path. If the worker already has this socket, we attach.
    bodex_python : str
        Path to the python interpreter inside the bodex conda env.
    auto_start : bool
        If True and no existing worker is responding, spawn one.
    startup_timeout : float
        Max seconds to wait for the worker to become ready.
    verbose : bool
        If True, print worker stdout.
    """

    def __init__(
        self,
        socket_path: str = DEFAULT_SOCKET,
        bodex_python: str = DEFAULT_BODEX_PYTHON,
        authkey: str = DEFAULT_AUTHKEY,
        auto_start: bool = True,
        startup_timeout: float = 60.0,
        verbose: bool = False,
    ):
        self.socket_path = socket_path
        self.bodex_python = bodex_python
        self.authkey = authkey.encode()
        self.verbose = verbose
        self._proc: subprocess.Popen | None = None
        self._owns_proc = False

        if not self._is_alive():
            if auto_start:
                self._start_server(startup_timeout)
            else:
                raise RuntimeError(
                    f"No BODex worker at {socket_path}. "
                    "Start one manually or pass auto_start=True."
                )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _is_alive(self) -> bool:
        if not os.path.exists(self.socket_path):
            return False
        try:
            resp = self._send({"op": "ping"}, timeout=2.0)
            return bool(resp.get("ok"))
        except Exception:
            return False

    def _start_server(self, startup_timeout: float):
        if not os.path.exists(self._server_script()):
            raise RuntimeError(f"server.py not found at {self._server_script()}")
        if not os.path.exists(self.bodex_python):
            raise RuntimeError(f"bodex python not found at {self.bodex_python}")

        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env["LD_LIBRARY_PATH"] = (
            os.path.dirname(os.path.dirname(self.bodex_python)) + "/lib:" +
            env.get("LD_LIBRARY_PATH", "")
        )
        env["BODEX_ISAACLAB_TASKS_PATH"] = _ISAACLAB_TASKS
        # Make bodex_service importable from the worker's sys.path
        env["PYTHONPATH"] = _BODEX_REPO + ":" + env.get("PYTHONPATH", "")

        cmd = [
            self.bodex_python, "-u",
            self._server_script(),
            "--socket", self.socket_path,
            "--authkey", self.authkey.decode(),
        ]
        if self.verbose:
            cmd.append("--verbose")

        stdout = sys.stdout if self.verbose else subprocess.DEVNULL
        stderr = sys.stderr if self.verbose else subprocess.DEVNULL
        self._proc = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr)
        self._owns_proc = True

        # Wait for the socket to appear AND accept ping
        t0 = time.time()
        while time.time() - t0 < startup_timeout:
            if os.path.exists(self.socket_path) and self._ping_once():
                return
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"BODex worker exited with code {self._proc.returncode} "
                    f"before becoming ready."
                )
            time.sleep(0.5)
        raise TimeoutError(
            f"BODex worker did not become ready within {startup_timeout}s"
        )

    def _server_script(self) -> str:
        return _SERVER_SCRIPT

    def _ping_once(self) -> bool:
        try:
            with Client(self.socket_path, family="AF_UNIX", authkey=self.authkey) as c:
                c.send({"op": "ping"})
                resp = c.recv()
                return bool(resp.get("ok"))
        except Exception:
            return False

    def close(self, shutdown_server: bool = True):
        """Close the client. If we started the server, stop it too."""
        if self._proc is not None and self._owns_proc:
            if shutdown_server:
                try:
                    self._send({"op": "shutdown"}, timeout=5.0)
                except Exception:
                    pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

    # Context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ------------------------------------------------------------------
    # Low-level send
    # ------------------------------------------------------------------
    def _send(self, req: dict, timeout: float = 120.0) -> dict:
        """One-shot request/response over a fresh connection."""
        with Client(self.socket_path, family="AF_UNIX", authkey=self.authkey) as conn:
            conn.send(req)
            if conn.poll(timeout):
                return conn.recv()
            raise TimeoutError(f"BODex worker did not respond within {timeout}s")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ping(self) -> bool:
        try:
            resp = self._send({"op": "ping"}, timeout=3.0)
            return bool(resp.get("ok"))
        except Exception:
            return False

    def synthesize_pick_and_place(
        self,
        active_arm: str,
        object_id: str,
        object_pose: np.ndarray,
        place_xyz: np.ndarray,
        approach: str = "top",
        seed_num: int = 20,
        hold_frames: int = 10,
        transit_n: int = 40,
        descend_n: int = 20,
        release_n: int = 12,
        table_z: float = 0.75,
        passive_arm: str | None = None,
        passive_joint_state: np.ndarray | None = None,
        thumb_up_min: float = 0.65,
        thumb_up_strict: bool = False,
        standoff_dist: float = 0.10,
        grasp_extra_in: float = 0.0,
        grasp_z_bonus: float = 0.0,
        top_front_min: float = 0.0,
        top_level_relax: bool = True,
        grasp_body_bias: float = 0.0,
        grasp_axis_align: float = 0.0,
        grasp_center_bias: float = 0.0,
        grasp_mode: str = "wrap",
        synth_seed: int = 123,
        obj_scale=(1.0, 1.0, 1.0),
        obb_z_world: float | None = None,
        release_lift_z: float = 0.08,
        post_grasp_lift_z: float = 0.10,
        transit_hover_z: float = 0.15,
        side_face_z_min: float = -1.0,
        top_level_min: float = 0.0,
    ) -> dict[str, Any]:
        """Synthesize a full pick+place trajectory.

        Parameters
        ----------
        active_arm : {'ra', 'la'}
        object_id : str
            Must match a folder under
            BODex/src/curobo/content/assets/object/custom/processed_data/.
        object_pose : array-like [7]
            Object's world pose: [x, y, z, qw, qx, qy, qz].
        place_xyz : array-like [3]
            World xyz where the object should be placed.
        approach : {'top', 'side', 'side_front', 'side_left', 'side_right'}
            Palm approach direction for the grasp. 'side' auto-picks the
            correct per-arm side (ra → +X, la → -X) with thumb-up to avoid
            scraping the table with the fingers.
        seed_num : int
            Number of BODex grasp seeds to evaluate (higher = more robust
            but slower; 20 is the typical good tradeoff).
        hold_frames : int
            How many copies of grasp/squeeze/lift waypoints to emit (PID
            settling time per stage).
        transit_n, descend_n, release_n : int
            Waypoint counts for the three place phases.
        table_z : float
            World z of the table top (used by BODex as a plane obstacle).

        Returns
        -------
        dict with keys:
            trajectory:        np.ndarray [T, 22] float32 — 6 arm + 16 hand joints
            pick_end_idx:      int — last pick waypoint index; rest is place
            grasp_error:       float — BODex L-BFGS grasp error for the best seed
                               (lower is better; ~<0.05 is a solid grasp,
                               >0.1 is marginal)
            feasible_count:    int — seeds with all 3 IK stages feasible
            feasible_total:    int — seed_num (so the ratio is the signal)
            object_pose_world: np.ndarray [7]
            place_xyz:         np.ndarray [3]
            active_arm, object_id, approach : input echoes
            success:           bool — always True on return (raises on failure)

        Raises
        ------
        RuntimeError : if the worker reports an error (e.g. infeasible IK).
        TimeoutError : if the worker does not respond within the deadline.
        """
        req = {
            "op": "synthesize",
            "active_arm": active_arm,
            "object_id": object_id,
            "object_pose": np.asarray(object_pose, dtype=float).tolist(),
            "place_xyz": np.asarray(place_xyz, dtype=float).tolist(),
            "approach": approach,
            "seed_num": int(seed_num),
            "hold_frames": int(hold_frames),
            "transit_n": int(transit_n),
            "descend_n": int(descend_n),
            "release_n": int(release_n),
            "table_z": float(table_z),
            "thumb_up_min": float(thumb_up_min),
            "thumb_up_strict": bool(thumb_up_strict),
            "standoff_dist": float(standoff_dist),
            "grasp_extra_in": float(grasp_extra_in),
            "grasp_z_bonus": float(grasp_z_bonus),
            "top_front_min": float(top_front_min),
            "top_level_relax": bool(top_level_relax),
            "grasp_body_bias": float(grasp_body_bias),
            "grasp_axis_align": float(grasp_axis_align),
            "grasp_center_bias": float(grasp_center_bias),
            "grasp_mode": str(grasp_mode),
            "synth_seed": int(synth_seed),
            "obj_scale": [float(s) for s in obj_scale],
            "post_grasp_lift_z": float(post_grasp_lift_z),
            "transit_hover_z": float(transit_hover_z),
            "side_face_z_min": float(side_face_z_min),
            "top_level_min": float(top_level_min),
            "release_lift_z": float(release_lift_z),
        }
        if obb_z_world is not None:
            req["obb_z_world"] = float(obb_z_world)
        # Optional Level-2 passive-arm collision: if provided, the planner
        # injects the passive arm's collision spheres as obstacles.
        if passive_arm is not None and passive_joint_state is not None:
            req["passive_arm"] = passive_arm
            req["passive_joint_state"] = np.asarray(
                passive_joint_state, dtype=float
            ).tolist()
        resp = self._send(req, timeout=120.0)
        if not resp.get("ok"):
            raise RuntimeError(f"BODex synthesis failed: {resp.get('error')}")
        return resp["result"]
