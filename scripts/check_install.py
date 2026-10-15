#!/usr/bin/env python3
"""Sanity-check the install. Run from the repo root inside the bodex env:

    python scripts/check_install.py            # imports, CUDA kernels, assets
    python scripts/check_install.py --synth    # + one real grasp+trajectory synthesis via the worker
"""
import argparse
import importlib
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
CUSTOM = os.path.join(ROOT, "src", "curobo", "content", "assets", "object", "custom", "processed_data")
OBJECTS = ["green_cube", "red_cube", "pikachu", "whale", "snorlax", "tiger",
           "milk", "mustard", "green_can", "tea_peach"]
ok = True


def check(name, fn):
    global ok
    try:
        info = fn()
        print(f"  [ OK ] {name}{(' - ' + info) if info else ''}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  [FAIL] {name} - {type(e).__name__}: {e}")


def _torch():
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    return f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}"


def _numpy():
    import numpy
    assert numpy.__version__.startswith("1.26"), f"numpy {numpy.__version__} (need 1.26.x)"
    return numpy.__version__


def _curobo():
    import curobo
    src = os.path.dirname(os.path.dirname(curobo.__file__))
    assert os.path.samefile(src, os.path.join(ROOT, "src")), (
        f"curobo is installed from {src}, not from this checkout - re-run `pip install -e .` here")
    return src


def _kernels():
    for m in ("curobo.curobolib.lbfgs_step_cu", "curobo.curobolib.kinematics_fused_cu",
              "curobo.curobolib.geom_cu"):
        importlib.import_module(m)
    return "lbfgs_step / kinematics_fused / geom"


def _objects():
    missing = [o for o in OBJECTS if not os.path.isdir(os.path.join(CUSTOM, o, "mesh"))]
    assert not missing, f"missing processed objects: {missing}"
    return f"{len(OBJECTS)} benchmark objects"


def _sim_assets():
    a = os.path.join(ROOT, "isaaclab_tasks", "assets")
    for f in ("tshape_.usd", "ra_ur3_allegro_isaac_", "la_ur3e_allegro_isaac_", "Materials"):
        assert os.path.exists(os.path.join(a, f)), f"isaaclab_tasks/assets/{f} missing (needed for sim only)"
    return "isaaclab_tasks/assets"


def _synth():
    import numpy as np
    from bodex_service import BODexClient
    sock = "/tmp/bodex_check_install.sock"
    with BODexClient(socket_path=sock, bodex_python=sys.executable) as c:
        t0 = time.time()
        r = c.synthesize_pick_and_place(
            active_arm="ra", object_id=a.object,
            object_pose=np.array([0.25, 0.47, 0.955, 1, 0, 0, 0]),
            place_xyz=np.array([-0.12, 0.48, 1.05]), approach="top",
            seed_num=60, table_z=0.92)
    return (f"trajectory {tuple(r['trajectory'].shape)} in {time.time() - t0:.1f}s, "
            f"grasp_error={r['grasp_error']:.3f}")


p = argparse.ArgumentParser()
p.add_argument("--synth", action="store_true", help="also run one synthesis through the worker")
p.add_argument("--object", default="red_cube", help="object for --synth")
a = p.parse_args()

print("dexterous-pick-and-place - install check")
check("torch + CUDA", _torch)
check("numpy 1.26", _numpy)
check("curobo editable install points here", _curobo)
check("cuRobo CUDA kernels", _kernels)
check("coal_openmp_wrapper", lambda: importlib.import_module("coal_openmp_wrapper") and None)
check("torch_scatter", lambda: importlib.import_module("torch_scatter").__version__)
check("processed object assets", _objects)
check("Isaac Lab scene assets", _sim_assets)
if a.synth:
    check("grasp + trajectory synthesis (worker)", _synth)
print("\nAll good." if ok else "\nSome checks failed - see README > Troubleshooting.")
sys.exit(0 if ok else 1)
