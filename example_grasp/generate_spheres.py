"""Generate a BODex-style collision-spheres YAML from a URDF's collision meshes.

Reads the URDF, walks each collision_link, fits N spheres to its collision mesh
(from curobo.geom.sphere_fit), and emits a YAML keyed by link name.

Usage:
  python example_grasp/generate_spheres.py \
      --robot_cfg ra_ur3_allegro.yml \
      --out_spheres spheres/ra_ur3_allegro.yml \
      --n_spheres_arm 10 --n_spheres_hand 4
"""
# Standard Library
import argparse
import os
from pathlib import Path

# Third Party
import numpy as np
import trimesh
import yaml
import yourdfpy

# CuRobo
from curobo.geom.sphere_fit import SphereFitType, fit_spheres_to_mesh
from curobo.util_file import get_assets_path, get_robot_configs_path, join_path, load_yaml


def mesh_for_link(urdf: yourdfpy.URDF, link_name: str, urdf_dir: str):
    """Return a trimesh of all collision geometry for a link, or None if no collisions.

    Supports mesh, box, cylinder, and sphere primitives.
    """
    link = urdf.link_map.get(link_name)
    if link is None or not link.collisions:
        return None
    meshes = []
    for coll in link.collisions:
        geom = coll.geometry
        tm = None
        if geom.mesh is not None:
            mesh_filename = geom.mesh.filename
            mesh_path = os.path.normpath(os.path.join(urdf_dir, mesh_filename))
            if not os.path.exists(mesh_path):
                print(f"  [warn] mesh missing: {mesh_path}")
                continue
            tm = trimesh.load(mesh_path, force="mesh")
            if geom.mesh.scale is not None:
                tm.apply_scale(geom.mesh.scale)
        elif geom.box is not None:
            tm = trimesh.creation.box(extents=np.asarray(geom.box.size, dtype=float))
        elif geom.cylinder is not None:
            tm = trimesh.creation.cylinder(radius=float(geom.cylinder.radius),
                                           height=float(geom.cylinder.length))
        elif geom.sphere is not None:
            tm = trimesh.creation.icosphere(radius=float(geom.sphere.radius))
        if tm is None:
            continue
        if coll.origin is not None:
            tm.apply_transform(coll.origin)
        meshes.append(tm)
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot_cfg", required=True, help="Robot config YAML name (e.g. ra_ur3_allegro.yml)")
    p.add_argument("--out_spheres", required=True, help="Output spheres YAML name (e.g. spheres/ra_ur3_allegro.yml)")
    p.add_argument("--n_spheres_arm", type=int, default=10, help="Per-arm-link sphere count")
    p.add_argument("--n_spheres_hand", type=int, default=4, help="Per-hand-link sphere count")
    args = p.parse_args()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot_cfg))
    k = robot_cfg["robot_cfg"]["kinematics"]
    urdf_rel = k["urdf_path"]
    urdf_abs = join_path(get_assets_path(), urdf_rel)
    print(f"URDF: {urdf_abs}")
    urdf = yourdfpy.URDF.load(urdf_abs, load_meshes=False, load_collision_meshes=False,
                              build_scene_graph=False)

    collision_links = k["collision_link_names"]
    out = {"robot": os.path.splitext(os.path.basename(args.robot_cfg))[0],
           "collision_spheres": {}}

    for link_name in collision_links:
        # Identify whether this is an arm link (UR3 body) or hand link (Allegro)
        is_arm = any(s in link_name for s in ("shoulder_link", "upper_arm_link", "forearm_link",
                                              "wrist_1_link", "wrist_2_link", "wrist_3_link"))
        n_sph = args.n_spheres_arm if is_arm else args.n_spheres_hand

        mesh = mesh_for_link(urdf, link_name, os.path.dirname(urdf_abs))
        if mesh is None:
            print(f"  [skip] {link_name}: no collision mesh")
            continue

        centers, radii = fit_spheres_to_mesh(
            mesh, n_sph,
            surface_sphere_radius=0.01,
            fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
        )
        out["collision_spheres"][link_name] = [
            {"center": [float(c) for c in centers[i]], "radius": float(radii[i])}
            for i in range(len(centers))
        ]
        print(f"  [ok]  {link_name}: {len(centers)} spheres (r_avg={np.mean(radii):.4f})")

    out_path = join_path(get_robot_configs_path(), args.out_spheres)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(out, f, sort_keys=False, default_flow_style=None)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
