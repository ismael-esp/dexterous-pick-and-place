"""Preprocess custom .obj files into BODex's expected object-asset layout.

Replaces MeshProcess for the BODex grasp-synthesis use case on Ubuntu 24.04,
where MeshProcess's VTK/ACVD/custom-CoACD source builds are painful. Uses only
pip-installable `trimesh` + `coacd`.

Output layout mirrors DGN_2k exactly:

    <out_root>/<obj_id>/
      mesh/simplified.obj      (original mesh, format-normalized)
      urdf/coacd.urdf          (convex pieces as collision geometry)
      urdf/meshes/convex_piece_000.obj, ...
      info/simplified.json     ({gravity_center, obb, scale, density, mass})

Also emits scene_cfg .npy for each object at:

    <scene_cfg_root>/<obj_id>/floating/scale100.npy
    <scene_cfg_root>/<obj_id>/tabletop_ra_ur3_allegro/scale100_pose000_0.npy
    <scene_cfg_root>/<obj_id>/tabletop_la_ur3e_allegro/scale100_pose000_0.npy
"""
# Standard Library
import argparse
import os
import sys
from pathlib import Path

# Third Party
import coacd
import numpy as np
import trimesh


def convex_decompose(input_obj: str, pieces_dir: str):
    """Run CoACD; save each convex piece as convex_piece_NNN.obj under pieces_dir."""
    os.makedirs(pieces_dir, exist_ok=True)
    tm = trimesh.load(input_obj, force="mesh")
    mesh = coacd.Mesh(tm.vertices, tm.faces)
    parts = coacd.run_coacd(mesh, threshold=0.05, max_convex_hull=-1, mcts_max_depth=3)
    piece_paths = []
    for i, (verts, tris) in enumerate(parts):
        piece_tm = trimesh.Trimesh(vertices=verts, faces=tris)
        piece_path = os.path.join(pieces_dir, f"convex_piece_{i:03d}.obj")
        piece_tm.export(piece_path)
        piece_paths.append(piece_path)
    return piece_paths


def write_urdf(pieces_dir: str, visual_mesh_relpath: str, urdf_out: str):
    """Emit a DGN_2k-format URDF: one link per convex piece, link name = 'link_<piece_filename>',
    both visual and collision point at the same convex piece mesh. Joints chain the links
    with `link_<piece>_joint` fixed-joints, parent = first link.

    `visual_mesh_relpath` is unused in this format (kept for backward compat) — BODex's
    UrdfKinematicsParser with `coacd_obj_version=True` treats every link's visual as a
    convex piece of the object.
    """
    import lxml.etree as et

    root = et.Element("robot", name="root")
    urdf_dir = os.path.dirname(urdf_out)

    piece_names = sorted(os.listdir(pieces_dir))
    if not piece_names:
        raise RuntimeError(f"No convex pieces in {pieces_dir}")

    for piece_filename in piece_names:
        link_name = f"link_{piece_filename}"
        rel = os.path.relpath(os.path.join(pieces_dir, piece_filename), urdf_dir)

        link = et.SubElement(root, "link", name=link_name)
        inertial = et.SubElement(link, "inertial")
        et.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
        visual = et.SubElement(link, "visual")
        et.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
        vg = et.SubElement(visual, "geometry")
        et.SubElement(vg, "mesh", filename=rel, scale="1.0 1.0 1.0")
        collision = et.SubElement(link, "collision")
        et.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
        cg = et.SubElement(collision, "geometry")
        et.SubElement(cg, "mesh", filename=rel, scale="1.0 1.0 1.0")

    root_link = f"link_{piece_names[0]}"
    for piece_filename in piece_names[1:]:
        child_link = f"link_{piece_filename}"
        joint = et.SubElement(root, "joint", name=f"{child_link}_joint", type="fixed")
        et.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
        et.SubElement(joint, "parent", link=root_link)
        et.SubElement(joint, "child", link=child_link)

    tree = et.ElementTree(root)
    tree.write(urdf_out, pretty_print=True, xml_declaration=True, encoding="utf-8")


def compute_info(simplified_obj: str):
    tm = trimesh.load(simplified_obj, force="mesh")
    obb_length = tm.bounding_box_oriented.primitive.extents
    try:
        gravity_center = tm.center_mass
    except Exception:
        gravity_center = np.mean(tm.vertices, axis=0)
    try:
        density = float(tm.density)
        mass = float(tm.mass)
    except Exception:
        density = 1000.0
        mass = density * float(tm.volume) if tm.is_watertight else 0.1
    return {
        "gravity_center": np.asarray(gravity_center, dtype=float).tolist(),
        "obb": np.asarray(obb_length, dtype=float).tolist(),
        "scale": 1.0,
        "density": density,
        "mass": mass,
    }


def process_one_obj(input_obj: str, obj_id: str, out_root: str):
    """Full pipeline for one .obj -> processed_data/<obj_id>/ layout."""
    obj_dir = os.path.join(out_root, obj_id)
    mesh_dir = os.path.join(obj_dir, "mesh")
    urdf_dir = os.path.join(obj_dir, "urdf")
    pieces_dir = os.path.join(urdf_dir, "meshes")
    info_dir = os.path.join(obj_dir, "info")
    for d in (mesh_dir, urdf_dir, info_dir):
        os.makedirs(d, exist_ok=True)

    # 1. Copy/re-export as simplified.obj (strip materials to avoid .mtl headaches)
    tm = trimesh.load(input_obj, force="mesh")
    tm.visual = trimesh.visual.ColorVisuals()
    simplified_obj = os.path.join(mesh_dir, "simplified.obj")
    tm.export(simplified_obj)

    # 2. Convex decomposition
    convex_decompose(simplified_obj, pieces_dir)

    # 3. URDF — visual = ../mesh/simplified.obj, collision = ./meshes/convex_piece_NNN.obj
    urdf_out = os.path.join(urdf_dir, "coacd.urdf")
    visual_rel = os.path.relpath(simplified_obj, urdf_dir)
    write_urdf(pieces_dir, visual_rel, urdf_out)

    # 4. info/simplified.json
    info = compute_info(simplified_obj)
    import json
    with open(os.path.join(info_dir, "simplified.json"), "w") as f:
        json.dump(info, f, indent=2)

    return info


def write_scene_cfg(scene_path: str, obj_id: str, scene_rel_paths: dict,
                    object_pose: np.ndarray, scale: float,
                    include_table: bool = False,
                    task_type: str = "force_closure",
                    table_z: float = 0.0):
    """Write a scene_cfg .npy.

    scene_rel_paths: dict with keys file_path, info_path, urdf_path — each a
        path relative to scene_path's parent dir.
    object_pose: length-7 array [x, y, z, qw, qx, qy, qz].
    table_z: world-frame height of the table top (only used if include_table=True).
    """
    entry = {
        "type": "rigid_object",
        "scale": np.array([scale, scale, scale], dtype=float),
        "pose": np.asarray(object_pose, dtype=float),
        **scene_rel_paths,
    }
    scene = {obj_id: entry}
    if include_table:
        scene["table"] = {
            "type": "plane",
            "pose": np.array([0., 0., table_z, 1., 0., 0., 0.]),
            "size": np.array([0., 0., 1.]),
        }
    rel_scene_id = os.path.relpath(scene_path, os.path.dirname(os.path.dirname(os.path.dirname(scene_path))))
    rel_scene_id = os.path.splitext(rel_scene_id)[0]
    task = {"obj_name": obj_id}
    if task_type == "force_closure":
        task["type"] = "force_closure"
    else:
        task["type"] = "slide"
        task["axis"] = np.array([0., 0., 1.])
        task["distance"] = 0.1
    cfg = {"scene": scene, "scene_id": rel_scene_id, "task": task}
    os.makedirs(os.path.dirname(scene_path), exist_ok=True)
    np.save(scene_path, cfg, allow_pickle=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, help="Dir of raw .obj files")
    p.add_argument("--processed_root", required=True,
                   help="Output dir for processed_data/ layout")
    p.add_argument("--scene_cfg_root", required=True,
                   help="Output dir for scene_cfg/ layout")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Scene scale applied to the object (default: 1.0 — trust iPhone-scan real scale)")
    p.add_argument("--ra_ur3_xyz", type=float, nargs=3, default=[0.0, 0.4, 0.75],
                   help="Object world position for ra_ur3 tabletop scene. Default matches Isaac Lab "
                        "silver-table top at z=0.75, object at y=0.4 (reachable from overhead).")
    p.add_argument("--la_ur3e_xyz", type=float, nargs=3, default=[0.0, 0.4, 0.75],
                   help="Object world position for la_ur3e tabletop scene.")
    p.add_argument("--table_z", type=float, default=0.75,
                   help="Height of table top in world frame (default: 0.75m per Isaac Lab motion_desk).")
    args = p.parse_args()

    os.makedirs(args.processed_root, exist_ok=True)
    os.makedirs(args.scene_cfg_root, exist_ok=True)

    obj_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".obj")])
    print(f"Found {len(obj_files)} .obj files: {obj_files}")

    for obj_file in obj_files:
        obj_id = os.path.splitext(obj_file)[0]
        print(f"[{obj_id}] preprocessing...")
        input_obj = os.path.join(args.input_dir, obj_file)
        try:
            info = process_one_obj(input_obj, obj_id, args.processed_root)
        except Exception as e:
            print(f"[{obj_id}] FAILED: {e}")
            continue

        # Scene_cfg — paths are relative to the scene_cfg .npy's parent directory.
        # scene_cfg/<obj_id>/floating/scale100.npy, so "../../../processed_data/..." reaches processed_data/
        obj_rel = f"../../../processed_data/{obj_id}"
        scene_paths = {
            "file_path": f"{obj_rel}/mesh/simplified.obj",
            "urdf_path": f"{obj_rel}/urdf/coacd.urdf",
            "info_path": f"{obj_rel}/info/simplified.json",
        }
        # processed_root may be the same directory that contains scene_cfg_root.
        # If user passes separate locations, path must still resolve. We assume
        # processed_data/ and scene_cfg/ are siblings under the same parent
        # (which is the BODex convention).

        # Floating scene_cfg: object at origin, identity quaternion
        gc = info["gravity_center"]
        obb = info["obb"]
        # Center the object at origin (floating) by subtracting gravity center in x/y,
        # and lift it above z=0 so the hand has clearance.
        fl_pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        fl_path = os.path.join(args.scene_cfg_root, obj_id, "floating",
                               f"scale{int(args.scale*100):03d}.npy")
        write_scene_cfg(fl_path, obj_id, scene_paths, fl_pose, args.scale,
                        include_table=False, task_type="force_closure")

        # Tabletop pose in WORLD frame (z=0 is ground, table top at args.table_z).
        # Object rests with its bottom on the table surface → center_z = table_z + obb_z/2.
        # The 10cm inflated-hull seed filter (collision_free=False in our configs) is
        # bypassed, but we still lift +0.02m for mesh-tolerance margin.
        tabletop_z = args.table_z + float(obb[2]) / 2.0 + 0.02
        ra_pose = np.array([args.ra_ur3_xyz[0], args.ra_ur3_xyz[1], tabletop_z,
                            1.0, 0.0, 0.0, 0.0])
        ra_path = os.path.join(args.scene_cfg_root, obj_id, "tabletop_ra_ur3_allegro",
                               f"scale{int(args.scale*100):03d}_pose000_0.npy")
        write_scene_cfg(ra_path, obj_id, scene_paths, ra_pose, args.scale,
                        include_table=True, task_type="slide", table_z=args.table_z)

        la_pose = np.array([args.la_ur3e_xyz[0], args.la_ur3e_xyz[1], tabletop_z,
                            1.0, 0.0, 0.0, 0.0])
        la_path = os.path.join(args.scene_cfg_root, obj_id, "tabletop_la_ur3e_allegro",
                               f"scale{int(args.scale*100):03d}_pose000_0.npy")
        write_scene_cfg(la_path, obj_id, scene_paths, la_pose, args.scale,
                        include_table=True, task_type="slide", table_z=args.table_z)

        print(f"[{obj_id}] OK  obb={[f'{x:.3f}' for x in obb]}  gc={[f'{x:.3f}' for x in gc]}")

    print("Done.")


if __name__ == "__main__":
    main()
