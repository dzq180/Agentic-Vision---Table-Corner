import blenderproc as bproc
import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image


def _to_numpy(data, key):
    value = data.get(key)
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim == 3 and key == "colors":
        arr = arr[None, ...]
    return arr


def _random_color(rng):
    return [float(rng.uniform(0.1, 0.95)), float(rng.uniform(0.1, 0.95)), float(rng.uniform(0.1, 0.95)), 1.0]


def _set_material(obj, name, rgba):
    mat = bproc.material.create(name)
    mat.set_principled_shader_value("Base Color", rgba)
    mat.set_principled_shader_value("Roughness", float(np.random.uniform(0.2, 0.7)))
    obj.replace_materials(mat)


def main():
    bproc.init()

    rng = np.random.default_rng(int(os.getenv("BPROC_SEED", "42")))
    num_samples = int(os.getenv("BPROC_NUM_SAMPLES", "24"))
    width = int(os.getenv("BPROC_WIDTH", "768"))
    height = int(os.getenv("BPROC_HEIGHT", "768"))

    script_dir = Path(__file__).resolve().parent
    output_root = script_dir / "output"
    run_name = os.getenv("BPROC_RUN_NAME", datetime.now().strftime("geom_dataset_%Y%m%d_%H%M%S"))
    dataset_dir = output_root / run_name
    image_dir = dataset_dir / "images"
    gt_dir = dataset_dir / "gt_json"
    hdf5_dir = dataset_dir / "hdf5"
    image_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    hdf5_dir.mkdir(parents=True, exist_ok=True)

    bproc.camera.set_resolution(width, height)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_segmentation_output(map_by=["instance", "category_id"])
    try:
        bproc.renderer.set_max_amount_of_samples(128)
    except AttributeError:
        pass

    # 地面与背景平面，便于几何关系和遮挡形成
    floor = bproc.object.create_primitive("PLANE")
    floor.set_scale([5, 5, 1])
    floor.set_location([0, 0, 0])
    floor.set_cp("category_id", 200)
    floor.set_cp("instance", 200)
    _set_material(floor, "floor_mat", [0.88, 0.88, 0.88, 1.0])

    back_wall = bproc.object.create_primitive("PLANE")
    back_wall.set_scale([5, 5, 1])
    back_wall.set_location([0, 2.5, 2.5])
    back_wall.set_rotation_euler([np.pi / 2, 0, 0])
    back_wall.set_cp("category_id", 201)
    back_wall.set_cp("instance", 201)
    _set_material(back_wall, "wall_mat", [0.78, 0.82, 0.9, 1.0])

    # 目标几何体（用于空间推理）
    primitive_types = ["CUBE", "CYLINDER", "CONE", "SPHERE"]
    target_objects = []
    num_targets = 6
    for idx in range(num_targets):
        prim = primitive_types[idx % len(primitive_types)]
        obj = bproc.object.create_primitive(prim)
        obj.set_cp("category_id", idx + 1)
        obj.set_cp("instance", idx + 1)
        obj.set_cp("is_target", True)
        obj.set_cp("shape_type", prim)
        obj.set_scale([
            float(rng.uniform(0.18, 0.45)),
            float(rng.uniform(0.18, 0.45)),
            float(rng.uniform(0.18, 0.45)),
        ])
        obj.set_location([
            float(rng.uniform(-1.4, 1.4)),
            float(rng.uniform(-0.3, 1.8)),
            float(rng.uniform(0.2, 0.9)),
        ])
        obj.set_rotation_euler([
            float(rng.uniform(0, np.pi)),
            float(rng.uniform(0, np.pi)),
            float(rng.uniform(0, np.pi)),
        ])
        _set_material(obj, f"target_mat_{idx}", _random_color(rng))
        target_objects.append(obj)

    # 遮挡体：增加推理难度（部分遮挡/重叠）
    occluders = []
    for occ_idx in range(3):
        occ = bproc.object.create_primitive("CUBE")
        occ.set_cp("category_id", 100 + occ_idx)
        occ.set_cp("instance", 100 + occ_idx)
        occ.set_cp("is_occluder", True)
        occ.set_scale([
            float(rng.uniform(0.12, 0.35)),
            float(rng.uniform(0.35, 0.85)),
            float(rng.uniform(0.25, 0.75)),
        ])
        occ.set_location([
            float(rng.uniform(-0.5, 0.6)),
            float(rng.uniform(0.4, 1.3)),
            float(rng.uniform(0.4, 1.2)),
        ])
        occ.set_rotation_euler([
            float(rng.uniform(0, np.pi / 5)),
            float(rng.uniform(0, np.pi / 5)),
            float(rng.uniform(0, np.pi)),
        ])
        _set_material(occ, f"occluder_mat_{occ_idx}", [0.18, 0.18, 0.18, 1.0])
        occluders.append(occ)

    # 灯光：多光源提高图像质量并制造阴影
    key_light = bproc.types.Light()
    key_light.set_type("AREA")
    key_light.set_location([2.5, -2.0, 4.5])
    key_light.set_energy(1200)

    fill_light = bproc.types.Light()
    fill_light.set_type("POINT")
    fill_light.set_location([-2.0, -1.0, 3.0])
    fill_light.set_energy(450)

    rim_light = bproc.types.Light()
    rim_light.set_type("POINT")
    rim_light.set_location([0.5, 2.8, 2.8])
    rim_light.set_energy(300)

    # 多视角采样：每个视角都会生成一张图片和一份GT
    camera_poses = []
    for _ in range(num_samples):
        cam_loc = np.array(
            [
                float(rng.uniform(-1.2, 1.2)),
                float(rng.uniform(-3.3, -2.6)),
                float(rng.uniform(1.3, 2.2)),
            ],
            dtype=np.float64,
        )
        poi = np.array([0.0, 0.9, 0.6], dtype=np.float64) + rng.uniform(-0.2, 0.2, 3)
        cam_rot_mat = bproc.camera.rotation_from_forward_vec(
            poi - cam_loc, inplane_rot=float(rng.uniform(-0.15, 0.15))
        )
        cam2world = np.eye(4, dtype=np.float64)
        cam2world[:3, :3] = cam_rot_mat
        cam2world[:3, 3] = cam_loc
        bproc.camera.add_camera_pose(cam2world)
        camera_poses.append(np.array(cam2world, dtype=np.float64))

    data = bproc.renderer.render()
    bproc.writer.write_hdf5(str(hdf5_dir), data)

    instance_maps = _to_numpy(data, "instance_segmaps")
    class_maps = _to_numpy(data, "class_segmaps")
    category_maps = _to_numpy(data, "category_id_segmaps")
    depth_maps = _to_numpy(data, "depth")
    color_maps = _to_numpy(data, "colors")
    k_mat = np.array(bproc.camera.get_intrinsics_as_K_matrix(), dtype=np.float64)

    if color_maps is None:
        raise RuntimeError("Rendering output does not contain required key: colors")
    if instance_maps is None and category_maps is None:
        raise RuntimeError("Rendering output does not contain usable segmentation maps.")

    manifest = {
        "run_name": run_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "num_samples": int(color_maps.shape[0]),
        "image_size_wh": [width, height],
        "image_dir": str(image_dir),
        "gt_json_dir": str(gt_dir),
        "files": [],
    }

    for frame_idx in range(color_maps.shape[0]):
        stem = f"sample_{frame_idx:05d}"
        image_name = f"{stem}.png"
        gt_name = f"{stem}.json"
        image_path = image_dir / image_name
        gt_path = gt_dir / gt_name

        rgb = color_maps[frame_idx]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        Image.fromarray(rgb).save(image_path)

        inst_map = instance_maps[frame_idx] if instance_maps is not None else None
        cls_map = class_maps[frame_idx] if class_maps is not None else None
        cat_map = category_maps[frame_idx] if category_maps is not None else None
        dep_map = depth_maps[frame_idx] if depth_maps is not None else None

        seg_map = inst_map
        id_field = "instance_id"
        if seg_map is None or np.max(seg_map) <= 0:
            seg_map = cat_map
            id_field = "category_id"
        if seg_map is None:
            seg_map = np.zeros((height, width), dtype=np.int32)

        instances = []
        for inst_id in np.unique(seg_map):
            inst_id = int(inst_id)
            if inst_id <= 0:
                continue

            mask = seg_map == inst_id
            ys, xs = np.where(mask)
            if xs.size == 0:
                continue

            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            bbox_area = max((x2 - x1 + 1) * (y2 - y1 + 1), 1)
            pix_cnt = int(mask.sum())
            cx = float(xs.mean())
            cy = float(ys.mean())

            record = {
                id_field: inst_id,
                "pixel_count": pix_cnt,
                "visible_ratio_in_bbox": float(pix_cnt / bbox_area),
                "centroid_uv": [cx, cy],
                "bbox_xyxy": [x1, y1, x2, y2],
            }

            if cls_map is not None:
                cls_values, cls_counts = np.unique(cls_map[mask], return_counts=True)
                record["majority_class_id"] = int(cls_values[np.argmax(cls_counts)])
            if cat_map is not None:
                cat_values, cat_counts = np.unique(cat_map[mask], return_counts=True)
                record["majority_category_id"] = int(cat_values[np.argmax(cat_counts)])

            if dep_map is not None:
                u = max(0, min(int(round(cx)), dep_map.shape[1] - 1))
                v = max(0, min(int(round(cy)), dep_map.shape[0] - 1))
                record["depth_at_centroid"] = float(dep_map[v, u])
                record["depth_min"] = float(dep_map[mask].min())
                record["depth_max"] = float(dep_map[mask].max())

            instances.append(record)

        frame_payload = {
            "sample_id": stem,
            "image_file": image_name,
            "gt_file": gt_name,
            "image_size_wh": [width, height],
            "cam2world": camera_poses[frame_idx].tolist() if frame_idx < len(camera_poses) else None,
            "k_matrix": k_mat.tolist(),
            "num_instances": len(instances),
            "instances": instances,
        }

        # 几何投影 GT：使用 BlenderProc 的 camera 投影工具，避免坐标系方向误差
        projected_targets = []
        for idx, obj in enumerate(target_objects):
            center_world = np.array(obj.get_location(), dtype=np.float64)
            uv = bproc.camera.project_points(center_world.reshape(1, 3), frame=frame_idx)[0]
            u, v = float(uv[0]), float(uv[1])
            in_image = 0 <= u < width and 0 <= v < height

            visible_by_depth = None
            if dep_map is not None and in_image:
                uu = max(0, min(int(round(u)), width - 1))
                vv = max(0, min(int(round(v)), height - 1))
                depth_at_pix = float(dep_map[vv, uu])
                visible_by_depth = depth_at_pix < 1e6
            else:
                depth_at_pix = None

            bbox_xyxy = None
            if hasattr(obj, "get_bound_box"):
                bbox_points = np.array(obj.get_bound_box(), dtype=np.float64)
                projected_pts = bproc.camera.project_points(bbox_points.reshape(-1, 3), frame=frame_idx)
                if projected_pts is not None and len(projected_pts) > 0:
                    us = [float(p[0]) for p in projected_pts]
                    vs = [float(p[1]) for p in projected_pts]
                    bbox_xyxy = [
                        float(max(0.0, min(us))),
                        float(max(0.0, min(vs))),
                        float(min(width - 1.0, max(us))),
                        float(min(height - 1.0, max(vs))),
                    ]

            projected_targets.append(
                {
                    "target_id": idx + 1,
                    "shape_type": obj.get_cp("shape_type"),
                    "center_3d_world": center_world.tolist(),
                    "center_2d_uv": [u, v],
                    "in_image": bool(in_image),
                    "visible_by_depth": bool(visible_by_depth) if visible_by_depth is not None else None,
                    "depth_at_projected_pixel": depth_at_pix,
                    "projected_bbox_xyxy": bbox_xyxy,
                }
            )

        frame_payload["projected_targets"] = projected_targets
        frame_payload["num_projected_targets"] = len(projected_targets)
        if not frame_payload["instances"]:
            # 某些 BlenderProc 版本中 segmap 可能为空，回退为几何投影结果，保证每帧都有GT可用。
            frame_payload["instances"] = [
                {
                    "target_id": t["target_id"],
                    "shape_type": t["shape_type"],
                    "centroid_uv": t["center_2d_uv"],
                    "bbox_xyxy": t["projected_bbox_xyxy"],
                    "in_image": t["in_image"],
                    "visible_by_depth": t["visible_by_depth"],
                }
                for t in projected_targets
            ]
            frame_payload["num_instances"] = len(frame_payload["instances"])

        with gt_path.open("w", encoding="utf-8") as f:
            json.dump(frame_payload, f, ensure_ascii=False, indent=2)

        manifest["files"].append(
            {
                "sample_id": stem,
                "image": image_name,
                "gt_json": gt_name,
            }
        )

    with (dataset_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Dataset generated: {dataset_dir}")
    print(f"Images: {image_dir}")
    print(f"GT json: {gt_dir}")
    print(f"Manifest: {dataset_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()