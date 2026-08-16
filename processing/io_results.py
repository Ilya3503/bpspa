import json
import logging
from pathlib import Path
from typing import List
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from processing.geometry import load_cam_to_table, _apply_cam_to_table_to_pose
log = logging.getLogger(__name__)


# ==============================================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ==============================================================================

def save_clusters(clusters, clusters_dir: str) -> List[str]:
    Path(clusters_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    for i, c in enumerate(clusters):
        p = Path(clusters_dir) / f"cluster_{i:03d}.ply"
        o3d.io.write_point_cloud(str(p), c)
        paths.append(str(p))
    return paths


def make_annotated_ply(pcd, clusters, results_dir: str) -> str:
    out = Path(results_dir) / "annotated_pointcloud.ply"
    if not clusters:
        o3d.io.write_point_cloud(str(out), pcd)
        return str(out)
    cmap = plt.get_cmap("tab10")(np.linspace(0, 1, 10))[:, :3]
    all_p, all_c = [], []
    for i, c in enumerate(clusters):
        pts = np.asarray(c.points)
        clr = np.tile(cmap[i % len(cmap)], (pts.shape[0], 1))
        all_p.append(pts)
        all_c.append(clr)
    m = o3d.geometry.PointCloud()
    m.points = o3d.utility.Vector3dVector(np.vstack(all_p))
    m.colors = o3d.utility.Vector3dVector(np.vstack(all_c))
    o3d.io.write_point_cloud(str(out), m)
    return str(out)


def save_position_json(result: dict, results_dir: str) -> str:
    p = Path(results_dir) / "position.json"

    # Перевод координат камеры → стола перед записью.
    # Применяем ко всем позам внутри result["clusters"][*]["pose"].
    T_table = load_cam_to_table()
    if T_table is not None:
        for ci in result.get("clusters", []):
            pose = ci.get("pose")
            if isinstance(pose, dict):
                _apply_cam_to_table_to_pose(pose, T_table)
        result["coords_frame"] = "table"
    else:
        result["coords_frame"] = "camera"

    def _clean(d):
        if isinstance(d, dict):
            return {k: _clean(v) for k, v in d.items() if k != "cad_points_transformed"}
        if isinstance(d, list):
            return [_clean(x) for x in d]
        return d
    with open(p, "w", encoding="utf-8") as f:
        json.dump(_clean(result), f, ensure_ascii=False, indent=2)
    return str(p)
