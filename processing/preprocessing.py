import logging
from pathlib import Path
from typing import Optional
import numpy as np
import open3d as o3d
log = logging.getLogger(__name__)

# ==============================================================================
# ЗАГРУЗКА / СОХРАНЕНИЕ
# ==============================================================================

def load_pcd(path: str) -> o3d.geometry.PointCloud:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл не найден: {p}")
    pcd = o3d.io.read_point_cloud(str(p))
    if len(pcd.points) == 0:
        raise ValueError(f"Пустой файл: {p}")
    return pcd


def save_pcd(pcd: o3d.geometry.PointCloud, path: str) -> str:
    o3d.io.write_point_cloud(str(path), pcd)
    return str(path)



# ==============================================================================
# ПРЕДОБРАБОТКА
# ==============================================================================

def clean_nan(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points)
    mask = np.isfinite(pts).all(axis=1)
    clean = o3d.geometry.PointCloud()
    clean.points = o3d.utility.Vector3dVector(pts[mask])
    if pcd.has_colors():
        clean.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
    return clean


def crop_roi(pcd: o3d.geometry.PointCloud,
             x_range: list, y_range: list, z_range: list) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return pcd
    mask = (
        (pts[:, 0] >= x_range[0]) & (pts[:, 0] <= x_range[1]) &
        (pts[:, 1] >= y_range[0]) & (pts[:, 1] <= y_range[1]) &
        (pts[:, 2] >= z_range[0]) & (pts[:, 2] <= z_range[1])
    )
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts[mask])
    if pcd.has_colors():
        out.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
    return out


def voxel_downsample(pcd, voxel_size):
    return pcd.voxel_down_sample(voxel_size=voxel_size)


def statistical_filter(pcd, nb_neighbors=20, std_ratio=2.0):
    if len(pcd.points) < nb_neighbors:
        return pcd
    cl, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return cl


def remove_plane(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    pts = np.asarray(pcd.points)
    if pts.shape[0] < ransac_n:
        return pcd, None
    try:
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        return pcd.select_by_index(inliers, invert=True), list(plane_model)
    except Exception as e:
        log.error(f"remove_plane failed: {e}")
        return pcd, None
