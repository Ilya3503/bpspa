"""
Алгоритмы обработки облака точек.
Все функции — чистые, без побочных эффектов кроме файлов на диск.
Никаких WebSocket-ов, FastAPI и прочего здесь нет.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

log = logging.getLogger(__name__)


# ==============================================================================
# ЗАГРУЗКА/СОХРАНЕНИЕ
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
# MERGE двух облаков
# ==============================================================================

def load_calibration_matrix(path: str) -> Tuple[np.ndarray, bool]:
    """
    Возвращает (T_4x4, is_stub).
    Если файл не найден или не существует — возвращает единичную матрицу.
    """
    p = Path(path)
    if not p.exists():
        log.warning(f"Калибровочный файл не найден: {p}. Используется единичная матрица.")
        return np.eye(4), True
    T = np.load(str(p))
    if T.shape != (4, 4):
        log.warning(f"Калибровочная матрица не 4x4: {T.shape}. Заменяю на единичную.")
        return np.eye(4), True
    is_stub = bool(np.allclose(T, np.eye(4)))
    return T, is_stub


def merge_two_clouds(file_a: str, file_b: str, T_b_to_a: np.ndarray,
                     output_dir: str = "data",
                     voxel_size: float = 0.005) -> str:
    """
    Загружает два .ply, применяет трансформацию к второму, объединяет, сохраняет.
    Возвращает путь к merged файлу.
    """
    pcd_a = load_pcd(file_a)
    pcd_b = load_pcd(file_b)

    if voxel_size > 0:
        pcd_a = pcd_a.voxel_down_sample(voxel_size)
        pcd_b = pcd_b.voxel_down_sample(voxel_size)

    pts_b = np.asarray(pcd_b.points)
    ones = np.ones((pts_b.shape[0], 1))
    pts_b_h = np.hstack([pts_b, ones])
    pts_b_transformed = (T_b_to_a @ pts_b_h.T).T[:, :3]

    pts_a = np.asarray(pcd_a.points)
    pts_merged = np.vstack([pts_a, pts_b_transformed])

    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(pts_merged)

    # цвета — если есть в обоих
    if pcd_a.has_colors() and pcd_b.has_colors():
        colors_a = np.asarray(pcd_a.colors)
        colors_b = np.asarray(pcd_b.colors)
        merged.colors = o3d.utility.Vector3dVector(np.vstack([colors_a, colors_b]))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = Path(output_dir) / f"merged_{ts}.ply"
    o3d.io.write_point_cloud(str(out_path), merged)
    log.info(f"Merged saved: {out_path}, points={len(merged.points)}")
    return str(out_path)


# ==============================================================================
# ПРЕДОБРАБОТКА
# ==============================================================================

def clean_nan(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points)
    mask = np.isfinite(pts).all(axis=1)
    clean = o3d.geometry.PointCloud()
    clean.points = o3d.utility.Vector3dVector(pts[mask])
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        clean.colors = o3d.utility.Vector3dVector(colors[mask])
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


# ==============================================================================
# КЛАСТЕРИЗАЦИЯ
# ==============================================================================

def cluster_dbscan(pcd, eps=0.025, min_points=50,
                   min_extent=0.02, max_extent=0.30) -> List[o3d.geometry.PointCloud]:
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return []
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    clusters = []
    for lab in np.unique(labels):
        if lab == -1:
            continue
        idx = np.where(labels == lab)[0]
        cluster = pcd.select_by_index(idx.tolist())
        cluster_pts = np.asarray(cluster.points)
        z_range = cluster_pts[:, 2].max() - cluster_pts[:, 2].min()
        if z_range < 0.003:
            continue
        extent = cluster.get_axis_aligned_bounding_box().get_extent()
        max_dim = float(np.max(extent))
        if max_dim < min_extent or max_dim > max_extent:
            continue
        clusters.append(cluster)
    return clusters


def cluster_info(cluster: o3d.geometry.PointCloud, cluster_id: int) -> dict:
    try:
        obb = cluster.get_oriented_bounding_box()
        center = list(map(float, obb.center))
        extent = list(map(float, obb.extent))
        R = np.asarray(obb.R)
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    except Exception:
        aabb = cluster.get_axis_aligned_bounding_box()
        center = list(map(float, aabb.get_center()))
        extent = list(map(float, aabb.get_extent()))
        R = np.eye(3)
        yaw = 0.0
    return {
        "id": cluster_id,
        "center": center,
        "extent": extent,
        "yaw": yaw,
        "rotation_matrix": R.tolist(),
        "points_count": int(len(cluster.points)),
    }


# ==============================================================================
# ICP — собственная реализация на numpy + scipy
# ==============================================================================

def rotation_to_quat(R: np.ndarray) -> List[float]:
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def _icp_step(src, tgt, max_dist):
    if len(src) == 0 or len(tgt) == 0:
        return np.eye(4), float('inf'), 0.0

    tree_tgt = KDTree(tgt)
    dists_forward, idx_forward = tree_tgt.query(src, k=1)
    tree_src = KDTree(src)
    dists_backward, _ = tree_src.query(tgt, k=1)

    inliers_fwd = dists_forward < max_dist
    inliers_bwd = dists_backward < max_dist
    num_inliers = min(inliers_fwd.sum(), inliers_bwd.sum())
    fitness = float(num_inliers) / max(len(src), len(tgt))
    rmse = float(np.sqrt((dists_forward[inliers_fwd] ** 2).mean())) if inliers_fwd.any() else float('inf')

    src_matched = src[inliers_fwd]
    tgt_matched = tgt[idx_forward[inliers_fwd]]
    if len(src_matched) < 6:
        return np.eye(4), float('inf'), 0.0

    src_c = src_matched - src_matched.mean(axis=0)
    tgt_c = tgt_matched - tgt_matched.mean(axis=0)
    H = src_c.T @ tgt_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = tgt_matched.mean(axis=0) - R @ src_matched.mean(axis=0)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T, rmse, fitness


def run_icp(cluster: o3d.geometry.PointCloud,
            cad_model: o3d.geometry.PointCloud,
            voxel_size: float = 0.003,
            max_correspondence_distance: float = 0.015,
            max_iterations: int = 50,
            fitness_threshold: float = 0.24) -> dict:
    """
    ICP через numpy. Open3D registration_icp падает на Jetson 0.18.0.
    Возвращает dict с position/orientation/fitness/cad_points_transformed.
    """
    pts_cad = np.asarray(cad_model.points).copy()
    pts_cad -= pts_cad.mean(axis=0)

    cluster_extent = np.asarray(cluster.get_axis_aligned_bounding_box().get_extent())
    cad_extent = pts_cad.max(axis=0) - pts_cad.min(axis=0)
    scale = np.mean(cluster_extent) / (np.mean(cad_extent) + 1e-9)
    pts_cad *= scale

    cluster_pts = np.asarray(cluster.points).copy()
    cluster_center = cluster_pts.mean(axis=0)
    pts_cad += cluster_center

    def _ds(pts, vsize):
        if vsize <= 0 or len(pts) == 0:
            return pts
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(pts)
        p = p.voxel_down_sample(vsize)
        return np.asarray(p.points)

    src = _ds(pts_cad, voxel_size)
    tgt = _ds(cluster_pts, voxel_size)

    if len(src) < 6 or len(tgt) < 6:
        return _obb_fallback(cluster, reason="too few points for ICP")

    T_total = np.eye(4)
    prev_rmse = float('inf')
    fitness = 0.0
    rmse = float('inf')

    for it in range(max_iterations):
        T_step, rmse, fitness = _icp_step(src, tgt, max_correspondence_distance)
        src_h = np.hstack([src, np.ones((len(src), 1))])
        src = (T_step @ src_h.T).T[:, :3]
        T_total = T_step @ T_total
        if abs(prev_rmse - rmse) < 1e-6:
            break
        prev_rmse = rmse

    if fitness < fitness_threshold:
        result = _obb_fallback(cluster, reason=f"low ICP fitness {fitness:.3f}")
        result["icp_fitness"] = fitness
        return result

    R_final = T_total[:3, :3]
    T_pose = np.eye(4)
    T_pose[:3, :3] = R_final
    T_pose[:3, 3] = cluster_center

    return {
        "method": "icp",
        "fitness": float(fitness),
        "inlier_rmse": float(rmse),
        "transformation": T_pose.tolist(),
        "position": cluster_center.tolist(),
        "orientation": rotation_to_quat(R_final),
        "extent": list(map(float, cluster_extent)),
        "cad_points_transformed": src,    # numpy Nx3 — для визуализации в UI
    }


def _obb_fallback(cluster, reason: str = "") -> dict:
    obb = cluster.get_oriented_bounding_box()
    R = np.asarray(obb.R)
    return {
        "method": "obb_fallback",
        "reason": reason,
        "fitness": None,
        "inlier_rmse": None,
        "position": list(map(float, obb.center)),
        "orientation": rotation_to_quat(R),
        "extent": list(map(float, obb.extent)),
        "cad_points_transformed": None,
    }


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
    # отдельно убираем numpy-массив cad_points_transformed (он не JSON-сериализуется и большой)
    def _clean(d):
        if isinstance(d, dict):
            return {k: _clean(v) for k, v in d.items() if k != "cad_points_transformed"}
        if isinstance(d, list):
            return [_clean(x) for x in d]
        return d
    with open(p, "w", encoding="utf-8") as f:
        json.dump(_clean(result), f, ensure_ascii=False, indent=2)
    return str(p)