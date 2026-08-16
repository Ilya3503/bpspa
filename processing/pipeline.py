"""
Алгоритмы обработки облака точек.
Все функции — чистые, без побочных эффектов кроме файлов на диск.
Никаких WebSocket-ов, FastAPI и прочего здесь нет.
"""
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

log = logging.getLogger(__name__)


# Кэш FPFH-дескрипторов CAD-модели. Считаем один раз на модель — переиспользуем.
_FPFH_CACHE = {
    "cad_id": None,
    "voxel_size": None,
    "cad_down": None,        # даунсэмпленное облако CAD
    "cad_fpfh": None,        # FPFH-фичи CAD
}

# Матрица перевода координат камеры → координаты стола.
# Применяется к позам перед записью в position.json.
_CAM_TO_TABLE_PATH = Path("hardware") / "transform_cam_to_world.npy"










# ==============================================================================
# ICP — собственная реализация на numpy + scipy
# (Open3D registration_icp падает на Jetson 0.18.0)
# ==============================================================================

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





def _icp_loop(src: np.ndarray, tgt: np.ndarray,
              max_correspondence_distance: float,
              max_iterations: int) -> Tuple[np.ndarray, float, float]:
    """Накопительный ICP. Возвращает (T_total, fitness, rmse)."""
    T_total = np.eye(4)
    prev_rmse = float('inf')
    fitness, rmse = 0.0, float('inf')

    for _ in range(max_iterations):
        T_step, rmse, fitness = _icp_step(src, tgt, max_correspondence_distance)
        src_h = np.hstack([src, np.ones((len(src), 1))])
        src = (T_step @ src_h.T).T[:, :3]
        T_total = T_total @ T_step
        if abs(prev_rmse - rmse) < 1e-6:
            break
        prev_rmse = rmse

    return T_total, fitness, rmse


def run_icp(cluster: o3d.geometry.PointCloud,
            cad_model: o3d.geometry.PointCloud,
            voxel_size: float = 0.003,
            max_correspondence_distance: float = 0.015,
            max_iterations: int = 50,
            fitness_threshold: float = 0.24) -> dict:
    """
    Локальный ICP. Начальное приближение — совмещение центров.
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

    src = _ds_np(pts_cad, voxel_size)
    tgt = _ds_np(cluster_pts, voxel_size)

    if len(src) < 6 or len(tgt) < 6:
        return _obb_fallback(cluster, reason="too few points for ICP")

    T_total, fitness, rmse = _icp_loop(src, tgt, max_correspondence_distance, max_iterations)

    if fitness < fitness_threshold:
        result = _obb_fallback(cluster, reason=f"low ICP fitness {fitness:.3f}")
        result["icp_fitness"] = float(fitness)
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
        "cad_points_transformed": src,
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
# GLOBAL REGISTRATION: FPFH + RANSAC
# Альтернатива PPF: находит грубую позу с нуля, без начального приближения.
# Затем результат уточняется через run_icp с этим начальным трансформом.
# ==============================================================================

def _preprocess_for_fpfh(pcd: o3d.geometry.PointCloud,
                        voxel_size: float,
                        normal_radius_factor: float = 2.0,
                        fpfh_radius_factor: float = 5.0,
                        fpfh_max_nn: int = 100):
    """
    Даунсэмпл + нормали + FPFH-дескрипторы.
    Возвращает (downsampled_pcd, fpfh_features).
    """
    down = pcd.voxel_down_sample(voxel_size)

    normal_radius = voxel_size * normal_radius_factor
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )

    fpfh_radius = voxel_size * fpfh_radius_factor
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=fpfh_max_nn)
    )
    return down, fpfh


def _train_fpfh_if_needed(cad_pcd: o3d.geometry.PointCloud,
                          cad_name: str,
                          voxel_size: float,
                          cfg: dict) -> Tuple[o3d.geometry.PointCloud, object]:
    """Считает FPFH для CAD один раз и кэширует."""
    n_cad = len(cad_pcd.points)
    # отпечаток содержимого — на случай если файл изменили, но имя не сменили
    pts_sample = np.asarray(cad_pcd.points)[:10].tobytes() if n_cad > 0 else b""
    cad_id = f"{cad_name}_{n_cad}_{hash(pts_sample)}"

    if (_FPFH_CACHE["cad_id"] == cad_id and
        _FPFH_CACHE["voxel_size"] == voxel_size and
        _FPFH_CACHE["cad_down"] is not None):
        return _FPFH_CACHE["cad_down"], _FPFH_CACHE["cad_fpfh"]

    log.info(f"[FPFH] Вычисляю дескрипторы CAD {cad_name} ({n_cad} точек)...")
    t0 = time.perf_counter()
    down, fpfh = _preprocess_for_fpfh(
        cad_pcd, voxel_size,
        normal_radius_factor=cfg.get("normal_radius_factor", 2.0),
        fpfh_radius_factor=cfg.get("fpfh_radius_factor", 5.0),
        fpfh_max_nn=cfg.get("fpfh_max_nn", 100),
    )
    log.info(f"[FPFH] CAD: {len(down.points)} точек, готово за {time.perf_counter()-t0:.2f}с")

    _FPFH_CACHE.update({
        "cad_id": cad_id,
        "voxel_size": voxel_size,
        "cad_down": down,
        "cad_fpfh": fpfh,
    })
    return down, fpfh


def _ransac_global_registration(source_down, source_fpfh,
                                target_down, target_fpfh,
                                voxel_size: float,
                                cfg: dict):
    """
    Запускает RANSAC global registration через Open3D.
    source = CAD, target = scene cluster.
    """
    distance_threshold = voxel_size * cfg.get("distance_threshold_factor", 1.5)
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=cfg.get("mutual_filter", True),
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=cfg.get("ransac_n", 3),
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                cfg.get("edge_length_threshold", 0.9)),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            cfg.get("max_iterations", 100000),
            cfg.get("confidence", 0.999)),
    )


def run_global_then_icp(cluster: o3d.geometry.PointCloud,
                        cad_model: o3d.geometry.PointCloud,
                        cad_name: str,
                        global_cfg: dict,
                        icp_cfg: dict) -> dict:
    """
    FPFH+RANSAC для грубой позы → ICP для уточнения.
    Возвращает тот же формат, что run_icp, плюс поля global_fitness/global_rmse.
    """
    voxel_size = global_cfg.get("voxel_size", 0.005)

    # 1) FPFH для CAD (с кэшем)
    cad_down, cad_fpfh = _train_fpfh_if_needed(cad_model, cad_name, voxel_size, global_cfg)

    # 2) FPFH для сцены (кластера)
    t0 = time.perf_counter()
    cluster_down, cluster_fpfh = _preprocess_for_fpfh(
        cluster, voxel_size,
        normal_radius_factor=global_cfg.get("normal_radius_factor", 2.0),
        fpfh_radius_factor=global_cfg.get("fpfh_radius_factor", 5.0),
        fpfh_max_nn=global_cfg.get("fpfh_max_nn", 100),
    )

    if len(cluster_down.points) < 6:
        log.warning(f"[FPFH] кластер слишком мал ({len(cluster_down.points)}) — fallback ICP")
        return run_icp(
            cluster, cad_model,
            voxel_size=icp_cfg["voxel_size"],
            max_correspondence_distance=icp_cfg["max_correspondence_distance"],
            max_iterations=icp_cfg["max_iterations"],
            fitness_threshold=icp_cfg["fitness_threshold"],
        )

    result = _ransac_global_registration(
        cad_down, cad_fpfh, cluster_down, cluster_fpfh,
        voxel_size, global_cfg,
    )
    # СРАЗУ извлекаем данные в plain Python типы и numpy.
    # Объект `result` (RegistrationResult) больше не трогаем — он остаётся в Open3D,
    # и любое обращение к нему после следующей PointCloud-операции может крашнуть процесс.
    global_fitness = float(result.fitness)
    global_rmse = float(result.inlier_rmse)
    initial_T = np.array(result.transformation, dtype=np.float64, copy=True)
    del result  # явный сигнал GC отдать объект Open3D как можно раньше

    log.info(f"[FPFH+RANSAC] fitness={global_fitness:.3f} rmse={global_rmse:.4f} "
             f"за {time.perf_counter() - t0:.2f}с")

    if global_fitness < global_cfg.get("min_fitness", 0.1):
        log.warning(f"[FPFH+RANSAC] низкий fitness {global_fitness:.3f} — fallback ICP")
        fallback = run_icp(
            cluster, cad_model,
            voxel_size=icp_cfg["voxel_size"],
            max_correspondence_distance=icp_cfg["max_correspondence_distance"],
            max_iterations=icp_cfg["max_iterations"],
            fitness_threshold=icp_cfg["fitness_threshold"],
        )
        fallback["global_fitness_attempted"] = global_fitness
        return fallback

    # 4) ICP refine с найденной позой как initial guess (всё на numpy)
    refined = _icp_with_initial_transform(
        cluster, cad_model,
        initial_transform=initial_T,
        voxel_size=icp_cfg["voxel_size"],
        max_correspondence_distance=icp_cfg["max_correspondence_distance"],
        max_iterations=icp_cfg["max_iterations"],
        fitness_threshold=icp_cfg["fitness_threshold"],
    )
    if refined.get("method") == "icp":
        refined["method"] = "fpfh+icp"
    refined["global_fitness"] = global_fitness
    refined["global_rmse"] = global_rmse
    return refined



def _icp_with_initial_transform(cluster: o3d.geometry.PointCloud,
                                cad_model: o3d.geometry.PointCloud,
                                initial_transform: np.ndarray,
                                voxel_size: float,
                                max_correspondence_distance: float,
                                max_iterations: int,
                                fitness_threshold: float) -> dict:
    """ICP с начальным приближением (от global registration), без масштабирования."""
    pts_cad = np.asarray(cad_model.points, dtype=np.float64).copy()
    pts_h = np.hstack([pts_cad, np.ones((len(pts_cad), 1))])
    pts_cad = (initial_transform @ pts_h.T).T[:, :3]

    cluster_pts = np.asarray(cluster.points, dtype=np.float64).copy()

    src = _ds_np(pts_cad, voxel_size)
    tgt = _ds_np(cluster_pts, voxel_size)

    if len(src) < 6 or len(tgt) < 6:
        return _obb_fallback(cluster, reason="too few points after global init")

    T_step_total, fitness, rmse = _icp_loop(src, tgt, max_correspondence_distance, max_iterations)
    T_total = T_step_total @ initial_transform

    if fitness < fitness_threshold:
        result = _obb_fallback(cluster, reason=f"low fitness after global+ICP {fitness:.3f}")
        result["icp_fitness"] = float(fitness)
        return result

    R_final = T_total[:3, :3]
    t_final = T_total[:3, 3]

    return {
        "method": "icp",   # переопределится в run_global_then_icp
        "fitness": float(fitness),
        "inlier_rmse": float(rmse),
        "transformation": T_total.tolist(),
        "position": t_final.tolist(),
        "orientation": rotation_to_quat(R_final),
        "extent": [float(src.max(axis=0)[i] - src.min(axis=0)[i]) for i in range(3)],
        "cad_points_transformed": src,
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
