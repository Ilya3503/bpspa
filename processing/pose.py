"""
Pose Estimation (classical): FPFH+RANSAC для грубой позы → ICP для уточнения.
ICP — собственная реализация на numpy+scipy (Open3D registration_icp
падает на Jetson, Open3D 0.18.0). При любом провале — честный failed-result,
не поза-костыль.
"""
import logging
import time
from typing import Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

from processing.geometry import rotation_to_quat, _ds_np

log = logging.getLogger(__name__)

# FPFH-дескрипторы CAD считаются один раз на модель и переиспользуются.
_FPFH_CACHE = {"cad_id": None, "voxel_size": None, "cad_down": None, "cad_fpfh": None}


def _failed_result(reason: str = "", **extra) -> dict:
    """Явный отказ оценки позы — сигнал 'не распознано', а не поза.
    Уходит в position.json со status=failed."""
    return {
        "status": "failed", "method": "failed", "reason": reason,
        "fitness": None, "inlier_rmse": None,
        "position": None, "orientation": None, "transformation": None,
        "cad_points_transformed": None,
        **extra,
    }


# ==============================================================================
# ICP (numpy)
# ==============================================================================

def _icp_step(src, tgt, max_dist):
    """Один шаг ICP: ближайшие пары (двусторонняя проверка) → SVD (Кабш).
    Возвращает (T 4x4, rmse, fitness)."""
    if len(src) == 0 or len(tgt) == 0:
        return np.eye(4), float('inf'), 0.0

    dists_fwd, idx_fwd = KDTree(tgt).query(src, k=1)
    dists_bwd, _ = KDTree(src).query(tgt, k=1)

    inliers_fwd = dists_fwd < max_dist
    num_inliers = min(inliers_fwd.sum(), (dists_bwd < max_dist).sum())
    fitness = float(num_inliers) / max(len(src), len(tgt))
    rmse = float(np.sqrt((dists_fwd[inliers_fwd] ** 2).mean())) if inliers_fwd.any() else float('inf')

    src_matched = src[inliers_fwd]
    tgt_matched = tgt[idx_fwd[inliers_fwd]]
    if len(src_matched) < 6:
        return np.eye(4), float('inf'), 0.0

    src_c = src_matched - src_matched.mean(axis=0)
    tgt_c = tgt_matched - tgt_matched.mean(axis=0)
    U, _, Vt = np.linalg.svd(src_c.T @ tgt_c)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:            # защита от отражения
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = tgt_matched.mean(axis=0) - R @ src_matched.mean(axis=0)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T, rmse, fitness


def _icp_loop(src, tgt, max_correspondence_distance, max_iterations):
    """Накопительный ICP до сходимости rmse. Возвращает (T_total, fitness, rmse)."""
    T_total = np.eye(4)
    prev_rmse = float('inf')
    fitness, rmse = 0.0, float('inf')

    for _ in range(max_iterations):
        T_step, rmse, fitness = _icp_step(src, tgt, max_correspondence_distance)
        src = (T_step @ np.hstack([src, np.ones((len(src), 1))]).T).T[:, :3]
        T_total = T_total @ T_step
        if abs(prev_rmse - rmse) < 1e-6:
            break
        prev_rmse = rmse

    return T_total, fitness, rmse


# ==============================================================================
# FPFH + RANSAC (global registration, Open3D)
# ==============================================================================

def _preprocess_for_fpfh(pcd, voxel_size, normal_radius_factor=2.0,
                         fpfh_radius_factor=5.0, fpfh_max_nn=100):
    """Даунсэмпл + нормали + FPFH. Возвращает (down_pcd, fpfh)."""
    down = pcd.voxel_down_sample(voxel_size)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size * normal_radius_factor, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * fpfh_radius_factor, max_nn=fpfh_max_nn))
    return down, fpfh


def _train_fpfh_if_needed(cad_pcd, cad_name, voxel_size, cfg):
    """FPFH для CAD с кэшем (пересчёт при смене модели/содержимого/voxel_size)."""
    n_cad = len(cad_pcd.points)
    pts_sample = np.asarray(cad_pcd.points)[:10].tobytes() if n_cad > 0 else b""
    cad_id = f"{cad_name}_{n_cad}_{hash(pts_sample)}"

    if (_FPFH_CACHE["cad_id"] == cad_id and _FPFH_CACHE["voxel_size"] == voxel_size
            and _FPFH_CACHE["cad_down"] is not None):
        return _FPFH_CACHE["cad_down"], _FPFH_CACHE["cad_fpfh"]

    t0 = time.perf_counter()
    down, fpfh = _preprocess_for_fpfh(
        cad_pcd, voxel_size,
        normal_radius_factor=cfg.get("normal_radius_factor", 2.0),
        fpfh_radius_factor=cfg.get("fpfh_radius_factor", 5.0),
        fpfh_max_nn=cfg.get("fpfh_max_nn", 100))
    log.info(f"[FPFH] CAD {cad_name}: {len(down.points)} точек за {time.perf_counter()-t0:.2f}с")

    _FPFH_CACHE.update({"cad_id": cad_id, "voxel_size": voxel_size,
                        "cad_down": down, "cad_fpfh": fpfh})
    return down, fpfh


def _ransac_global_registration(source_down, source_fpfh, target_down, target_fpfh,
                                voxel_size, cfg):
    """RANSAC feature matching (source=CAD, target=сцена)."""
    dist = voxel_size * cfg.get("distance_threshold_factor", 1.5)
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=cfg.get("mutual_filter", True),
        max_correspondence_distance=dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=cfg.get("ransac_n", 3),
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                cfg.get("edge_length_threshold", 0.9)),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            cfg.get("max_iterations", 100000), cfg.get("confidence", 0.999)))


def _icp_refine(cluster, cad_model, initial_transform,
                voxel_size, max_correspondence_distance,
                max_iterations, fitness_threshold) -> dict:
    """ICP-уточнение от начальной позы RANSAC (без масштабирования CAD)."""
    pts_cad = np.asarray(cad_model.points, dtype=np.float64)
    pts_cad = (initial_transform @ np.hstack([pts_cad, np.ones((len(pts_cad), 1))]).T).T[:, :3]
    cluster_pts = np.asarray(cluster.points, dtype=np.float64)

    src = _ds_np(pts_cad, voxel_size)
    tgt = _ds_np(cluster_pts, voxel_size)
    if len(src) < 6 or len(tgt) < 6:
        return _failed_result(reason="too few points after global init")

    T_step, fitness, rmse = _icp_loop(src, tgt, max_correspondence_distance, max_iterations)
    if fitness < fitness_threshold:
        return _failed_result(reason=f"low fitness after global+ICP {fitness:.3f}",
                              icp_fitness=float(fitness))

    T_total = T_step @ initial_transform
    return {
        "status": "ok",
        "method": "icp",   # переопределяется вызывающим на fpfh+icp
        "fitness": float(fitness),
        "inlier_rmse": float(rmse),
        "transformation": T_total.tolist(),
        "position": T_total[:3, 3].tolist(),
        "orientation": rotation_to_quat(T_total[:3, :3]),
        "extent": [float(src.max(axis=0)[i] - src.min(axis=0)[i]) for i in range(3)],
        "cad_points_transformed": src,
    }


def run_global_then_icp(cluster, cad_model, cad_name, global_cfg, icp_cfg) -> dict:
    """FPFH+RANSAC (грубая поза) → ICP (уточнение). Единственный боевой путь."""
    voxel_size = global_cfg.get("voxel_size", 0.005)

    cad_down, cad_fpfh = _train_fpfh_if_needed(cad_model, cad_name, voxel_size, global_cfg)

    t0 = time.perf_counter()
    cluster_down, cluster_fpfh = _preprocess_for_fpfh(
        cluster, voxel_size,
        normal_radius_factor=global_cfg.get("normal_radius_factor", 2.0),
        fpfh_radius_factor=global_cfg.get("fpfh_radius_factor", 5.0),
        fpfh_max_nn=global_cfg.get("fpfh_max_nn", 100))

    if len(cluster_down.points) < 6:
        return _failed_result(reason=f"too few scene points: {len(cluster_down.points)}")

    result = _ransac_global_registration(
        cad_down, cad_fpfh, cluster_down, cluster_fpfh, voxel_size, global_cfg)
    # Данные из RegistrationResult извлекаем СРАЗУ в numpy/py-типы и удаляем объект:
    # обращение к нему после следующей PointCloud-операции роняет процесс на Jetson.
    global_fitness = float(result.fitness)
    global_rmse = float(result.inlier_rmse)
    initial_T = np.array(result.transformation, dtype=np.float64, copy=True)
    del result

    log.info(f"[FPFH+RANSAC] fitness={global_fitness:.3f} rmse={global_rmse:.4f} "
             f"за {time.perf_counter()-t0:.2f}с")

    if global_fitness < global_cfg.get("min_fitness", 0.1):
        return _failed_result(reason=f"low global fitness {global_fitness:.3f}",
                              global_fitness=global_fitness)

    refined = _icp_refine(
        cluster, cad_model, initial_T,
        voxel_size=icp_cfg["voxel_size"],
        max_correspondence_distance=icp_cfg["max_correspondence_distance"],
        max_iterations=icp_cfg["max_iterations"],
        fitness_threshold=icp_cfg["fitness_threshold"])

    if refined.get("method") == "icp":
        refined["method"] = "fpfh+icp"
    refined["global_fitness"] = global_fitness
    refined["global_rmse"] = global_rmse
    return refined