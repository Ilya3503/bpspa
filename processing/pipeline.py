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


def load_cam_to_table(path: Path = _CAM_TO_TABLE_PATH) -> Optional[np.ndarray]:
    """
    Грузит 4x4 матрицу cam->table из .npy. Возвращает float64 (4,4) или None,
    если файл отсутствует/некорректен (тогда трансформация не применяется).
    """
    if not path.exists():
        log.warning(f"[cam->table] матрица не найдена: {path}. Координаты останутся в системе камеры.")
        return None
    try:
        T = np.load(str(path)).astype(np.float64)
    except Exception as e:
        log.error(f"[cam->table] не удалось загрузить {path}: {e}. Координаты останутся в системе камеры.")
        return None
    if T.shape != (4, 4):
        log.error(f"[cam->table] матрица не 4x4: {T.shape}. Координаты останутся в системе камеры.")
        return None
    return T


def _apply_cam_to_table_to_pose(pose: dict, T_table: np.ndarray) -> None:
    """
    Трансформирует одну позу (in-place) из координат камеры в координаты стола.

    ПОЗИЦИЯ считается всегда как умножение ВЕКТОРА position на матрицу:
        p_table = T_table @ [px, py, pz, 1]
    Это надёжно и не зависит от того, согласована ли трансляция внутри
    transformation с фактическим центром объекта (она бывает рассогласована —
    тогда умножение матрицы давало бы позицию камеры, а не объекта).

    ОРИЕНТАЦИЯ берётся из повёрнутой матрицы R_table @ R_pose, когда есть
    transformation; поворотная часть надёжна.
    """
    R_table = T_table[:3, :3]

    # 1) Позиция — всегда из вектора position.
    if pose.get("position") is not None:
        p = np.array(pose["position"], dtype=np.float64)
        pose["position"] = (T_table @ np.append(p, 1.0))[:3].tolist()

    # 2) Ориентация и transformation — если есть полная матрица.
    if pose.get("transformation") is not None:
        T_pose = np.array(pose["transformation"], dtype=np.float64)
        R_new = R_table @ T_pose[:3, :3]
        pose["orientation"] = rotation_to_quat(R_new)
        # transformation пересобираем согласованно: поворот R_new + уже посчитанная позиция.
        T_new = np.eye(4)
        T_new[:3, :3] = R_new
        T_new[:3, 3] = np.array(pose["position"], dtype=np.float64)
        pose["transformation"] = T_new.tolist()


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
# MERGE двух облаков
# ==============================================================================

def load_calibration_matrix(path: str) -> Tuple[np.ndarray, bool]:
    p = Path(path)
    if not p.exists():
        log.warning(f"Калибровочный файл не найден: {p}. Использую единичную матрицу.")
        return np.eye(4), True
    T = np.load(str(p))
    if T.shape != (4, 4):
        log.warning(f"Калибровочная матрица не 4x4: {T.shape}. Заменяю на единичную.")
        return np.eye(4), True
    return T, bool(np.allclose(T, np.eye(4)))


def merge_two_clouds(file_a: str, file_b: str, T_b_to_a: np.ndarray,
                     output_dir: str = "data",
                     voxel_size: float = 0.005) -> str:
    pcd_a = load_pcd(file_a)
    pcd_b = load_pcd(file_b)

    if voxel_size > 0:
        pcd_a = pcd_a.voxel_down_sample(voxel_size)
        pcd_b = pcd_b.voxel_down_sample(voxel_size)

    pts_b = np.asarray(pcd_b.points)
    pts_b_h = np.hstack([pts_b, np.ones((pts_b.shape[0], 1))])
    pts_b_transformed = (T_b_to_a @ pts_b_h.T).T[:, :3]

    pts_a = np.asarray(pcd_a.points)
    pts_merged = np.vstack([pts_a, pts_b_transformed])

    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(pts_merged)

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
        if cluster_pts[:, 2].max() - cluster_pts[:, 2].min() < 0.003:
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
# КВАТЕРНИОНЫ
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


def _ds_np(pts: np.ndarray, vsize: float) -> np.ndarray:

    if vsize <= 0 or len(pts) == 0:
        return pts
    # квантование в индексы вокселя
    voxel_indices = np.floor(pts / vsize).astype(np.int64)
    # уникальные воксели и индексы первого вхождения каждой точки
    _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
    return pts[unique_idx]


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
        T_total = T_step @ T_total
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


def run_iterative_global(scene_pcd: o3d.geometry.PointCloud,
                         cad_model: o3d.geometry.PointCloud,
                         cad_name: str,
                         global_cfg: dict,
                         icp_cfg: dict,
                         iter_cfg: dict) -> List[dict]:
    """
    Iterative FPFH+RANSAC: ищет несколько экземпляров CAD во всём облаке сцены.
    После каждой найденной позы удаляет inlier-точки и запускает поиск снова.
    Возвращает список поз (тот же формат что run_global_then_icp).
    Без DBSCAN — работает прямо на облаке сцены после плоскости.
    """
    max_instances = int(iter_cfg.get("max_instances", 10))
    min_fitness = float(iter_cfg.get("min_fitness", 0.15))
    min_remaining_points = int(iter_cfg.get("min_remaining_points", 200))
    inlier_radius = float(iter_cfg.get("inlier_radius", 0.005))

    voxel_size = global_cfg.get("voxel_size", 0.005)

    # FPFH для CAD один раз
    cad_down, cad_fpfh = _train_fpfh_if_needed(cad_model, cad_name, voxel_size, global_cfg)

    remaining_pts = np.asarray(scene_pcd.points, dtype=np.float64).copy()
    poses: List[dict] = []

    for it in range(max_instances):
        if len(remaining_pts) < min_remaining_points:
            log.info(f"[iter-RANSAC] осталось {len(remaining_pts)} точек < порога — стоп")
            break

        # собираем временный PointCloud из оставшихся точек
        scene_tmp = o3d.geometry.PointCloud()
        scene_tmp.points = o3d.utility.Vector3dVector(remaining_pts)

        scene_down, scene_fpfh = _preprocess_for_fpfh(
            scene_tmp, voxel_size,
            normal_radius_factor=global_cfg.get("normal_radius_factor", 2.0),
            fpfh_radius_factor=global_cfg.get("fpfh_radius_factor", 5.0),
            fpfh_max_nn=global_cfg.get("fpfh_max_nn", 100),
        )
        if len(scene_down.points) < 6:
            log.info(f"[iter-RANSAC] after downsample слишком мало точек — стоп")
            break

        result = _ransac_global_registration(
            cad_down, cad_fpfh, scene_down, scene_fpfh,
            voxel_size, global_cfg,
        )
        global_fitness = float(result.fitness)
        global_rmse = float(result.inlier_rmse)
        initial_T = np.array(result.transformation, dtype=np.float64, copy=True)
        del result

        log.info(f"[iter-RANSAC] iter {it}: fitness={global_fitness:.3f} rmse={global_rmse:.4f} "
                 f"on {len(remaining_pts)} pts")

        if global_fitness < min_fitness:
            log.info(f"[iter-RANSAC] fitness {global_fitness:.3f} < {min_fitness} — стоп")
            break

        # ICP refine на «полной» сцене для точности позы
        tmp_pcd_for_icp = o3d.geometry.PointCloud()
        tmp_pcd_for_icp.points = o3d.utility.Vector3dVector(remaining_pts)
        refined = _icp_with_initial_transform(
            tmp_pcd_for_icp, cad_model,
            initial_transform=initial_T,
            voxel_size=icp_cfg["voxel_size"],
            max_correspondence_distance=icp_cfg["max_correspondence_distance"],
            max_iterations=icp_cfg["max_iterations"],
            fitness_threshold=icp_cfg["fitness_threshold"],
        )

        # Если ICP не сошёлся (fallback в OBB) — используем позу от RANSAC как есть.
        # Это нормально для iterative-режима: RANSAC уже нашёл достоверное совпадение.
        if refined.get("method") == "obb_fallback" or "transformation" not in refined:
            log.info(f"[iter-RANSAC] iter {it}: ICP refine не прошёл, использую позу от RANSAC")
            R_init = initial_T[:3, :3]
            t_init = initial_T[:3, 3]
            cad_pts = np.asarray(cad_model.points, dtype=np.float64)
            cad_h = np.hstack([cad_pts, np.ones((len(cad_pts), 1))])
            cad_transformed = (initial_T @ cad_h.T).T[:, :3]
            refined = {
                "method": "fpfh_only",
                "fitness": global_fitness,
                "inlier_rmse": global_rmse,
                "transformation": initial_T.tolist(),
                "position": t_init.tolist(),
                "orientation": rotation_to_quat(R_init),
                "extent": [float(cad_transformed.max(axis=0)[i] - cad_transformed.min(axis=0)[i]) for i in range(3)],
                "cad_points_transformed": cad_transformed,
            }
        else:
            if refined.get("method") == "icp":
                refined["method"] = "fpfh+icp"

        refined["global_fitness"] = global_fitness
        refined["global_rmse"] = global_rmse
        refined["iteration"] = it
        poses.append(refined)

        # удаляем точки сцены, объяснённые этой позой (всё в numpy — без падений Open3D)
        T = np.array(refined["transformation"], dtype=np.float64)
        cad_pts = np.asarray(cad_model.points, dtype=np.float64)
        cad_pts_h = np.hstack([cad_pts, np.ones((len(cad_pts), 1))])
        cad_transformed = (T @ cad_pts_h.T).T[:, :3]

        tree = KDTree(cad_transformed)
        dists, _ = tree.query(remaining_pts, k=1)
        keep_mask = dists > inlier_radius
        removed = int((~keep_mask).sum())
        remaining_pts = remaining_pts[keep_mask]
        log.info(f"[iter-RANSAC] удалено {removed} точек, осталось {len(remaining_pts)}")

        if removed < 10:
            # ничего не удалили — нет смысла продолжать
            log.info("[iter-RANSAC] нечего удалять — стоп")
            break

    log.info(f"[iter-RANSAC] найдено экземпляров: {len(poses)}")
    return poses


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

def make_iterative_annotated_ply(scene_pcd, poses, cad_model, results_dir: str) -> str:
    """
    Визуализация для iterative-режима: исходная сцена серым + CAD-модели
    по найденным позам разными цветами.
    """
    out = Path(results_dir) / "annotated_pointcloud.ply"
    all_p, all_c = [], []

    # Сцена серым
    scene_pts = np.asarray(scene_pcd.points)
    if len(scene_pts) > 0:
        all_p.append(scene_pts)
        all_c.append(np.tile([0.4, 0.4, 0.4], (len(scene_pts), 1)))

    # CAD по найденным позам — каждая своим цветом
    cmap = plt.get_cmap("tab10")(np.linspace(0, 1, 10))[:, :3]
    cad_pts_orig = np.asarray(cad_model.points)

    for i, pose in enumerate(poses):
        T = np.array(pose["transformation"], dtype=np.float64)
        cad_h = np.hstack([cad_pts_orig, np.ones((len(cad_pts_orig), 1))])
        cad_transformed = (T @ cad_h.T).T[:, :3]
        clr = np.tile(cmap[i % len(cmap)], (len(cad_transformed), 1))
        all_p.append(cad_transformed)
        all_c.append(clr)

    if not all_p:
        return str(out)

    m = o3d.geometry.PointCloud()
    m.points = o3d.utility.Vector3dVector(np.vstack(all_p))
    m.colors = o3d.utility.Vector3dVector(np.vstack(all_c))
    o3d.io.write_point_cloud(str(out), m)
    return str(out)

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
