from typing import List, Optional, Tuple
import logging
import time

import numpy as np
from scipy.spatial import KDTree
import cv2

log = logging.getLogger(__name__)


# ==============================================================================
# НОРМАЛИ + КРИВИЗНА (PCA на k ближайших соседях)
# ==============================================================================

def estimate_normals_and_curvature_np(pts: np.ndarray,
                                       k: int = 30,
                                       viewpoint: Tuple[float, float, float] = (0.0, 0.0, 0.0)
                                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Считает нормали и кривизну для каждой точки через PCA на k ближайших соседях.
    Нормали ориентируются в сторону viewpoint (по умолчанию — оптический центр камеры).

    Возвращает: (normals shape (N,3), curvature shape (N,))
    """
    n = len(pts)
    if n < k + 1:
        # слишком мало точек — невозможно посчитать осмысленно
        return np.zeros((n, 3)), np.zeros(n)

    tree = KDTree(pts)
    # +1 потому что сама точка тоже включена в результат
    _, idx = tree.query(pts, k=k + 1)

    # Векторизованная PCA: собираем все neighbourhoods в (N, k+1, 3)
    nbh = pts[idx]                              # (N, k+1, 3)
    centered = nbh - nbh.mean(axis=1, keepdims=True)   # центрируем
    # Ковариация на батче: (N, 3, 3)
    cov = np.einsum('nki,nkj->nij', centered, centered) / k

    # eigh быстрее eig и возвращает отсортированные eigenvalues по возрастанию
    eigvals, eigvecs = np.linalg.eigh(cov)      # eigvals: (N,3), eigvecs: (N,3,3)

    # Нормаль = вектор минимального собственного значения (eigvals[:,0])
    normals = eigvecs[:, :, 0]                  # (N, 3)
    # Кривизна = lambda_min / sum(lambdas)
    sum_eig = eigvals.sum(axis=1)
    curvature = np.where(sum_eig > 1e-12, eigvals[:, 0] / sum_eig, 0.0)

    # Ориентируем нормали в сторону viewpoint
    vp = np.array(viewpoint, dtype=np.float64)
    to_vp = vp - pts                            # (N, 3)
    flip_mask = np.einsum('ni,ni->n', normals, to_vp) < 0
    normals[flip_mask] = -normals[flip_mask]

    return normals, curvature


# ==============================================================================
# REGION GROWING на нормалях
# ==============================================================================

def region_growing_planar(pts: np.ndarray,
                          normals: np.ndarray,
                          curvature: np.ndarray,
                          k_neighbors: int = 30,
                          theta_threshold_deg: float = 15.0,
                          curvature_threshold: float = 0.05,
                          min_region_points: int = 200,
                          max_regions: int = 20,
                          max_edge_dist: float = 0.005,
                          ) -> List[np.ndarray]:
    """
    Region growing по нормалям (Smoothness Constraint).

    Принимает pts, нормали, кривизны (уже посчитанные).
    Возвращает список массивов индексов точек, отсортированный по размеру убывающе.
    Регионы меньше min_region_points отбрасываются.

    Используется while-цикл по seeds (НЕ for, иначе seeds, добавленные после
    старта цикла, не итерируются — известный баг StackOverflow-реализации).
    """
    n = len(pts)
    if n < k_neighbors + 1:
        return []

    tree = KDTree(pts)
    _, neighbours = tree.query(pts, k=k_neighbors + 1)

    theta_rad = np.deg2rad(theta_threshold_deg)
    cos_theta_min = np.cos(theta_rad)

    available = np.ones(n, dtype=bool)
    order = np.argsort(curvature)               # точки от самой плоской к самой кривой

    regions: List[np.ndarray] = []
    for seed_start in order:
        if not available[seed_start]:
            continue
        if len(regions) >= max_regions:
            break

        region: List[int] = [int(seed_start)]
        seeds: List[int] = [int(seed_start)]
        available[seed_start] = False

        # while-цикл по индексу — корректно обрабатывает рост seeds во время прохода
        i = 0
        while i < len(seeds):
            sp = seeds[i]
            sp_normal = normals[sp]
            for nb in neighbours[sp]:
                if not available[nb]:
                    continue
                # Евклидов разрыв: сосед из KDTree может физически лежать на другом
                # объекте (копланарная грань соседнего куба). Если он дальше порога —
                # не растём в него, даже если нормали параллельны.
                if np.linalg.norm(pts[sp] - pts[nb]) > max_edge_dist:
                    continue
                # |cos(angle)| — учитываем что нормаль может быть ориентирована
                # в любую сторону; нас интересует только параллельность поверхностей
                cos_a = abs(sp_normal @ normals[nb])
                if cos_a < cos_theta_min:
                    continue
                # точка проходит smoothness — добавляем в регион
                region.append(int(nb))
                available[nb] = False
                # если её кривизна низкая — она тоже становится seed
                if curvature[nb] < curvature_threshold:
                    seeds.append(int(nb))
            i += 1

        if len(region) >= min_region_points:
            regions.append(np.asarray(region, dtype=np.int64))

    regions.sort(key=len, reverse=True)
    return regions


# ==============================================================================
# ВОССТАНОВЛЕНИЕ ПОЗЫ КУБА ИЗ ПЛОСКОГО РЕГИОНА
# ==============================================================================

def _quat_from_R(R: np.ndarray) -> List[float]:
    """Кватернион [x,y,z,w] из 3x3 матрицы. Дубль из pipeline.rotation_to_quat,
    чтобы не плодить зависимости между файлами."""
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


def pose_from_planar_region_cube(region_pts: np.ndarray,
                                  cube_size: float,
                                  squareness_tolerance: float = 0.3,
                                  size_tolerance: float = 0.3,
                                  ) -> Optional[dict]:
    """
    Восстанавливает позу куба по плоскому региону (предположительно — верхней грани).

    Параметры:
        region_pts: точки региона (N,3) в координатах камеры
        cube_size:  длина ребра куба, в метрах (из CAD extent)
        squareness_tolerance:  макс относительная разница длины/ширины minAreaRect
                               (1.0 = квадрат; >0.3 → не квадратная грань, отказ)
        size_tolerance: макс относительное отклонение реального размера от cube_size

    Возвращает dict в том же формате что run_global_then_icp:
        method, fitness (None), inlier_rmse (None), confidence (0..1),
        transformation (4x4), position, orientation (quat), extent,
        cad_points_transformed (8 вершин куба для визуализации)
    Или None если регион не похож на грань куба.
    """
    if len(region_pts) < 10:
        return None

    centroid = region_pts.mean(axis=0)
    centered = region_pts - centroid

    # PCA на точках региона
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # Главные оси в порядке убывания дисперсии:
    # Vt[0] — самая большая (длинная сторона грани)
    # Vt[1] — средняя (короткая сторона грани)
    # Vt[2] — самая малая (нормаль к плоскости)
    axis_long = Vt[0]
    axis_short = Vt[1]
    normal = Vt[2]

    # Нормаль ориентируем к камере (viewpoint = (0,0,0))
    if normal @ (-centroid) < 0:
        normal = -normal
        axis_short = -axis_short  # сохраняем правую тройку

    # Проектируем точки в 2D-плоскость грани (basis: axis_long, axis_short)
    coords_2d = np.column_stack([
        centered @ axis_long,
        centered @ axis_short,
    ]).astype(np.float32)

    # minAreaRect — повёрнутый прямоугольник минимальной площади
    # rect: ((cx, cy), (w, h), angle_degrees)
    rect = cv2.minAreaRect(coords_2d)
    (cx_2d, cy_2d), (w, h), angle_deg = rect

    # Проверка: грань должна быть примерно квадратной
    if max(w, h) < 1e-6:
        return None
    squareness = abs(w - h) / max(w, h)
    if squareness > squareness_tolerance:
        log.info(f"[planar] регион не квадратный (w={w:.4f}, h={h:.4f}, "
                 f"diff={squareness:.2f}) — отказ")
        return None

    # Проверка: размер должен соответствовать кубу
    mean_side = (w + h) / 2.0
    size_err = abs(mean_side - cube_size) / cube_size
    if size_err > size_tolerance:
        log.info(f"[planar] размер региона {mean_side:.4f} м не совпадает с CAD "
                 f"{cube_size:.4f} м (отклонение {size_err:.2f}) — отказ")
        return None

    # Собираем 3D-оси куба:
    #   Z = -normal (наружу из верхней грани НАРУЖУ, как ось грипера)
    #   Длинная сторона грани в 3D = поворот axis_long на angle_deg в плоскости
    angle_rad = np.deg2rad(angle_deg)
    x_axis = axis_long * np.cos(angle_rad) + axis_short * np.sin(angle_rad)
    x_axis /= np.linalg.norm(x_axis) + 1e-12

    z_axis = -normal  # «вверх» относительно грани (то, куда должен подходить грипер)
    z_axis /= np.linalg.norm(z_axis) + 1e-12

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-12
    # корректируем x чтобы тройка была идеально ортогональной
    x_axis = np.cross(y_axis, z_axis)

    R = np.column_stack([x_axis, y_axis, z_axis])

    # Центр грани в 3D = центроид региона + смещение по 2D-центру rect относительно (0,0)
    face_center = centroid + cx_2d * axis_long + cy_2d * axis_short
    # Центр куба = центр грани - z_axis * (cube_size / 2)
    # (z_axis смотрит наружу, поэтому центр куба сдвинут «внутрь» сцены)
    cube_center = face_center - z_axis * (cube_size / 2.0)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cube_center

    # 8 вершин куба для визуализации
    s = cube_size / 2.0
    corners_local = np.array([
        [-s, -s, -s], [+s, -s, -s], [+s, +s, -s], [-s, +s, -s],
        [-s, -s, +s], [+s, -s, +s], [+s, +s, +s], [-s, +s, +s],
    ], dtype=np.float64)
    corners_world = (R @ corners_local.T).T + cube_center

    # confidence: для planar fallback фиксируем низкое значение,
    # так как 6-DoF поза восстановлена из одной грани (4 эквивалентные ориентации)
    confidence = 0.3

    return {
        "method": "planar_fallback",
        "fitness": None,
        "inlier_rmse": None,
        "confidence": confidence,
        "transformation": T.tolist(),
        "position": cube_center.tolist(),
        "orientation": _quat_from_R(R),
        "extent": [cube_size, cube_size, cube_size],
        "region_points_count": int(len(region_pts)),
        "region_squareness": float(squareness),
        "region_size_error": float(size_err),
        "cad_points_transformed": corners_world,
    }


# ==============================================================================
# ВЫСОКОУРОВНЕВАЯ ОБЁРТКА
# ==============================================================================

def run_planar_fallback(pts: np.ndarray,
                        cad_model_pts: np.ndarray,
                        cfg: dict) -> Optional[dict]:
    """
    Запускает Region Growing на pts, выбирает самый большой плоский регион,
    восстанавливает позу куба.

    Параметры:
        pts:           Nx3 точки сцены (после remove_plane!)
        cad_model_pts: Mx3 точки CAD модели (нужны только для extent)
        cfg:           словарь planar_fallback из config.yaml

    Возвращает один dict-позу (как run_global_then_icp) или None.
    """
    if len(pts) < cfg.get("min_region_points", 200):
        log.info(f"[planar] слишком мало точек ({len(pts)}) — пропуск")
        return None

    # Размер куба берём строго из CAD extent (договорённость)
    cad_extent = cad_model_pts.max(axis=0) - cad_model_pts.min(axis=0)
    cube_size = float(cad_extent.mean())
    # sanity: CAD должен быть в метрах
    if cube_size > 1.0 or cube_size < 0.005:
        log.error(f"[planar] размер куба из CAD = {cube_size}: CAD должна быть в метрах "
                  f"в диапазоне 5мм..1м. Прерываем fallback.")
        return None

    t0 = time.perf_counter()
    normals, curvature = estimate_normals_and_curvature_np(
        pts, k=int(cfg.get("k_neighbors", 30))
    )
    log.info(f"[planar] нормали+кривизна для {len(pts)} точек за {time.perf_counter()-t0:.2f}с")

    t1 = time.perf_counter()
    regions = region_growing_planar(
        pts, normals, curvature,
        k_neighbors=int(cfg.get("k_neighbors", 30)),
        theta_threshold_deg=float(cfg.get("theta_threshold_deg", 15.0)),
        curvature_threshold=float(cfg.get("curvature_threshold", 0.05)),
        min_region_points=int(cfg.get("min_region_points", 200)),
        max_regions=int(cfg.get("max_regions", 20)),
        max_edge_dist=float(cfg.get("max_edge_dist", 0.005)),
    )
    log.info(f"[planar] region growing: {len(regions)} регионов за {time.perf_counter()-t1:.2f}с")

    if not regions:
        return None

    # Пробуем регионы по убыванию размера — берём первый годный для куба
    sq_tol = float(cfg.get("squareness_tolerance", 0.30))
    sz_tol = float(cfg.get("size_tolerance", 0.30))
    for ri, region_idx in enumerate(regions):
        region_pts = pts[region_idx]
        pose = pose_from_planar_region_cube(
            region_pts, cube_size,
            squareness_tolerance=sq_tol,
            size_tolerance=sz_tol,
        )
        if pose is not None:
            log.info(f"[planar] поза из региона #{ri} ({len(region_idx)} точек)")
            return pose

    log.info("[planar] ни один регион не подошёл под куб")
    return None