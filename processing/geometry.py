"""
Общая математика: кватернионы, воксельный даунсемпл на numpy,
загрузка и применение матрицы cam->table. Без o3d-пайплайнов и без состояния —
только чистые преобразования. Импортируется слоями pose / io_results.
"""
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)

_CAM_TO_TABLE_PATH = Path("hardware") / "transform_cam_to_world.npy"


def _ds_np(pts: np.ndarray, vsize: float) -> np.ndarray:
    """Воксельный даунсемпл на numpy: одна точка на воксель (первое вхождение)."""
    if vsize <= 0 or len(pts) == 0:
        return pts
    voxel_indices = np.floor(pts / vsize).astype(np.int64)
    _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
    return pts[unique_idx]


def rotation_to_quat(R: np.ndarray) -> List[float]:
    """Матрица поворота 3x3 → кватернион [x, y, z, w] (алгоритм Шепперда)."""
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


def load_cam_to_table(path: Path = _CAM_TO_TABLE_PATH) -> Optional[np.ndarray]:
    """Грузит 4x4 матрицу cam->table из .npy. None, если файла нет/битый —
    тогда координаты остаются в системе камеры."""
    if not path.exists():
        log.warning(f"[cam->table] матрица не найдена: {path}. Координаты в системе камеры.")
        return None
    try:
        T = np.load(str(path)).astype(np.float64)
    except Exception as e:
        log.error(f"[cam->table] не загрузилась {path}: {e}. Координаты в системе камеры.")
        return None
    if T.shape != (4, 4):
        log.error(f"[cam->table] матрица не 4x4: {T.shape}. Координаты в системе камеры.")
        return None
    return T


def _apply_cam_to_table_to_pose(pose: dict, T_table: np.ndarray) -> None:
    """Переводит позу (in-place) из координат камеры в координаты стола.

    Позиция — всегда умножением вектора position на T (надёжно, не зависит
    от согласованности трансляции внутри transformation). Ориентация —
    из R_table @ R_pose, когда есть полная матрица."""
    R_table = T_table[:3, :3]

    if pose.get("position") is not None:
        p = np.array(pose["position"], dtype=np.float64)
        pose["position"] = (T_table @ np.append(p, 1.0))[:3].tolist()

    if pose.get("transformation") is not None:
        T_pose = np.array(pose["transformation"], dtype=np.float64)
        R_new = R_table @ T_pose[:3, :3]
        pose["orientation"] = rotation_to_quat(R_new)
        T_new = np.eye(4)
        T_new[:3, :3] = R_new
        T_new[:3, 3] = np.array(pose["position"], dtype=np.float64)
        pose["transformation"] = T_new.tolist()