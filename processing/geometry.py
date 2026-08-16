def _ds_np(pts: np.ndarray, vsize: float) -> np.ndarray:

    if vsize <= 0 or len(pts) == 0:
        return pts
    # квантование в индексы вокселя
    voxel_indices = np.floor(pts / vsize).astype(np.int64)
    # уникальные воксели и индексы первого вхождения каждой точки
    _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
    return pts[unique_idx]



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
