from pathlib import Path
from datetime import datetime
import json

import cv2
import numpy as np
import pyrealsense2 as rs

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "calibration"

# Измеренные координаты ЦЕНТРОВ маркеров в метрах.
# +X вправо, +Y к верхней стороне изображения стола, +Z над столом.
MARKERS_TABLE_M = {
    0: (-0.260, -0.160, 0.0),  # левый нижний
    1: (-0.260, +0.160, 0.0),  # левый верхний
    2: (+0.260, +0.160, 0.0),  # правый верхний
    3: (+0.260, -0.160, 0.0),  # правый нижний
}
WIDTH, HEIGHT, FPS = 1280, 720, 6
REPROJECTION_MAX_PX = 3.0
MIN_CAMERA_HEIGHT_M = 0.20
MAX_CAMERA_HEIGHT_M = 2.00
SAMPLES = 20


def aruco_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    raise RuntimeError("Требуется OpenCV с cv2.aruco.ArucoDetector")


def get_intrinsics(profile):
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]], dtype=np.float64)
    distortion = np.array(intr.coeffs, dtype=np.float64)
    return K, distortion


def estimate(image, detector, K, distortion):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None

    detected = {}
    for quadrilateral, marker_id in zip(corners, ids.flatten()):
        marker_id = int(marker_id)
        if marker_id in MARKERS_TABLE_M:
            detected[marker_id] = np.mean(quadrilateral.reshape(4, 2), axis=0)
    if len(detected) != 4:
        return None

    order = sorted(MARKERS_TABLE_M)
    object_points = np.array([MARKERS_TABLE_M[i] for i in order], dtype=np.float64)
    image_points = np.array([detected[i] for i in order], dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, K, distortion,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return None

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, distortion)
    residual = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    rmse = float(np.sqrt(np.mean(residual ** 2)))

    # solvePnP возвращает преобразование стола в камеру.
    R_table_to_cam, _ = cv2.Rodrigues(rvec)
    R_cam_to_table = R_table_to_cam.T
    t_cam_to_table = (-R_cam_to_table @ tvec).ravel()

    # Проверяем, что плоскость перед камерой, а камера над столом.
    table_points_cam = (R_table_to_cam @ object_points.T + tvec).T
    if np.any(table_points_cam[:, 2] <= 0):
        return None
    if not (MIN_CAMERA_HEIGHT_M <= t_cam_to_table[2] <= MAX_CAMERA_HEIGHT_M):
        return None
    if rmse > REPROJECTION_MAX_PX:
        return None

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_cam_to_table
    T[:3, 3] = t_cam_to_table
    return T, rmse, residual

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    detector = aruco_detector()
    print("Ожидание 4 маркеров (ID: 0, 1, 2, 3)...")
    profile = None
    try:
        profile = pipeline.start(config)
        K, distortion = get_intrinsics(profile)
        print(f"RGB intrinsics: fx={K[0,0]:.2f}, fy={K[1,1]:.2f} px")
        samples = []
        frames_seen = 0
        while len(samples) < SAMPLES and frames_seen < 600:
            frames_seen += 1
            color_frame = pipeline.wait_for_frames().get_color_frame()
            if not color_frame:
                continue
            image = np.asanyarray(color_frame.get_data())
            result = estimate(image, detector, K, distortion)
            if result is not None:
                samples.append(result)
                print(f"Принято кадров: {len(samples)}/{SAMPLES}; RMSE={result[1]:.2f} px", end="\r", flush=True)
        print()
        if len(samples) != SAMPLES:
            raise RuntimeError("Недостаточно корректных кадров. Проверьте видимость, ID и геометрию маркеров.")

        # Используем кадр с минимальной ошибкой репроекции.
        # Не усредняем элементы матриц поворота непосредственно.
        best_T, best_rmse, best_residual = min(samples, key=lambda x: x[1])
        print("Калибровка рассчитана, но ещё не подтверждена физическими измерениями.")
        print("T_camera_to_table [м]:\n", best_T)
        print("Положение камеры в СК стола [мм]:", np.round(best_T[:3, 3] * 1000, 2))
        print("RMSE [px]:", round(best_rmse, 3))
        print("Отклонения четырёх центров [px]:", np.round(best_residual, 3))

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUT_DIR / "transform_cam_to_world.npy"
        np.save(output_path, best_T)
        np.save(OUT_DIR / "camera_matrix.npy", K)
        np.save(OUT_DIR / "dist_coeffs.npy", distortion)
        report = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "units": "meters",
            "coordinate_frame": "table_origin_at_marker_rectangle_center",
            "camera_frame": "RealSense_COLOR_optical_frame",
            "frame_count": len(samples),
            "reprojection_rmse_px": best_rmse,
            "marker_center_residuals_px": best_residual.tolist(),
            "camera_position_table_m": best_T[:3, 3].tolist(),
        }
        (OUT_DIR / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Сохранено:", output_path)
        print("Рабочая матрица hardware/transform_cam_to_world.npy НЕ изменена.")
    finally:
        if profile is not None:
            pipeline.stop()


if __name__ == "__main__":
    main()