import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs

# ==================== НАСТРОЙКИ ====================
MARKER_SIZE = 0.080          # <-- Реальный размер чёрного квадрата маркера в МЕТРАХ (100 мм)

# === КООРДИНАТЫ УГЛОВ МАРКЕРОВ В СИСТЕМЕ СТОЛА ===
# Порядок углов: верх-лево, верх-право, низ-право, низ-лево
# Маркер 0 стоит в начале координат. Маркер 1 правее (задаёт +X)

marker_corners_3d = {
    0: np.array([
        [0.000, MARKER_SIZE, 0.000],
        [MARKER_SIZE, MARKER_SIZE, 0.000],
        [MARKER_SIZE, 0, 0.000],
        [0.000, 0.000, 0.000]
    ], dtype=np.float32),   

    1: np.array([
        [0.640, MARKER_SIZE, 0.000],                    # 25 см правее маркера 0
        [0.640 + MARKER_SIZE, MARKER_SIZE, 0.000],
        [0.640 + MARKER_SIZE, 0.000, 0.000],
        [0.640, 0.000, 0.000]
    ], dtype=np.float32),

    2: np.array([
        [0.640, 0.400, 0.000],                    # выше маркера 1
        [0.640 + MARKER_SIZE, 0.400, 0.000],
        [0.640 + MARKER_SIZE, 0.400 - MARKER_SIZE, 0.000],
        [0.640, 0.400 - MARKER_SIZE, 0.000]
    ], dtype=np.float32),

    3: np.array([
        [0.000, 0.400, 0.000],                    # выше маркера 0
        [MARKER_SIZE, 0.400, 0.000],
        [MARKER_SIZE, 0.400 - MARKER_SIZE, 0.000],
        [0.000, 0.400 - MARKER_SIZE, 0.000]
    ], dtype=np.float32),
}

def get_realsense_intrinsics():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    profile = pipeline.get_active_profile()
    color_profile = profile.get_stream(rs.stream.color)
    intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

    camera_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float32)
    pipeline.stop()
    return camera_matrix, dist_coeffs

def calibrate_camera_to_table():
    camera_matrix, dist_coeffs = get_realsense_intrinsics()
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    print("=== КАЛИБРОВКА КАМЕРА → СТОЛ ===")
    print("Нажми 's' — сохранить матрицу, 'q' — выход")

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            object_points = []
            image_points = []

            for i, marker_id in enumerate(ids.flatten()):
                if marker_id not in marker_corners_3d:
                    continue
                object_points.extend(marker_corners_3d[marker_id])
                image_points.extend(corners[i][0])

            if len(object_points) >= 4:
                object_points = np.array(object_points, dtype=np.float32)
                image_points = np.array(image_points, dtype=np.float32)

                success, rvec, tvec = cv2.solvePnP(
                    object_points, image_points, camera_matrix, dist_coeffs
                )

                if success:
                    R, _ = cv2.Rodrigues(rvec)
                    T = np.eye(4, dtype=np.float32)
                    T[:3, :3] = R
                    T[:3, 3] = tvec.flatten()

                    cv2.putText(frame, "Pose OK - press 's' to save", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    key = cv2.waitKey(1)
                    if key == ord('s'):
                        np.save("T_cam_to_table.npy", T)
                        print("\nМатрица сохранена в T_cam_to_table.npy")
                        print(T)
                    if key == ord('q'):
                        break

        cv2.imshow("ArUco Calibration (RealSense D415)", frame)

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrate_camera_to_table()