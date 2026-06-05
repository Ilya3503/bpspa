import cv2
import numpy as np
import pyrealsense2 as rs
import os
from datetime import datetime

def rotation_matrix_to_euler(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        roll  = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll  = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = 0
    return np.degrees([roll, pitch, yaw])


def main():
    # ==================== НАСТРОЙКИ ====================
    marker_ids = [0, 1, 2, 3]
    dist_x = 520.0   # мм
    dist_y = 320.0   # мм

    half_x = dist_x / 2
    half_y = dist_y / 2

    object_points = np.array([
        [-half_x, -half_y, 0.0],
        [ half_x, -half_y, 0.0],
        [ half_x,  half_y, 0.0],
        [-half_x,  half_y, 0.0]
    ], dtype=np.float32)

    # ==================== RealSense ====================
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
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

    print(f"RealSense подключена. Focal ≈ {intrinsics.fx:.1f}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    print("=== АВТОМАТИЧЕСКАЯ КАЛИБРОВКА ===")
    print("Ожидание обнаружения всех 4 маркеров...")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            gray = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2GRAY)

            corners, ids, _ = detector.detectMarkers(gray)

            if ids is not None and len(ids) >= 4:
                detected = {}
                for i, mid in enumerate(ids.flatten()):
                    if mid in marker_ids:
                        idx = marker_ids.index(mid)
                        detected[idx] = corners[i]

                if len(detected) == 4:
                    image_points = np.array([
                        np.mean(detected[0][0], axis=0),
                        np.mean(detected[1][0], axis=0),
                        np.mean(detected[2][0], axis=0),
                        np.mean(detected[3][0], axis=0)
                    ], dtype=np.float32)

                    success, rvec, tvec = cv2.solvePnP(
                        object_points, image_points, camera_matrix, dist_coeffs
                    )

                    if success:
                        R_cam_to_world = cv2.Rodrigues(rvec)[0].T
                        t_cam_to_world = -R_cam_to_world @ tvec

                        transform = np.eye(4, dtype=np.float32)
                        transform[:3, :3] = R_cam_to_world
                        transform[:3, 3] = t_cam_to_world.ravel()

                        # === АВТОМАТИЧЕСКОЕ СОХРАНЕНИЕ ===
                        os.makedirs("calibration", exist_ok=True)
                        ts = datetime.now().strftime("%Y%m%d_%H%M")

                        np.save("calibration/camera_matrix.npy", camera_matrix)
                        np.save("calibration/dist_coeffs.npy", dist_coeffs)
                        np.save("calibration/transform_cam_to_world.npy", transform)
                        np.save("calibration/rvec.npy", rvec)
                        np.save("calibration/tvec.npy", tvec)

                        with open("calibration/README.txt", "w", encoding="utf-8") as f:
                            f.write(f"Calibration {ts}\n")
                            f.write(f"Focal length: {intrinsics.fx:.2f} pixels\n\n")
                            f.write("=== Transform Camera → World ===\n")
                            f.write(str(transform))

                        pos = t_cam_to_world.ravel() * 1000
                        euler = rotation_matrix_to_euler(R_cam_to_world)

                        print("\n✅ Калибровка успешно выполнена и сохранена!")
                        print(f"X = {pos[0]:.2f} мм | Y = {pos[1]:.2f} мм | Z = {pos[2]:.2f} мм")
                        print(f"Roll = {euler[0]:.2f}° | Pitch = {euler[1]:.2f}° | Yaw = {euler[2]:.2f}°")
                        print("Файлы сохранены в папку 'calibration/'")

                        break  # Выходим после успешной калибровки

    finally:
        pipeline.stop()
        print("Калибровка завершена.")


if __name__ == "__main__":
    main()