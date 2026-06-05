import cv2
import numpy as np
import pyrealsense2 as rs
import os
from datetime import datetime

def rotation_matrix_to_euler(R):
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        roll  = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = np.arctan2(R[1,0], R[0,0])
    else:
        roll  = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = 0
    return np.degrees([roll, pitch, yaw])


def main():
    # ==================== НАСТРОЙКИ ====================
    marker_ids = [0, 1, 2, 3]
    dist_x = 520.0   # мм
    dist_y = 320.0   # мм
    marker_size_mm = 80.0

    half_x = dist_x / 2
    half_y = dist_y / 2

    object_points = np.array([
        [-half_x, -half_y, 0.0],
        [-half_x,  half_y, 0.0],
        [ half_x,  half_y, 0.0],
        [ half_x, -half_y, 0.0]
    ], dtype=np.float32)

    focal = 1450.0   # Подбирай!

    # ==================== RealSense ====================
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

    print(f"RealSense подключена. Focal ≈ {intrinsics.fx:.1f}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    print("=== РЕАЛ-ТАЙМ КАЛИБРОВКА ===")
    print("Нажми 's' — сохранить, 'q' — выход")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
                        R = cv2.Rodrigues(rvec)[0].T
                        t = -R @ tvec
                        pos = t.ravel() * 1000
                        euler = rotation_matrix_to_euler(R)

                        # Визуализация
                        cv2.aruco.drawDetectedMarkers(img, corners)
                        axis = np.float32([[0,0,0], [150,0,0], [0,150,0], [0,0,150]])
                        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
                        imgpts = np.int32(imgpts).reshape(-1, 2)
                        origin = tuple(imgpts[0])

                        cv2.line(img, origin, tuple(imgpts[1]), (0,0,255), 4)
                        cv2.line(img, origin, tuple(imgpts[2]), (0,255,0), 4)
                        cv2.line(img, origin, tuple(imgpts[3]), (255,0,0), 4)

                        cv2.putText(img, f"X={pos[0]:.1f} Y={pos[1]:.1f} Z={pos[2]:.1f} mm",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                        key = cv2.waitKey(1)
                        if key == ord('s'):
                            transform = np.eye(4, dtype=np.float32)
                            transform[:3,:3] = R
                            transform[:3,3] = t.ravel()
                            np.save("calibration/transform_cam_to_world.npy", transform)
                            print("Матрица сохранена")
                        if key == ord('q'):
                            break

            cv2.imshow("Real-time ArUco Calibration", img)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()