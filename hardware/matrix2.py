import pyrealsense2 as rs
import numpy as np
import cv2

# ========================= НАСТРОЙКИ =========================
MARKER_SIZE_MM = 80.0
TABLE_WIDTH_MM = 600.0
TABLE_HEIGHT_MM = 400.0

ARUCO_DICT = cv2.aruco.DICT_4X4_50
# ===========================================================

def main():
    # Настройка RealSense
    pipeline = rs.pipeline()
    config = rs.config()

    # Включаем цветной поток (RGB)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    # Можно включить глубину, если понадобится позже:
    # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    print("Запуск RealSense D415...")
    pipeline.start(config)

    # ArUco детектор
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    print("Камера запущена. Нажми 'q' для выхода.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())

            # Обработка ArUco
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is not None and len(ids) >= 4:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.putText(frame, f"Найдено маркеров: {len(ids)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                # Здесь будет логика гомографии (4 маркера) — могу добавить в следующем шаге
            else:
                cv2.putText(frame, "Ожидаем 4 маркера...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            cv2.imshow("RealSense D415 - ArUco", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()