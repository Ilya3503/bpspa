import pyrealsense2 as rs
import numpy as np
import cv2
import sys

# ========================= НАСТРОЙКИ =========================
MARKER_SIZE_MM = 80.0
TABLE_WIDTH_MM = 600.0   # расстояние между внешними краями маркеров по X
TABLE_HEIGHT_MM = 400.0  # расстояние между внешними краями маркеров по Y

ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50

# Разрешение камеры
WIDTH = 1280
HEIGHT = 720
# ===========================================================

def order_markers_by_position(markers):
    """Определяет позиции маркеров: BL, BR, TL, TR по координатам центров"""
    if len(markers) < 4:
        return None
    
    # Сортируем по Y (сверху вниз)
    sorted_by_y = sorted(markers.items(), key=lambda x: x[1]['center'][1])
    
    top_two = sorted(sorted_by_y[:2], key=lambda x: x[1]['center'][0])  # левый и правый верхние
    bottom_two = sorted(sorted_by_y[2:], key=lambda x: x[1]['center'][0]) # левый и правый нижние

    return {
        'BL': bottom_two[0][1],  # Bottom Left  → (0, 0)
        'BR': bottom_two[1][1],  # Bottom Right → (600, 0)
        'TL': top_two[0][1],     # Top Left     → (0, 400)
        'TR': top_two[1][1],     # Top Right    → (600, 400)
    }


def main():
    # ==================== RealSense ====================
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 6)

    print("Запуск Intel RealSense D415...")
    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Ошибка запуска RealSense: {e}")
        print("Проверьте подключение камеры по USB 3.0")
        sys.exit(1)

    # ==================== ArUco ====================
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    parameters = cv2.aruco.DetectorParameters()
    # Улучшаем детекцию
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    print("Камера запущена. Ищем 4 маркера...")
    print("Левый нижний = (0, 0) | Вправо = +X | Вверх = +Y")

    H = None  # Матрица гомографии

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            display_frame = frame.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            status_text = "Ожидаем 4 маркера..."

            if ids is not None and len(ids) >= 4:
                # Создаём словарь маркеров
                markers = {}
                for i, corner in enumerate(corners):
                    marker_id = int(ids[i][0])
                    center = np.mean(corner[0], axis=0)
                    markers[marker_id] = {
                        'corners': corner[0].astype(np.float32),
                        'center': center
                    }

                cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)

                ordered = order_markers_by_position(markers)
                
                if ordered:
                    # Точки в системе камеры (центры)
                    camera_points = np.array([
                        ordered['BL']['center'],
                        ordered['BR']['center'],
                        ordered['TR']['center'],
                        ordered['TL']['center']
                    ], dtype=np.float32)

                    # Реальные координаты стола в мм
                    world_points = np.array([
                        [0, 0],
                        [TABLE_WIDTH_MM, 0],
                        [TABLE_WIDTH_MM, TABLE_HEIGHT_MM],
                        [0, TABLE_HEIGHT_MM]
                    ], dtype=np.float32)

                    # Вычисляем гомографию
                    H, status = cv2.findHomography(camera_points, world_points)

                    status_text = f"✓ Гомография рассчитана | 4 маркера"
                    
                    # Показываем координаты центра кадра
                    if H is not None:
                        center_cam = np.array([[[WIDTH//2, HEIGHT//2]]], dtype=np.float32)
                        center_table = cv2.perspectiveTransform(center_cam, H)[0][0]
                        x, y = center_table
                        cv2.putText(display_frame, f"Центр: X={x:6.1f} Y={y:6.1f} мм", 
                                  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            else:
                cv2.putText(display_frame, f"Найдено маркеров: {len(ids) if ids is not None else 0}/4", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Информационная панель
            cv2.putText(display_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Нажми Q для выхода", (10, HEIGHT-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("RealSense D415 — Table Coordinate System", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()