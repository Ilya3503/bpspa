import cv2
import numpy as np
import sys

# ========================= НАСТРОЙКИ =========================
MARKER_SIZE_MM = 80.0                    # размер маркера
TABLE_WIDTH_MM = 600.0                   # расстояние между внешними краями маркеров по X
TABLE_HEIGHT_MM = 400.0                  # расстояние между внешними краями маркеров по Y

# Словарь ArUco (можно изменить)
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# IDs маркеров (замени на реальные, если знаешь)
# По умолчанию определяем по позиции на изображении
MARKER_BOTTOM_LEFT  = None   # будет определён автоматически
MARKER_BOTTOM_RIGHT = None
MARKER_TOP_LEFT     = None
MARKER_TOP_RIGHT    = None
# ===========================================================

def order_points(pts):
    """Упорядочиваем точки: BL, BR, TR, TL"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    rect[0] = pts[np.argmin(s)]      # Bottom Left
    rect[1] = pts[np.argmax(diff)]   # Bottom Right
    rect[2] = pts[np.argmax(s)]      # Top Right
    rect[3] = pts[np.argmin(diff)]   # Top Left
    return rect


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть камеру")
        sys.exit(1)

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    print("Программа запущена. Нажми 'q' для выхода.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None or len(ids) < 4:
            cv2.putText(frame, "Не все 4 маркера найдены", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Camera -> Table Transform", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Преобразуем corners в удобный формат
        markers = {}
        for i, corner in enumerate(corners):
            marker_id = int(ids[i][0])
            center = np.mean(corner[0], axis=0)
            markers[marker_id] = {
                'corners': corner[0].astype(np.float32),
                'center': center
            }

        # Определяем маркеры по позиции (если ID неизвестны)
        centers = [(m['center'], mid) for mid, m in markers.items()]
        centers.sort(key=lambda x: x[0][1])  # сортируем по Y (сверху вниз)

        top_two = sorted(centers[:2], key=lambda x: x[0][0])   # левый и правый верхние
        bottom_two = sorted(centers[2:], key=lambda x: x[0][0]) # левый и правый нижние

        bl_id = bottom_two[0][1]
        br_id = bottom_two[1][1]
        tl_id = top_two[0][1]
        tr_id = top_two[1][1]

        # Точки в камере (центры маркеров)
        camera_points = np.array([
            markers[bl_id]['center'],  # Bottom Left  -> (0, 0)
            markers[br_id]['center'],  # Bottom Right -> (600, 0)
            markers[tr_id]['center'],  # Top Right    -> (600, 400)
            markers[tl_id]['center'],  # Top Left     -> (0, 400)
        ], dtype=np.float32)

        # Реальные координаты на столе (в мм)
        world_points = np.array([
            [0, 0],
            [TABLE_WIDTH_MM, 0],
            [TABLE_WIDTH_MM, TABLE_HEIGHT_MM],
            [0, TABLE_HEIGHT_MM]
        ], dtype=np.float32)

        # Вычисляем гомографию
        H, _ = cv2.findHomography(camera_points, world_points)

        # Рисуем маркеры и оси
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Пример: преобразование точки (центр изображения)
        if H is not None:
            # Преобразуем центр камеры в координаты стола
            center_cam = np.array([[frame.shape[1]/2, frame.shape[0]/2]], dtype=np.float32)
            center_table = cv2.perspectiveTransform(center_cam.reshape(1,1,2), H)
            
            x, y = center_table[0][0]
            cv2.putText(frame, f"Center: X={x:.1f}mm Y={y:.1f}mm", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Camera -> Table Transform", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()