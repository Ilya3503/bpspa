import cv2
import numpy as np
import pyrealsense2 as rs
import time

# ========================= НАСТРОЙКИ =========================
MARKER_SIZE_MM = 80.0
TABLE_WIDTH_MM = 600.0
TABLE_HEIGHT_MM = 400.0

# ID маркеров (замени на реальные после первого запуска)
MARKER_IDS = {
    "bottom_left":  0,   # ← левый нижний — начало координат
    "bottom_right": 1,
    "top_right":    2,
    "top_left":     3
}

# ============================================================

# Инициализация ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# =================== RealSense D415 ===================
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)  # 6 FPS как у тебя

profile = pipeline.start(config)

# Получаем параметры камеры
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

print(f"Камера запущена: {intrinsics.width}x{intrinsics.height} @ {intrinsics.fps} FPS")

# Для хранения предыдущей гомографии (стабильность)
last_homography = None
last_corners = None

def get_marker_center(corners):
    return np.mean(corners[0], axis=0)

def detect_and_transform(frame):
    global last_homography, last_corners
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is None or len(ids) < 3:
        # Если мало маркеров — используем предыдущую трансформацию
        if last_homography is not None:
            return frame, last_homography
        return frame, None

    # Создаём словарь id -> corners
    marker_dict = {}
    for i, marker_id in enumerate(ids.flatten()):
        marker_dict[int(marker_id)] = corners[i]

    # Проверяем наличие ключевых маркеров
    required = [MARKER_IDS["bottom_left"], MARKER_IDS["bottom_right"],
                MARKER_IDS["top_right"], MARKER_IDS["top_left"]]
    
    if not all(r in marker_dict for r in required):
        print("⚠️ Не все 4 маркера найдены")
        return frame, last_homography

    # ==================== Точки в пикселях ====================
    bl = get_marker_center(marker_dict[MARKER_IDS["bottom_left"]])   # (0, 0)
    br = get_marker_center(marker_dict[MARKER_IDS["bottom_right"]])  # (600, 0)
    tr = get_marker_center(marker_dict[MARKER_IDS["top_right"]])     # (600, 400)
    tl = get_marker_center(marker_dict[MARKER_IDS["top_left"]])      # (0, 400)

    # Точки источника (камера)
    src_points = np.array([bl, br, tr, tl], dtype=np.float32)

    # Точки назначения (стол в мм)
    dst_points = np.array([
        [0, 0],
        [TABLE_WIDTH_MM, 0],
        [TABLE_WIDTH_MM, TABLE_HEIGHT_MM],
        [0, TABLE_HEIGHT_MM]
    ], dtype=np.float32)

    # Вычисляем гомографию
    H, _ = cv2.findHomography(src_points, dst_points)
    last_homography = H
    last_corners = (bl, br, tr, tl)

    # Визуализация
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    for pt in [bl, br, tr, tl]:
        cv2.circle(frame, tuple(pt.astype(int)), 8, (0, 255, 0), -1)

    return frame, H


def pixel_to_table(x, y, H):
    """Преобразует пиксельные координаты в мм на столе"""
    if H is None:
        return None
    point = np.array([[[x, y]]], dtype=np.float32)
    table_point = cv2.perspectiveTransform(point, H)
    return table_point[0][0]


# ======================= ГЛАВНЫЙ ЦИКЛ =======================
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        processed, homography = detect_and_transform(frame)

        # Пример: клик мышкой → координаты на столе
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and homography is not None:
                mm = pixel_to_table(x, y, homography)
                print(f"📍 Пиксель: ({x}, {y}) → Стол: ({mm[0]:.1f}, {mm[1]:.1f}) мм")

        cv2.namedWindow("RealSense + ArUco → Table")
        cv2.setMouseCallback("RealSense + ArUco → Table", mouse_callback)

        cv2.imshow("RealSense + ArUco → Table", processed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.05)  # небольшая задержка при 6 FPS

finally:
    pipeline.stop()
    cv2.destroyAllWindows()