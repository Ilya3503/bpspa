import cv2
import cv2.aruco as aruco

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
MARKER_SIZE_PX = 800        

for marker_id in [0, 1, 2, 3]:
    marker_image = aruco.drawMarker(aruco_dict, marker_id, MARKER_SIZE_PX)
    cv2.imwrite(f"marker_id{marker_id}.png", marker_image)
    print(f"Создан marker_id{marker_id}.png")

print("\nНапечатай маркеры так, чтобы чёрный квадрат был примерно 100 мм.")