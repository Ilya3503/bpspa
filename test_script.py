import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import os

def capture_pointcloud(save_path="pointcloud.ply", use_filters=True):
    """
    Захватывает облако точек с RealSense и сохраняет в .ply
    use_filters=True  -> с дефолтными фильтрами (более гладко, но скруглённые углы)
    use_filters=False -> минимальная обработка (более чёткие углы, но больше шума)
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)

    profile = pipeline.start(config)

    # Выравнивание depth к color
    align = rs.align(rs.stream.color)

    # Получаем сенсор глубины
    depth_sensor = profile.get_device().first_depth_sensor()

    # Настройка фильтров
    if use_filters:
        # Дефолтные/умеренные фильтры
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 2)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial.set_option(rs.option.filter_smooth_delta, 20)

        temporal = rs.temporal_filter()
        hole_filling = rs.hole_filling_filter()
    else:
        # Минимальная обработка — для максимальной чёткости краёв
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 1)      # минимальное сглаживание
        spatial.set_option(rs.option.filter_smooth_alpha, 0.1)
        spatial.set_option(rs.option.filter_smooth_delta, 10)

        temporal = rs.temporal_filter()
        temporal.set_option(rs.option.filter_smooth_alpha, 0.1)
        hole_filling = rs.hole_filling_filter()
        hole_filling.set_option(rs.option.holes_fill, 0)       # отключаем заполнение дыр

    print(f"Захват облака точек... (use_filters={use_filters})")
    print("Подождите 2-3 секунды для стабилизации...")

    # Пропускаем первые несколько кадров (прогрев)
    for _ in range(30):
        pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        print("Ошибка: не удалось получить кадры")
        return

    # Применяем фильтры
    if use_filters:
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)

    # Создаём point cloud через Open3D
    pc = rs.pointcloud()
    points = pc.calculate(depth_frame)
    vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

    # Цвета (опционально)
    color_image = np.asanyarray(color_frame.get_data())
    colors = color_image.reshape(-1, 3) / 255.0

    # Создаём Open3D облако
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Удаляем точки с нулевой глубиной
    pcd = pcd.remove_non_finite_points()

    # Сохраняем
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    o3d.io.write_point_cloud(save_path, pcd)
    print(f"Сохранено: {save_path}")
    print(f"Количество точек: {len(pcd.points)}")

    # Визуализация
    print("Открываю визуализацию... (закрой окно, чтобы продолжить)")
    o3d.visualization.draw_geometries([pcd])

    pipeline.stop()


if __name__ == "__main__":
    # === Вариант 1: С фильтрами (по умолчанию) ===
    capture_pointcloud("pointcloud_with_filters.ply", use_filters=True)

    # === Вариант 2: Без сильного сглаживания (более чёткие углы) ===
    capture_pointcloud("pointcloud_sharp_edges.ply", use_filters=False)