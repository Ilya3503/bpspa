import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import os

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Уменьшаем разрешение — меньше шансов на крах
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    # Минимальные фильтры
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 1)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.25)

    print("Захват облака точек...")

    # Прогрев
    for _ in range(15):
        pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    depth_frame = spatial.process(depth_frame)   # лёгкий фильтр

    # Создаём point cloud
    pc = rs.pointcloud()
    points = pc.calculate(depth_frame)

    # Получаем вершины
    vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

    # Создаём Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

    # Сохраняем
    save_path = "pointcloud_test.ply"
    o3d.io.write_point_cloud(save_path, pcd)
    print(f"Сохранено: {save_path}")
    print(f"Точек: {len(pcd.points)}")

    # Визуализация
    o3d.visualization.draw_geometries([pcd])

    pipeline.stop()

if __name__ == "__main__":
    main()