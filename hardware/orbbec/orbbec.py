import time
from pyorbbecsdk import Pipeline, AlignFrame, AlignMode, PointCloudFilter
import numpy as np
import open3d as o3d

def main():
    print("Инициализация камеры Orbbec...")
    pipeline = Pipeline()

    try:
        pipeline.start()
        print("Камера успешно запущена.")

        print("Ожидание кадров...")
        frames = pipeline.wait_for_frames(2000)  # ждём до 2 секунд

        if frames is None:
            print("Не удалось получить кадры!")
            return

        # Делаем alignment (Depth → Color). Лучше использовать hardware D2C
        align = AlignFrame(AlignMode.D2C)
        aligned_frames = align.process(frames)

        # Генерируем цветное облако точек
        pc_filter = PointCloudFilter()
        point_cloud = pc_filter.process(aligned_frames)

        # Получаем данные
        points = np.asarray(point_cloud.get_points())
        colors = np.asarray(point_cloud.get_colors())

        print(f"Получено точек: {len(points)}")

        if len(points) == 0:
            print("Облако точек пустое. Попробуйте ещё раз.")
            return

        # Создаём объект Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if colors.size > 0:
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

        # Сохраняем в .ply
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        OUTPUT_DIR = PROJECT_ROOT / "data"
        filename = OUTPUT_DIR / f"orbbec_pointcloud_{int(time.time())}.ply"
        o3d.io.write_point_cloud(filename, pcd)
        print(f"✓ Облако точек успешно сохранено в файл: {filename}")

    except Exception as e:
        print(f"Ошибка: {e}")

    finally:
        pipeline.stop()
        print("Камера остановлена.")


if __name__ == "__main__":
    main()