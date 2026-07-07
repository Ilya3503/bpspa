from pyorbbecsdk import Pipeline, Config, OBSensorType, PointCloudFilter
from pathlib import Path
import time
import numpy as np
import open3d as o3d

def main():
    print("=== Orbbec Femto Bolt - Получение облака точек ===\n")

    pipeline = Pipeline()
    config = Config()

    try:
        # === Включаем поток глубины ===
        config.enable_stream(OBSensorType.DEPTH)
        # При желании можно включить и цвет:
        # config.enable_stream(OBSensorType.COLOR)

        pipeline.start(config)
        print("[OK] Камера запущена с потоком Depth\n")

        # Ждём кадры
        frames = pipeline.wait_for_frames(3000)
        if frames is None:
            print("[ОШИБКА] Не удалось получить кадры")
            return

        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            print("[ОШИБКА] Depth frame отсутствует")
            return

        print(f"[OK] Получен depth frame: {depth_frame.width}x{depth_frame.height}")

        # === Генерация облака точек ===
        pc_filter = PointCloudFilter()
        point_cloud = pc_filter.process(frames)

        points = np.asarray(point_cloud.get_points())
        print(f"[OK] Точек в облаке: {len(points)}")

        if len(points) == 0:
            print("[ПРЕДУПРЕЖДЕНИЕ] Облако пустое")
            return

        # === Сохранение ===
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        OUTPUT_DIR = PROJECT_ROOT / "data"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        filename = OUTPUT_DIR / f"orbbec_{int(time.time())}.ply"

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        o3d.io.write_point_cloud(str(filename), pcd)
        print(f"\n[ГОТОВО] Облако точек сохранено: {filename}")

    except Exception as e:
        print(f"[ОШИБКА] {e}")

    finally:
        pipeline.stop()
        print("\nКамера остановлена.")


if __name__ == "__main__":
    main()