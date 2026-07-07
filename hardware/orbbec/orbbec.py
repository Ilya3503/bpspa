from pyorbbecsdk import Pipeline, PointCloudFilter
from pathlib import Path
import time
import numpy as np
import open3d as o3d

def main():
    print("=== Orbbec Femto Bolt - Тест ===\n")

    pipeline = Pipeline()

    try:
        pipeline.start()
        print("[OK] Камера запущена\n")

        # Получаем кадры
        frames = pipeline.wait_for_frames(3000)
        if frames is None:
            print("[ОШИБКА] Не удалось получить кадры от камеры")
            return

        # Генерируем облако точек (пока без цвета)
        pc_filter = PointCloudFilter()
        point_cloud = pc_filter.process(frames)

        points = np.asarray(point_cloud.get_points())
        print(f"[OK] Получено точек: {len(points)}")

        if len(points) == 0:
            print("[ПРЕДУПРЕЖДЕНИЕ] Облако точек пустое")
            return

        # === Сохранение в папку data/ ===
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        OUTPUT_DIR = PROJECT_ROOT / "data"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        filename = OUTPUT_DIR / f"orbbec_{int(time.time())}.ply"

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        o3d.io.write_point_cloud(str(filename), pcd)
        print(f"\n[ГОТОВО] Файл сохранён: {filename}")

    except Exception as e:
        print(f"[ОШИБКА] {e}")

    finally:
        pipeline.stop()
        print("\nКамера остановлена.")


if __name__ == "__main__":
    main()