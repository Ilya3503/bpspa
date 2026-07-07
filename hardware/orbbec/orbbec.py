from pyorbbecsdk import Pipeline, AlignFilter, AlignMode, PointCloudFilter
from pathlib import Path
import time
import numpy as np
import open3d as o3d

def main():
    print("Запуск Orbbec Femto Bolt...")

    pipeline = Pipeline()

    try:
        pipeline.start()
        print("[OK] Камера подключена")

        frames = pipeline.wait_for_frames(3000)
        if frames is None:
            print("[ОШИБКА] Не удалось получить кадры")
            return

        # === Выравнивание (Depth → Color) ===
        align = AlignFilter(AlignMode.D2C)
        aligned = align.process(frames)

        # === Генерация облака точек ===
        pc_filter = PointCloudFilter()
        point_cloud = pc_filter.process(aligned)

        points = np.asarray(point_cloud.get_points())
        colors = np.asarray(point_cloud.get_colors())

        print(f"[OK] Получено точек: {len(points)}")

        if len(points) == 0:
            print("[ПРЕДУПРЕЖДЕНИЕ] Облако пустое")
            return

        # === Сохранение в data/ ===
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        OUTPUT_DIR = PROJECT_ROOT / "data"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        filename = OUTPUT_DIR / f"orbbec_{int(time.time())}.ply"

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors.size > 0:
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

        o3d.io.write_point_cloud(str(filename), pcd)
        print(f"[ГОТОВО] Сохранено: {filename}")

    except Exception as e:
        print(f"[ОШИБКА] {e}")

    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()