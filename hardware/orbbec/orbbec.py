from pyorbbecsdk import Pipeline, Config, PointCloudFilter, OBSensorType, save_point_cloud_to_ply
from pathlib import Path
from datetime import datetime
import time


def main():
    print("=== Orbbec Femto Bolt — Получение облака точек ===\n")

    pipeline = Pipeline()
    config = Config()

    try:
        config.enable_stream(OBSensorType.DEPTH_SENSOR)
        pipeline.start(config)
        print("[OK] Камера успешно запущена\n")

        # Ждём depth frame
        frames = None
        for attempt in range(10):
            frames = pipeline.wait_for_frames(500)
            if frames is not None and frames.get_depth_frame() is not None:
                break
            print(f"  Попытка {attempt + 1}/10...")

        if frames is None or frames.get_depth_frame() is None:
            print("[ОШИБКА] Не удалось получить depth frame")
            return

        depth_frame = frames.get_depth_frame()
        print(f"[OK] Получен depth frame: {depth_frame.get_width()}x{depth_frame.get_height()}")

        # Генерация облака точек
        pc_filter = PointCloudFilter()
        point_cloud = pc_filter.process(frames)

        # === Сохранение с красивым именем ===
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        OUTPUT_DIR = PROJECT_ROOT / "data"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = str(OUTPUT_DIR / f"orbbec_{timestamp}.ply")

        save_point_cloud_to_ply(filename, point_cloud)
        print(f"\n[ГОТОВО] Облако точек сохранено: {filename}")

    except Exception as e:
        print(f"[ОШИБКА] {e}")

    finally:
        pipeline.stop()
        print("\nКамера остановлена.")


if __name__ == "__main__":
    main()