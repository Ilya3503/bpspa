from pyorbbecsdk import (
    Pipeline, Config, PointCloudFilter, OBSensorType, save_point_cloud_to_ply
)
from pathlib import Path
from datetime import datetime


def main():
    print("=== Orbbec Femto Bolt — Цветное облако точек (тест без alignment) ===\n")

    pipeline = Pipeline()
    config = Config()

    try:
        config.enable_stream(OBSensorType.DEPTH_SENSOR)
        config.enable_stream(OBSensorType.COLOR_SENSOR)

        pipeline.start(config)
        print("[OK] Камера запущена (Depth + Color)\n")

        frames = pipeline.wait_for_frames(3000)
        if frames is None:
            print("[ОШИБКА] Не удалось получить кадры")
            return

        # Пробуем сгенерировать облако без alignment
        pc_filter = PointCloudFilter()
        point_cloud = pc_filter.process(frames)

        # Сохранение
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        OUTPUT_DIR = PROJECT_ROOT / "data"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = str(OUTPUT_DIR / f"orbbec_color_{timestamp}.ply")

        save_point_cloud_to_ply(filename, point_cloud)
        print(f"[ГОТОВО] Файл сохранён: {filename}")

    except Exception as e:
        print(f"[ОШИБКА] {e}")

    finally:
        pipeline.stop()
        print("\nКамера остановлена.")


if __name__ == "__main__":
    main()