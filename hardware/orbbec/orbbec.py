from pyorbbecsdk import (
    Pipeline, Config, PointCloudFilter, OBSensorType,
    AlignFilter, OBAlignMode, save_point_cloud_to_ply
)
from pathlib import Path
from datetime import datetime


def main():
    print("=== Orbbec Femto Bolt — Цветное облако точек ===\n")

    pipeline = Pipeline()
    config = Config()

    try:
        # Включаем оба потока
        config.enable_stream(OBSensorType.DEPTH_SENSOR)
        config.enable_stream(OBSensorType.COLOR_SENSOR)

        pipeline.start(config)
        print("[OK] Камера запущена (Depth + Color)\n")

        # Получаем кадры
        frames = pipeline.wait_for_frames(3000)
        if frames is None:
            print("[ОШИБКА] Не удалось получить кадры")
            return

        # === Alignment (Depth → Color) ===
        align = AlignFilter(OBAlignMode.ALIGN_D2C)
        aligned_frames = align.process(frames)

        # === Генерация цветного облака точек ===
        pc_filter = PointCloudFilter()
        point_cloud = pc_filter.process(aligned_frames)

        # === Сохранение ===
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        OUTPUT_DIR = PROJECT_ROOT / "data"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = str(OUTPUT_DIR / f"orbbec_color_{timestamp}.ply")

        save_point_cloud_to_ply(filename, point_cloud)
        print(f"[ГОТОВО] Цветное облако точек сохранено: {filename}")

    except Exception as e:
        print(f"[ОШИБКА] {e}")

    finally:
        pipeline.stop()
        print("\nКамера остановлена.")


if __name__ == "__main__":
    main()