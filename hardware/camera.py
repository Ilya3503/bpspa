"""
RealSense камера — единственный экземпляр на всё приложение.
Постоянно открытое соединение, потокобезопасный доступ через threading.Lock.

Главная идея: pyrealsense2 синхронная, поэтому мы НЕ дёргаем её из asyncio напрямую.
Все обращения к камере идут либо из:
  (а) фонового потока видеопотока,
  (б) executor-а (run_in_executor) в момент захвата облака.
Lock гарантирует, что в моменте к камере обращается только одно.
"""
import threading
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import cv2

log = logging.getLogger(__name__)

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False
    log.warning("pyrealsense2 не установлен — камера в режиме заглушки")

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


class RealSenseCamera:
    """
    Один экземпляр на всю программу. Открывается в startup, закрывается в shutdown.
    """

    def __init__(self, config: dict):
        cap_cfg = config.get("capture", {})
        self.width = cap_cfg.get("width", 1280)
        self.height = cap_cfg.get("height", 720)
        self.fps = cap_cfg.get("fps", 6)
        self.pixel_crop = cap_cfg.get("pixel_crop", {})

        self._pipeline = None
        self._align = None
        self._spatial = None
        self._temporal = None
        self._hole_filling = None
        self._depth_scale = 1.0
        self._intr = None

        self._lock = threading.Lock()        # защищает обращения к pipeline
        self._latest_color: Optional[np.ndarray] = None
        self._latest_color_ts: float = 0.0
        self._latest_lock = threading.Lock() # защищает кэш последнего кадра

        self._running = False
        self._stub = not HAS_REALSENSE

    # ---------- lifecycle ----------

    def start(self):
        if self._stub:
            log.warning("Камера в режиме заглушки (нет pyrealsense2)")
            self._running = True
            return

        try:
            self._pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

            profile = self._pipeline.start(cfg)
            self._align = rs.align(rs.stream.color)

            # === ФИЛЬТРЫ ОТКЛЮЧЕНЫ ===
            self._spatial = None
            self._temporal = None
            self._hole_filling = None

            depth_sensor = profile.get_device().first_depth_sensor()
            self._depth_scale = depth_sensor.get_depth_scale()
            self._intr = (
                profile.get_stream(rs.stream.depth)
                .as_video_stream_profile()
                .get_intrinsics()
            )

            # Дополнительно отключаем
            try:
                depth_sensor.set_option(rs.option.high_accuracy, 0)
                depth_sensor.set_option(rs.option.disparity_shift, 0)
            except:
                pass

            self._running = True
            log.info(f"Камера запущена: {self.width}x{self.height}@{self.fps}fps, depth_scale={self._depth_scale} (фильтры отключены)")
        except Exception as e:
            log.error(f"Не удалось запустить камеру: {e}. Переход в режим заглушки.")
            self._stub = True
            self._running = True

    def stop(self):
        if self._stub:
            self._running = False
            return
        try:
            with self._lock:
                if self._pipeline is not None:
                    self._pipeline.stop()
                    self._pipeline = None
            self._running = False
            log.info("Камера остановлена")
        except Exception as e:
            log.warning(f"Ошибка при остановке камеры: {e}")

    # ---------- основные методы ----------

    def grab_frames(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Получает один (depth, color) кадр.
        Применяет align + фильтры глубины.
        Возвращает (None, None) если камера не работает.
        НЕ вызывать из asyncio напрямую — только из threadpool / фонового потока.
        """
        if self._stub:
            # Заглушка: возвращаем фиктивный кадр
            color = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(color, "NO CAMERA (stub)", (50, self.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            depth = np.zeros((self.height, self.width), dtype=np.uint16)
            return depth, color

        with self._lock:
            if self._pipeline is None:
                return None
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=5000)
                frames = self._align.process(frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    return None

                # фильтры глубины
                pass

                depth = np.asanyarray(depth_frame.get_data())
                color = np.asanyarray(color_frame.get_data())
                return depth, color
            except Exception as e:
                log.error(f"grab_frames failed: {e}")
                return None

    def get_color_frame_for_stream(self) -> Optional[np.ndarray]:
        """
        Лёгкий метод для видеопотока: один кадр цвета, без фильтров глубины.
        Кэширует результат, чтобы соседние вызовы не ждали.
        """
        if self._stub:
            color = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(color, "NO CAMERA (stub)", (50, self.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            return color

        with self._lock:
            if self._pipeline is None:
                return None
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=2000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    return None
                return np.asanyarray(color_frame.get_data())
            except Exception as e:
                log.warning(f"get_color_frame_for_stream failed: {e}")
                return None

    def capture_pointcloud(self, output_dir: str = "data") -> Optional[str]:
        """
        Снимает один полный кадр depth+color, делает backprojection,
        применяет pixel crop, сохраняет .ply файл.
        Возвращает путь к файлу.
        """
        if not HAS_OPEN3D:
            log.error("Open3D не установлен — не могу сохранить облако")
            return None

        out = self.grab_frames()
        if out is None:
            log.error("capture_pointcloud: не удалось получить кадры")
            return None
        depth, color = out

        # --- pixel crop (как в твоей старой версии) ---
        left   = int(self.pixel_crop.get("left", 0))
        right  = int(self.pixel_crop.get("right", 0))
        top    = int(self.pixel_crop.get("top", 0))
        bottom = int(self.pixel_crop.get("bottom", 0))

        if top > 0:
            depth[:top, :] = 0
            color[:top, :] = 0
        if bottom > 0:
            depth[-bottom:, :] = 0
            color[-bottom:, :] = 0
        if left > 0:
            depth[:, :left] = 0
            color[:, :left] = 0
        if right > 0:
            depth[:, -right:] = 0
            color[:, -right:] = 0

        # --- depth → meters ---
        depth_m = depth.astype(np.float32) * self._depth_scale if not self._stub else depth.astype(np.float32)

        h, w = depth_m.shape

        # --- intrinsics (или заглушка, если stub) ---
        if self._stub or self._intr is None:
            fx = fy = 600.0
            cx, cy = w / 2, h / 2
        else:
            fx, fy = self._intr.fx, self._intr.fy
            cx, cy = self._intr.ppx, self._intr.ppy

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth_m
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        # BGR → RGB и в [0,1]
        colors = cv2.cvtColor(color, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0

        mask = z.reshape(-1) > 0
        points = points[mask]
        colors = colors[mask]

        if len(points) == 0:
            log.warning("capture_pointcloud: после фильтрации 0 точек")
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = Path(output_dir) / f"pointcloud_{timestamp}.ply"
        o3d.io.write_point_cloud(str(filepath), pcd)
        log.info(f"Сохранено облако: {filepath} ({len(points)} точек)")
        return str(filepath)