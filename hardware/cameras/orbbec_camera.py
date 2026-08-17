import threading
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import cv2

log = logging.getLogger(__name__)

try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
    HAS_ORBBEC = True
except ImportError:
    HAS_ORBBEC = False
    log.warning("pyorbbecsdk не установлен — Orbbec в режиме заглушки")

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


class OrbbecCamera:
    def __init__(self, config: dict):
        cap = config.get("capture", {})
        orb = config.get("../orbbec", {})
        self.pixel_crop = cap.get("pixel_crop", {})
        # мм -> м. Femto Bolt (ToF Azure Kinect) отдаёт глубину uint16 в миллиметрах.
        self.depth_scale_to_m = float(orb.get("depth_scale_to_meters", 0.001))
        self.width = int(orb.get("preview_width", 640))
        self.height = int(orb.get("preview_height", 576))

        self._pipeline = None
        self._intr = None            # (fx, fy, cx, cy) — кэшируем в start()
        self._lock = threading.Lock()
        self._running = False
        self._stub = not (HAS_ORBBEC and HAS_OPEN3D)

    # ---------- lifecycle ----------

    def start(self):
        if self._stub:
            log.warning("Orbbec в режиме заглушки")
            self._running = True
            return
        try:
            self._pipeline = Pipeline()
            cfg = Config()
            cfg.enable_stream(OBSensorType.DEPTH_SENSOR)     # как в твоём рабочем orbbec.py
            try:
                cfg.enable_stream(OBSensorType.COLOR_SENSOR)  # нужен для видеопотока
            except Exception as e:
                log.warning(f"Orbbec: цветной поток недоступен: {e}")
            self._pipeline.start(cfg)

            # intrinsics глубины — берём один раз. VERIFY (1): имя метода/полей.
            try:
                param = self._pipeline.get_camera_param()
                di = param.depth_intrinsic
                self._intr = (float(di.fx), float(di.fy), float(di.cx), float(di.cy))
                log.info(f"Orbbec intrinsics (depth): {self._intr}")
            except Exception as e:
                log.warning(f"Orbbec: get_camera_param не сработал ({e}); возьму из кадра")

            self._running = True
            log.info("Orbbec камера запущена")
        except Exception as e:
            log.error(f"Не удалось запустить Orbbec: {e}. Режим заглушки.")
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
            log.info("Orbbec камера остановлена")
        except Exception as e:
            log.warning(f"Ошибка при остановке Orbbec: {e}")

    # ---------- видеопоток ----------

    def _stub_color(self):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.putText(img, "NO ORBBEC (stub)", (30, self.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return img

    def get_color_frame_for_stream(self) -> Optional[np.ndarray]:
        if self._stub:
            return self._stub_color()
        with self._lock:
            if self._pipeline is None:
                return None
            try:
                frames = self._pipeline.wait_for_frames(2000)
                if frames is None:
                    return None
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    return None
                return self._color_to_bgr(color_frame)
            except Exception as e:
                log.warning(f"Orbbec get_color_frame_for_stream: {e}")
                return None

    @staticmethod
    def _color_to_bgr(color_frame) -> Optional[np.ndarray]:
        # VERIFY (3): набор форматов зависит от сборки SDK. Если видео чёрное/битое —
        # проще всего взять готовый frame_to_bgr_image из pyorbbecsdk/examples/utils.py.
        w, h = color_frame.get_width(), color_frame.get_height()
        data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
        fmt = color_frame.get_format()
        try:
            if fmt == OBFormat.MJPG:
                return cv2.imdecode(data, cv2.IMREAD_COLOR)
            if fmt == OBFormat.RGB:
                return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
            if fmt == OBFormat.BGR:
                return data.reshape(h, w, 3)
            if fmt == OBFormat.YUYV:
                return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
        except Exception as e:
            log.warning(f"Orbbec: не декодировал цвет (fmt={fmt}): {e}")
        # последний шанс — вдруг это JPEG
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    # ---------- захват облака ----------

    def capture_pointcloud(self, output_dir: str = "data") -> Optional[str]:
        if self._stub:
            log.error("Orbbec stub — облако не снять")
            return None

        with self._lock:
            if self._pipeline is None:
                return None
            try:
                frames = self._pipeline.wait_for_frames(5000)
                if frames is None:
                    log.error("Orbbec: нет кадров")
                    return None
                depth_frame = frames.get_depth_frame()
                if depth_frame is None:
                    log.error("Orbbec: нет depth-кадра")
                    return None

                w = depth_frame.get_width()
                h = depth_frame.get_height()
                depth = np.frombuffer(depth_frame.get_data(),
                                      dtype=np.uint16).reshape(h, w).copy()

                intr = self._intr
                if intr is None:
                    # VERIFY (1, запасной путь): intrinsics из профиля кадра.
                    vsp = depth_frame.get_stream_profile().as_video_stream_profile()
                    ob = vsp.get_intrinsic()
                    intr = (float(ob.fx), float(ob.fy), float(ob.cx), float(ob.cy))
            except Exception as e:
                log.error(f"Orbbec capture grab failed: {e}")
                return None

        # дальше — чистая математика, без блокировки камеры
        fx, fy, cx, cy = intr

        left = int(self.pixel_crop.get("left", 0));   right = int(self.pixel_crop.get("right", 0))
        top = int(self.pixel_crop.get("top", 0));     bottom = int(self.pixel_crop.get("bottom", 0))
        if top > 0:    depth[:top, :] = 0
        if bottom > 0: depth[-bottom:, :] = 0
        if left > 0:   depth[:, :left] = 0
        if right > 0:  depth[:, -right:] = 0

        # VERIFY (2): единицы. Femto Bolt = мм -> множитель 0.001.
        depth_m = depth.astype(np.float32) * self.depth_scale_to_m

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth_m
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        mask = z.reshape(-1) > 0
        points = points[mask]
        if len(points) == 0:
            log.warning("Orbbec: 0 точек после фильтрации")
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = Path(output_dir) / f"orbbec_{ts}.ply"
        o3d.io.write_point_cloud(str(filepath), pcd)
        log.info(f"Orbbec: сохранено облако {filepath} ({len(points)} точек)")
        return str(filepath)