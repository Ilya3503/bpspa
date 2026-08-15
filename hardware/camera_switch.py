"""
Переключатель камер. Держит обе реализации, делегирует активной.
Для orchestrator/pipeline это по-прежнему один объект с методами
start / stop / capture_pointcloud / get_color_frame_for_stream.

Активна всегда одна камера, остальные остановлены.
"""
import threading
import logging

from hardware.camera import RealSenseCamera
from hardware.orbbec_camera import OrbbecCamera

log = logging.getLogger(__name__)

_BACKENDS = {
    "realsense": RealSenseCamera,
    "orbbec": OrbbecCamera,
}


class CameraSwitch:
    def __init__(self, config: dict):
        default = config.get("capture", {}).get("backend", "realsense")
        if default not in _BACKENDS:
            log.warning(f"Неизвестный backend '{default}', беру realsense")
            default = "realsense"
        # конструируем обе (не открывает железо — открытие в start())
        self._cameras = {k: cls(config) for k, cls in _BACKENDS.items()}
        self._active_key = default
        self._lock = threading.Lock()   # сериализует переключение и обращения к камере
        self._started = False

    @property
    def active_key(self) -> str:
        return self._active_key

    def available(self):
        return list(self._cameras.keys())

    # --- интерфейс камеры (то, что зовёт orchestrator) ---

    def start(self):
        with self._lock:
            self._cameras[self._active_key].start()
            self._started = True

    def stop(self):
        with self._lock:
            for cam in self._cameras.values():
                try:
                    cam.stop()
                except Exception as e:
                    log.warning(f"stop {cam}: {e}")
            self._started = False

    def capture_pointcloud(self, output_dir: str = "data"):
        with self._lock:
            return self._cameras[self._active_key].capture_pointcloud(output_dir)

    def get_color_frame_for_stream(self):
        with self._lock:
            return self._cameras[self._active_key].get_color_frame_for_stream()

    # --- переключение ---

    def switch(self, key: str) -> str:
        if key not in self._cameras:
            raise ValueError(f"Неизвестная камера: {key}. Доступно: {self.available()}")
        with self._lock:
            if key == self._active_key:
                return self._active_key
            log.info(f"Переключение камеры {self._active_key} → {key}")
            if self._started:
                self._cameras[self._active_key].stop()
                self._cameras[key].start()
            self._active_key = key
            return self._active_key