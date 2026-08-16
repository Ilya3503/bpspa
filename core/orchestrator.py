"""
Координатор: знает, что делать в каждом состоянии.
Шлёт WebSocket-события на каждом шаге.

"""
import asyncio
import base64
import logging
from pathlib import Path

import cv2
import numpy as np

from core.state_machine import StateMachine, State
from api.ws_manager import WSManager
from hardware.camera import RealSenseCamera
from processing import pipeline as pl
from processing import preprocessing as pre

log = logging.getLogger(__name__)


class Orchestrator:
    """
    Принимает команды (start/reset/stop) и продвигает state machine.
    Тяжёлые синхронные операции (камера, обработка) выполняются в threadpool.
    """

    def __init__(self, sm: StateMachine, ws: WSManager,
                 camera: RealSenseCamera, config: dict):
        self.sm = sm
        self.ws = ws
        self.camera = camera
        self.config = config
        self._busy = asyncio.Lock()

    # ---------------- публичный API ----------------

    async def handle_command(self, action: str):
        if action == "stop":
            self.sm.trigger("stop")
            await self._emit_state()
            return

        if action == "reset":
            self.sm.trigger("reset")
            await self._emit_state()
            return

        if action == "start":
            if self._busy.locked():
                await self.ws.broadcast({"event": "error", "message": "Уже идёт цикл"})
                return
            self.sm.trigger_start()
            await self._emit_state()
            asyncio.create_task(self._run_single_view_cycle())
            return

        raise ValueError(f"Неизвестная команда: {action}")

    # ---------------- циклы ----------------

    async def _run_single_view_cycle(self):
        """Одно-ракурсный режим: один снимок → processing → execute → done."""
        async with self._busy:
            try:
                input_file = await self._step_capture(view=1, single_mode=True)
                self.sm.advance(State.PROCESSING)
                await self._emit_state()

                result = await self._step_process(input_file)
                self.sm.set_data(last_result=result)
                self.sm.advance(State.EXECUTING)
                await self._emit_state()

                await self._step_execute(result)
                self.sm.advance(State.DONE)
                await self._emit_state()
                await self.ws.broadcast({"event": "done", "result_file": "results/position.json"})
            except Exception as e:
                log.exception("Ошибка в _run_single_view_cycle")
                self.sm.fail(str(e))
                await self.ws.broadcast({"event": "error", "message": str(e)})
                await self._emit_state()

    # ---------------- шаги ----------------

    async def _step_capture(self) -> str:
        await self.ws.broadcast({"event": "capture_start"})
        loop = asyncio.get_event_loop()
        filepath = await loop.run_in_executor(None, self.camera.capture_pointcloud, "data")
        if filepath is None:
            raise RuntimeError("Не удалось захватить облако")
        pcd = pre.load_pcd(filepath)
        points = len(pcd.points)

        self.sm.set_data(file_single=filepath)

        await self.ws.broadcast({
            "event": "capture_done",
            "file": filepath,
            "points": points,
        })
        return filepath

    async def _step_process(self, input_file: str) -> dict:
        loop = asyncio.get_event_loop()
        ws_sync = self.ws.broadcast_sync

        def _do():
            return self._process_sync(input_file, ws_sync)

        return await loop.run_in_executor(None, _do)

    def _process_sync(self, input_file: str, emit) -> dict:
        """Тонкая обёртка: run_dir + вызов фасада pipeline + финализация папки."""
        run_dir = self._make_run_dir()
        result = pl.run_pipeline(input_file, self.config, run_dir, emit)
        self._finalize_run_dir(run_dir, result.get("num_clusters", 0))
        return result

    async def _step_execute(self, result: dict):
        """Симуляция робота. Пока заглушка — реализует напарник в robot/executor.py."""
        if not self.config.get("robot", {}).get("enabled", False):
            await self.ws.broadcast({
                "event": "simulation_skipped",
                "reason": "robot.enabled = false в конфиге"
            })
            return
        await self.ws.broadcast({"event": "simulation_start"})
        await asyncio.sleep(0.1)
        await self.ws.broadcast({"event": "pick_and_place_done", "object_placed": False})

    # ---------------- видеопоток ----------------

    async def video_stream_task(self):
        fps = self.config["capture"].get("video_feed_fps", 1)
        delay = 1.0 / max(fps, 0.1)
        loop = asyncio.get_event_loop()
        log.info(f"Видеопоток запущен ({fps} fps)")

        while True:
            try:
                frame = await loop.run_in_executor(None, self.camera.get_color_frame_for_stream)
                if frame is not None:
                    small = cv2.resize(frame, (640, 360))
                    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                        await self.ws.broadcast({
                            "event": "video_frame",
                            "image_base64": f"data:image/jpeg;base64,{b64}",
                        })
            except Exception as e:
                log.warning(f"video_stream_task: {e}")
            await asyncio.sleep(delay)

    # ---------------- helpers ----------------

    async def _emit_state(self):
        await self.ws.broadcast({
            "event": "state_changed",
            "state": self.sm.state.value,
            "data_keys": list(self.sm.data.keys()),
        })

    @staticmethod
    def _make_run_dir() -> str:
        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = Path("results") / f"run_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return str(run_dir)


    @staticmethod
    def _finalize_run_dir(run_dir: str, n_clusters: int) -> str:
        src = Path(run_dir)
        dst = src.with_name(f"{src.name}_n{n_clusters}")
        try:
            src.rename(dst)
            return str(dst)
        except OSError as e:
            log.warning(f"Не удалось переименовать {src} → {dst}: {e}")
            return run_dir