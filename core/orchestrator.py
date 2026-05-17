"""
Координатор: знает, что делать в каждом состоянии.
Шлёт WebSocket-события на каждом шаге.

Поддерживает два режима съёмки (config.capture.n_views):
- 1: один снимок, без merge
- 2 и больше: два снимка с merge между ними
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

log = logging.getLogger(__name__)


class Orchestrator:
    """
    Принимает команды (start/next_view/reset/stop) и продвигает state machine.
    Тяжёлые синхронные операции (камера, обработка) выполняются в threadpool.
    """

    def __init__(self, sm: StateMachine, ws: WSManager,
                 camera: RealSenseCamera, config: dict):
        self.sm = sm
        self.ws = ws
        self.camera = camera
        self.config = config
        self._busy = asyncio.Lock()

    @property
    def n_views(self) -> int:
        """Текущее число ракурсов из конфига. Читаем динамически — позволяет менять без рестарта."""
        return int(self.config.get("capture", {}).get("n_views", 2))

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
            n = self.n_views
            self.sm.trigger_start(n_views=n)
            await self._emit_state()
            if n >= 2:
                asyncio.create_task(self._run_two_view_cycle_first_half())
            else:
                asyncio.create_task(self._run_single_view_cycle())
            return

        if action == "next_view":
            self.sm.trigger("next_view")
            await self._emit_state()
            asyncio.create_task(self._run_two_view_cycle_second_half())
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

    async def _run_two_view_cycle_first_half(self):
        """Двух-ракурсный режим, часть 1: захват view1 → ждём next_view."""
        async with self._busy:
            try:
                await self._step_capture(view=1)
                self.sm.advance(State.WAITING_VIEW2)
                await self._emit_state()
                await self.ws.broadcast({
                    "event": "waiting_for_next_view",
                    "message": "Переставьте камеру и нажмите NEXT"
                })
            except Exception as e:
                log.exception("Ошибка в _run_two_view_cycle_first_half")
                self.sm.fail(str(e))
                await self.ws.broadcast({"event": "error", "message": str(e)})
                await self._emit_state()

    async def _run_two_view_cycle_second_half(self):
        """Двух-ракурсный режим, часть 2: захват view2 → merge → process → execute → done."""
        async with self._busy:
            try:
                await self._step_capture(view=2)
                self.sm.advance(State.MERGING)
                await self._emit_state()

                merged_file = await self._step_merge()
                self.sm.advance(State.PROCESSING)
                await self._emit_state()

                result = await self._step_process(merged_file)
                self.sm.set_data(last_result=result)
                self.sm.advance(State.EXECUTING)
                await self._emit_state()

                await self._step_execute(result)
                self.sm.advance(State.DONE)
                await self._emit_state()
                await self.ws.broadcast({"event": "done", "result_file": "results/position.json"})
            except Exception as e:
                log.exception("Ошибка в _run_two_view_cycle_second_half")
                self.sm.fail(str(e))
                await self.ws.broadcast({"event": "error", "message": str(e)})
                await self._emit_state()

    # ---------------- шаги ----------------

    async def _step_capture(self, view: int, single_mode: bool = False) -> str:
        await self.ws.broadcast({"event": "capture_start", "view": view, "single_mode": single_mode})
        loop = asyncio.get_event_loop()
        filepath = await loop.run_in_executor(None, self.camera.capture_pointcloud, "data")
        if filepath is None:
            raise RuntimeError(f"Не удалось захватить view {view}")
        pcd = pl.load_pcd(filepath)
        points = len(pcd.points)

        if single_mode:
            self.sm.set_data(file_single=filepath)
        elif view == 1:
            self.sm.set_data(file_view1=filepath)
        else:
            self.sm.set_data(file_view2=filepath)

        await self.ws.broadcast({
            "event": "capture_done",
            "view": view,
            "single_mode": single_mode,
            "file": filepath,
            "points": points,
        })
        return filepath

    async def _step_merge(self) -> str:
        data = self.sm.data
        file_a = data.get("file_view1")
        file_b = data.get("file_view2")
        if not file_a or not file_b:
            raise RuntimeError("Не хватает файлов для merge")

        await self.ws.broadcast({"event": "merging_start", "files": [file_a, file_b]})

        cfg = self.config["merge"]
        T, is_stub = pl.load_calibration_matrix(cfg["calibration_file"])

        loop = asyncio.get_event_loop()
        merged = await loop.run_in_executor(
            None,
            lambda: pl.merge_two_clouds(
                file_a, file_b, T,
                output_dir="data",
                voxel_size=cfg.get("voxel_size", 0.005)
            )
        )
        pcd = pl.load_pcd(merged)
        await self.ws.broadcast({
            "event": "merging_done",
            "merged_file": merged,
            "points": len(pcd.points),
            "calibration_is_stub": is_stub,
        })
        self.sm.set_data(merged_file=merged)
        return merged

    async def _step_process(self, input_file: str) -> dict:
        loop = asyncio.get_event_loop()
        ws_sync = self.ws.broadcast_sync

        def _do():
            return self._process_sync(input_file, ws_sync)

        return await loop.run_in_executor(None, _do)

    def _process_sync(self, input_file: str, emit) -> dict:
        """Синхронный pipeline. emit — функция для рассылки событий."""
        cfg = self.config
        pre = cfg["preprocessing"]
        plane = cfg["plane_removal"]
        db = cfg["dbscan"]
        icp_cfg = cfg["icp"]
        global_cfg = cfg.get("global_registration", {})

        pcd = pl.load_pcd(input_file)
        pcd = pl.clean_nan(pcd)
        n0 = len(pcd.points)

        pcd = pl.crop_roi(pcd, pre["roi"]["x"], pre["roi"]["y"], pre["roi"]["z"])
        emit({"event": "processing_step", "step": "crop_roi",
              "points_before": n0, "points_after": len(pcd.points)})

        if len(pcd.points) == 0:
            return {"status": "empty", "num_clusters": 0, "clusters": []}

        n = len(pcd.points)
        pcd = pl.voxel_downsample(pcd, pre["voxel_size"])
        emit({"event": "processing_step", "step": "voxel_downsample",
              "points_before": n, "points_after": len(pcd.points)})

        n = len(pcd.points)
        pcd = pl.statistical_filter(pcd, pre["nb_neighbors"], pre["std_ratio"])
        emit({"event": "processing_step", "step": "statistical_filter",
              "points_before": n, "points_after": len(pcd.points)})

        plane_model = None
        if plane.get("enabled", True):
            n = len(pcd.points)
            pcd, plane_model = pl.remove_plane(
                pcd,
                distance_threshold=plane["distance_threshold"],
                ransac_n=plane["ransac_n"],
                num_iterations=plane["num_iterations"],
            )
            emit({"event": "processing_step", "step": "ransac_plane",
                  "points_before": n, "points_after": len(pcd.points)})

        clusters = pl.cluster_dbscan(
            pcd,
            eps=db["eps"],
            min_points=db["min_points"],
            min_extent=db["min_extent"],
            max_extent=db["max_extent"],
        )
        emit({
            "event": "clusters_found",
            "num_clusters": len(clusters),
            "clusters": [{"id": i, "points": len(c.points)} for i, c in enumerate(clusters)]
        })

        # CAD модель
        cad_model = None
        cad_name = icp_cfg.get("cad_file")
        if cad_name:
            cad_path = Path("cad_models") / cad_name
            if cad_path.exists():
                cad_model = pl.load_pcd(str(cad_path))
            else:
                log.warning(f"CAD не найден: {cad_path}")

        Path("results/clusters").mkdir(parents=True, exist_ok=True)
        pl.save_clusters(clusters, "results/clusters")

        clusters_info = []
        for i, cluster in enumerate(clusters):
            info = pl.cluster_info(cluster, i)

            emit({"event": "pose_estimation_start",
                  "cluster_id": i,
                  "cad_model": cad_name if cad_model else None})

            pose = self._estimate_pose(cluster, cad_model, cad_name, icp_cfg, global_cfg, i)

            emit({
                "event": "pose_estimated",
                "cluster_id": i,
                "method": pose["method"],
                "fitness": pose.get("fitness"),
                "inlier_rmse": pose.get("inlier_rmse"),
                "position": pose["position"],
                "orientation": pose["orientation"],
                "global_fitness": pose.get("global_fitness"),
                "global_rmse": pose.get("global_rmse"),
            })

            # ICP визуализация (для всех методов, у которых есть cad_points_transformed)
            if pose.get("cad_points_transformed") is not None:
                cluster_pts = np.asarray(cluster.points)
                cad_pts = pose["cad_points_transformed"]

                def _ds_arr(arr, max_n=5000):
                    if len(arr) > max_n:
                        step = len(arr) // max_n
                        return arr[::step][:max_n]
                    return arr

                emit({
                    "event": "icp_visualization",
                    "cluster_id": i,
                    "cluster_points": _ds_arr(cluster_pts).tolist(),
                    "cad_points": _ds_arr(cad_pts).tolist(),
                    "cad_model_name": cad_name,
                })

            info["pose"] = {k: v for k, v in pose.items() if k != "cad_points_transformed"}
            clusters_info.append(info)

        annotated = pl.make_annotated_ply(pcd, clusters, "results")

        result = {
            "status": "ok",
            "input_file": input_file,
            "num_clusters": len(clusters),
            "clusters": clusters_info,
            "annotated_ply": annotated,
            "plane_model": plane_model,
        }
        pl.save_position_json(result, "results")
        return result

    def _estimate_pose(self, cluster, cad_model, cad_name, icp_cfg, global_cfg, cluster_id):
        """Выбор метода оценки позы. Изолировано чтобы не загромождать _process_sync."""
        if cad_model is None:
            return pl._obb_fallback(cluster, reason="no CAD")

        if global_cfg.get("enabled", False):
            try:
                return pl.run_global_then_icp(
                    cluster, cad_model, cad_name,
                    global_cfg=global_cfg,
                    icp_cfg=icp_cfg,
                )
            except Exception as e:
                log.warning(f"Global registration failed for cluster {cluster_id}: {e}. Fallback to ICP.")

        return pl.run_icp(
            cluster, cad_model,
            voxel_size=icp_cfg["voxel_size"],
            max_correspondence_distance=icp_cfg["max_correspondence_distance"],
            max_iterations=icp_cfg["max_iterations"],
            fitness_threshold=icp_cfg["fitness_threshold"],
        )

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