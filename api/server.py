"""
FastAPI сервер: эндпоинты + WebSocket.
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core import config_store
from core.state_machine import StateMachine
from core.orchestrator import Orchestrator
from hardware.camera_switch import CameraSwitch
from api.ws_manager import WSManager

log = logging.getLogger(__name__)

CONFIG_PATH = Path("config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


# ---------- модели ----------

class CommandRequest(BaseModel):
    action: str   # start | next_view | reset | stop


class CADSelectRequest(BaseModel):
    name: str

class CameraSelectRequest(BaseModel):
    backend: str   # realsense | orbbec


# ---------- сборка приложения ----------

def create_app() -> FastAPI:
    config = config_store.load_effective()

    app = FastAPI(title="Bin-Picking System", version="1.0")

    # ---- singletons ----
    ws_manager = WSManager()
    camera = CameraSwitch(config)
    sm = StateMachine()
    orch = Orchestrator(sm, ws_manager, camera, config)

    app.state.config = config
    app.state.ws = ws_manager
    app.state.camera = camera
    app.state.sm = sm
    app.state.orch = orch
    app.state.video_task = None

    # ---- lifecycle ----

    @app.on_event("startup")
    async def _startup():
        log.info("=== STARTUP ===")
        camera.start()
        app.state.video_task = asyncio.create_task(orch.video_stream_task())
        log.info("=== READY ===")

    @app.on_event("shutdown")
    async def _shutdown():
        log.info("=== SHUTDOWN ===")
        if app.state.video_task:
            app.state.video_task.cancel()
        camera.stop()

    # ---- статика UI ----
    ui_dir = Path("ui")
    if ui_dir.exists():
        app.mount("/static", StaticFiles(directory=str(ui_dir)), name="static")

    # ---- эндпоинты ----

    @app.get("/")
    async def root():
        return {"service": "bin-picking", "state": sm.state.value}

    @app.get("/viewer", response_class=HTMLResponse)
    async def viewer():
        index = ui_dir / "index.html"
        if not index.exists():
            raise HTTPException(404, "ui/index.html не найден")
        return FileResponse(str(index))

    @app.get("/state")
    async def get_state():
        return {"state": sm.state.value, "data": sm.data}

    @app.post("/command")
    async def command(req: CommandRequest):
        try:
            await orch.handle_command(req.action)
            return {"ok": True, "state": sm.state.value}
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            log.exception("command failed")
            raise HTTPException(500, str(e))

    @app.get("/result")
    async def get_result():
        p = Path("results/position.json")
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    @app.get("/camera")
    async def camera_get():
        return {"active": camera.active_key, "available": camera.available()}

    @app.post("/camera/select")
    async def camera_select(req: CameraSelectRequest):
        # переключаемся только в покое, чтобы не оборвать активный цикл
        if sm.state.value != "IDLE":
            raise HTTPException(409, "Переключать камеру можно только в состоянии IDLE")
        try:
            active = await asyncio.get_event_loop().run_in_executor(
                None, camera.switch, req.backend
            )
        except ValueError as e:
            raise HTTPException(400, str(e))
        await ws_manager.broadcast({"event": "camera_changed", "active": active})
        return {"ok": True, "active": active}

    @app.get("/cad_models")
    async def cad_models():
        d = Path("cad_models")
        if not d.exists():
            return {"models": [], "selected": app.state.config["icp"].get("cad_file")}
        models = sorted([p.name for p in d.glob("*.ply")])
        return {"models": models, "selected": app.state.config["icp"].get("cad_file")}

    @app.post("/cad_models/select")
    async def cad_select(req: CADSelectRequest):
        d = Path("cad_models") / req.name
        if not d.exists():
            raise HTTPException(404, f"Модель не найдена: {req.name}")
        app.state.config["icp"]["cad_file"] = req.name
        save_config(app.state.config)
        await ws_manager.broadcast({"event": "cad_selected", "name": req.name})
        return {"ok": True, "selected": req.name}

    @app.get("/config")
    async def config_get():
        return app.state.config

    @app.post("/config")
    async def config_set(patch: dict = Body(...)):
        # неглубокий merge
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    deep_update(d[k], v)
                else:
                    d[k] = v
            return d
        deep_update(app.state.config, patch)
        save_config(app.state.config)
        return app.state.config

    @app.get("/files/{folder}")
    async def list_files(folder: str):
        if folder not in ("data", "results", "cad_models", "results/clusters"):
            raise HTTPException(400, "недопустимая папка")
        p = Path(folder)
        if not p.exists():
            return {"files": []}
        pattern = "**/*.ply" if folder == "results" else "*.ply"
        files = sorted(str(f.relative_to(p)) for f in p.glob(pattern))
        return {"files": files}

    @app.get("/file/{path:path}")
    async def get_file(path: str):
        # отдаём только из data/, results/, cad_models/
        allowed_roots = ("data", "results", "cad_models")
        p = Path(path)
        if not any(str(p).startswith(r) for r in allowed_roots):
            raise HTTPException(400, "доступ запрещён")
        if not p.exists():
            raise HTTPException(404, "файл не найден")
        return FileResponse(str(p))

    @app.websocket("/events")
    async def ws_events(ws: WebSocket):
        await ws_manager.connect(ws)
        try:
            # шлём текущее состояние сразу при подключении
            await ws.send_text(json.dumps({
                "event": "state_changed",
                "state": sm.state.value,
            }))
            while True:
                # держим соединение, игнорируем входящие
                await ws.receive_text()
        except WebSocketDisconnect:
            await ws_manager.disconnect(ws)
        except Exception as e:
            log.warning(f"WS error: {e}")
            await ws_manager.disconnect(ws)







    class ConfigApply(BaseModel):
        config: dict

    class Snapshot(BaseModel):
        config: dict
        note: str = ""

    @app.get("/config_ui", response_class=HTMLResponse)
    async def config_ui():
        f = ui_dir / "config.html"
        if not f.exists():
            raise HTTPException(404, "ui/config.html не найден")
        return FileResponse(str(f))

    @app.get("/config/effective")
    async def config_effective():
        return app.state.config

    @app.post("/config/apply")
    async def config_apply(req: ConfigApply):
        config_store.save_effective(req.config)
        app.state.config.clear()
        app.state.config.update(req.config)  # живой конфиг обновлён на месте
        await ws_manager.broadcast({"event": "config_changed"})
        return {"ok": True}

    @app.post("/config/reset")
    async def config_reset():
        config_store.reset_local()
        new_cfg = config_store.load_effective()
        app.state.config.clear()
        app.state.config.update(new_cfg)
        await ws_manager.broadcast({"event": "config_changed"})
        return {"ok": True, "config": new_cfg}

    @app.post("/config/snapshot")
    async def config_snapshot(req: Snapshot):
        name, body = config_store.snapshot_bytes(req.config, req.note)
        return Response(
            content=body, media_type="application/x-yaml",
            headers={"Content-Disposition": f'attachment; filename="{name}"'})

    return app