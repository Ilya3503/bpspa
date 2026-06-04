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
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.state_machine import StateMachine
from core.orchestrator import Orchestrator
from hardware.camera import RealSenseCamera
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


# ---------- сборка приложения ----------

def create_app() -> FastAPI:
    config = load_config()

    app = FastAPI(title="Bin-Picking System", version="1.0")

    # ---- singletons ----
    ws_manager = WSManager()
    camera = RealSenseCamera(config)
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

    return app