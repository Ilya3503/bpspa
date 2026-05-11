"""
WebSocket менеджер: держит список подключённых клиентов и рассылает им события.
"""
import asyncio
import json
import logging
import time
from typing import Set
from fastapi import WebSocket

log = logging.getLogger(__name__)


class WSManager:
    def __init__(self):
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)
        log.info(f"WS connected. Total clients: {len(self._clients)}")

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            self._clients.discard(ws)
        log.info(f"WS disconnected. Total clients: {len(self._clients)}")

    async def broadcast(self, message: dict):
        """
        Отправляет JSON-сообщение всем подключённым клиентам.
        Тихо удаляет упавшие соединения.
        """
        if "timestamp" not in message:
            message["timestamp"] = time.time()
        payload = json.dumps(message, ensure_ascii=False)

        async with self._lock:
            clients = list(self._clients)

        dead = []
        for ws in clients:
            try:
                await ws.send_text(payload)
            except Exception as e:
                log.warning(f"WS send failed: {e}")
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)

    def broadcast_sync(self, message: dict):
        """
        Синхронная обёртка для вызова из не-asyncio кода (например из threadpool).
        Планирует broadcast в event loop.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(self.broadcast(message), loop)
            else:
                loop.run_until_complete(self.broadcast(message))
        except RuntimeError:
            # event loop отсутствует — просто логируем
            log.debug(f"No event loop, skip broadcast: {message.get('event')}")