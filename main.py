"""
Точка входа.
"""
import logging
import uvicorn

from core import config_store
from api.server import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = create_app()

if __name__ == "__main__":
    cfg = config_store.load_effective()
    s = cfg["server"]
    uvicorn.run(app, host=s["host"], port=s["port"], log_level="info")