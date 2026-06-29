"""
Точка входа.
"""
import logging
import uvicorn
import yaml

from api.server import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


app = create_app()

if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    s = cfg["server"]
    uvicorn.run(app, host=s["host"], port=s["port"], log_level="info")
