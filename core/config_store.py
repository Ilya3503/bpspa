"""
Конфиг = config.default.yaml (эталон, в git) + config.local.yaml (оверрайды, не в git).
Эффективный конфиг = default, поверх которого положен local (глубокий merge).
Приложение правит local; снимок отдаёт эффективный конфиг с датой и пометкой.
"""
import logging
from datetime import datetime
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

DEFAULT_PATH = Path("config.default.yaml")
LOCAL_PATH = Path("config.local.yaml")


def _deep_merge(base: dict, over: dict) -> dict:
    """over поверх base, рекурсивно. Возвращает новый dict, аргументы не мутирует."""
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_effective() -> dict:
    """Эффективный конфиг: default + local. Его читает вся система."""
    return _deep_merge(_read_yaml(DEFAULT_PATH), _read_yaml(LOCAL_PATH))


def save_effective(cfg: dict) -> None:
    """Пишет ВЕСЬ конфиг в local (local становится полным рабочим конфигом).
    default не трогаем никогда."""
    with open(LOCAL_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    log.info("config.local.yaml обновлён")


def reset_local() -> None:
    """Удаляет оверрайды — возврат к эталону."""
    if LOCAL_PATH.exists():
        LOCAL_PATH.unlink()
        log.info("config.local.yaml удалён — сброс к default")


def snapshot_bytes(cfg: dict, note: str = "") -> tuple[str, bytes]:
    """Готовит снимок для скачивания: (имя_файла, содержимое).
    Имя: config_ДАТА_ВРЕМЯ[_ПОМЕТКА].yaml"""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in note).strip("_")
    name = f"config_{ts}_{safe}.yaml" if safe else f"config_{ts}.yaml"
    body = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False).encode("utf-8")
    return name, body