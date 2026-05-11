"""
Конечный автомат системы bin-picking.
Один словарь переходов, без классов состояний, без наследования.
"""
from enum import Enum
from typing import Optional
import logging

log = logging.getLogger(__name__)


class State(str, Enum):
    IDLE = "IDLE"
    CAPTURING_VIEW1 = "CAPTURING_VIEW1"
    WAITING_VIEW2 = "WAITING_VIEW2"
    CAPTURING_VIEW2 = "CAPTURING_VIEW2"
    MERGING = "MERGING"
    PROCESSING = "PROCESSING"
    EXECUTING = "EXECUTING"
    DONE = "DONE"
    ERROR = "ERROR"


# Явные переходы по командам пользователя.
# Автоматические переходы делает orchestrator после завершения работы.
USER_TRANSITIONS = {
    State.IDLE:          {"start": State.CAPTURING_VIEW1},
    State.WAITING_VIEW2: {"next_view": State.CAPTURING_VIEW2},
    State.DONE:          {"reset": State.IDLE},
    State.ERROR:         {"reset": State.IDLE},
}

# Команда "stop" разрешена из любого состояния и ведёт в IDLE.
STOP_COMMAND = "stop"


class StateMachine:
    """
    Источник истины о текущем состоянии системы.
    Все остальные части либо читают .state, либо вызывают .trigger().
    """

    def __init__(self):
        self._state: State = State.IDLE
        self._data: dict = {}   # произвольные данные текущего цикла (пути к файлам, результаты)

    @property
    def state(self) -> State:
        return self._state

    @property
    def data(self) -> dict:
        return dict(self._data)

    def set_data(self, **kwargs):
        self._data.update(kwargs)

    def clear_data(self):
        self._data.clear()

    def can_trigger(self, action: str) -> bool:
        if action == STOP_COMMAND:
            return True
        return action in USER_TRANSITIONS.get(self._state, {})

    def trigger(self, action: str) -> State:
        """
        Применяет пользовательскую команду. Возвращает новое состояние.
        Бросает ValueError если переход недопустим.
        """
        if action == STOP_COMMAND:
            log.info(f"STOP: переход {self._state} → IDLE")
            self._state = State.IDLE
            self.clear_data()
            return self._state

        allowed = USER_TRANSITIONS.get(self._state, {})
        if action not in allowed:
            raise ValueError(
                f"Команда '{action}' недопустима в состоянии {self._state.value}. "
                f"Разрешены: {list(allowed.keys()) + [STOP_COMMAND]}"
            )

        new_state = allowed[action]
        log.info(f"USER: {self._state} → {new_state} (action={action})")
        self._state = new_state
        return self._state

    def advance(self, new_state: State):
        """
        Автоматический переход (вызывается orchestrator-ом после завершения шага).
        """
        log.info(f"AUTO: {self._state} → {new_state}")
        self._state = new_state

    def fail(self, error_msg: str):
        log.error(f"FAIL: {self._state} → ERROR ({error_msg})")
        self._state = State.ERROR
        self._data["last_error"] = error_msg