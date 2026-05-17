"""
Конечный автомат системы bin-picking.
Один словарь переходов, без классов состояний, без наследования.

Поддерживает два режима работы:
- двух-ракурсный: IDLE → CAPTURING_VIEW1 → WAITING_VIEW2 → CAPTURING_VIEW2 → MERGING → PROCESSING → ...
- одно-ракурсный: IDLE → CAPTURING_SINGLE → PROCESSING → ...

Развилка по режимам делается в orchestrator на основе config.capture.n_views.
"""
from enum import Enum
import logging

log = logging.getLogger(__name__)


class State(str, Enum):
    IDLE = "IDLE"
    CAPTURING_VIEW1 = "CAPTURING_VIEW1"
    WAITING_VIEW2 = "WAITING_VIEW2"
    CAPTURING_VIEW2 = "CAPTURING_VIEW2"
    CAPTURING_SINGLE = "CAPTURING_SINGLE"   # для режима одного ракурса
    MERGING = "MERGING"
    PROCESSING = "PROCESSING"
    EXECUTING = "EXECUTING"
    DONE = "DONE"
    ERROR = "ERROR"


# Явные переходы по командам пользователя.
# Автоматические переходы делает orchestrator после завершения работы.
# "start" из IDLE ведёт в неопределённое состояние — orchestrator сам выбирает
# куда переходить (CAPTURING_VIEW1 или CAPTURING_SINGLE) на основе n_views.
USER_TRANSITIONS = {
    State.WAITING_VIEW2: {"next_view": State.CAPTURING_VIEW2},
    State.DONE:          {"reset": State.IDLE},
    State.ERROR:         {"reset": State.IDLE},
}

STOP_COMMAND = "stop"
START_COMMAND = "start"   # обрабатывается специально (см. trigger_start)


class StateMachine:
    """
    Источник истины о текущем состоянии системы.
    Все остальные части либо читают .state, либо вызывают .trigger()/.trigger_start().
    """

    def __init__(self):
        self._state: State = State.IDLE
        self._data: dict = {}

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
        if action in (STOP_COMMAND, START_COMMAND):
            return True if action == STOP_COMMAND else (self._state == State.IDLE)
        return action in USER_TRANSITIONS.get(self._state, {})

    def trigger_start(self, n_views: int) -> State:
        """
        Команда start. Переводит в CAPTURING_VIEW1 (двух-ракурсный режим)
        или CAPTURING_SINGLE (одно-ракурсный режим) в зависимости от n_views.
        Разрешена только из IDLE.
        """
        if self._state != State.IDLE:
            raise ValueError(
                f"Команда 'start' недопустима в состоянии {self._state.value}. "
                f"Разрешена только из IDLE."
            )
        new_state = State.CAPTURING_VIEW1 if n_views >= 2 else State.CAPTURING_SINGLE
        log.info(f"USER: {self._state} → {new_state} (start, n_views={n_views})")
        self._state = new_state
        return self._state

    def trigger(self, action: str) -> State:
        """
        Применяет пользовательскую команду (next_view, reset, stop).
        Для команды start используй trigger_start().
        """
        if action == STOP_COMMAND:
            log.info(f"STOP: переход {self._state} → IDLE")
            self._state = State.IDLE
            self.clear_data()
            return self._state

        if action == START_COMMAND:
            raise ValueError("Для start используйте trigger_start(n_views)")

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
        """Автоматический переход (вызывается orchestrator-ом после завершения шага)."""
        log.info(f"AUTO: {self._state} → {new_state}")
        self._state = new_state

    def fail(self, error_msg: str):
        log.error(f"FAIL: {self._state} → ERROR ({error_msg})")
        self._state = State.ERROR
        self._data["last_error"] = error_msg