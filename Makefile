.PHONY: help install run stop restart clean clean-data lint check

VENV := venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PORT := 8000
APP := main.py

help:  ## показать список команд
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

$(VENV):  ## создать venv если нет
	python3 -m venv $(VENV)

install: $(VENV)  ## установить зависимости
	$(PIP) install -r requirements.txt

run:  ## запустить приложение (foreground)
	$(PY) $(APP)

stop:  ## остановить процесс на порту
	-@fuser -k $(PORT)/tcp 2>/dev/null || true
	@sleep 1

restart: stop run  ## перезапустить

clean:  ## удалить кэши python
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete

clean-data:  ## очистить сгенерированные облака и результаты
	rm -rf data/*.ply results/run_* results/*.json 2>/dev/null || true

check:  ## проверить что сервис отвечает
	@curl -fsS http://localhost:$(PORT)/viewer > /dev/null && \
		echo "OK: сервис отвечает" || echo "FAIL: нет ответа"