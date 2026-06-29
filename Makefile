.PHONY: help install run start stop restart status logs clean clean-data check

VENV := venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PORT := 8000
APP := main.py
SERVICE := bpspa

help:  ## показать список команд
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

$(VENV):  ## создать venv если нет
	python3 -m venv $(VENV)

install: $(VENV)  ## установить зависимости
	$(PIP) install -r requirements.txt

run:  ## запустить вручную в foreground (для отладки; сначала сделай make stop)
	$(PY) $(APP)

start:  ## запустить сервис
	sudo systemctl start $(SERVICE)

stop:  ## остановить сервис
	sudo systemctl stop $(SERVICE)

restart:  ## перезапустить сервис
	sudo systemctl restart $(SERVICE)

status:  ## статус сервиса
	systemctl status $(SERVICE) --no-pager

logs:  ## live-логи сервиса
	journalctl -u $(SERVICE) -f

clean:  ## удалить кэши python
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete

clean-data:  ## очистить сгенерированные облака и результаты
	rm -rf data/*.ply results/run_* results/*.json 2>/dev/null || true

check:  ## проверить что сервис отвечает
	@curl -fsS http://localhost:$(PORT)/viewer > /dev/null && \
		echo "OK: сервис отвечает" || echo "FAIL: нет ответа"