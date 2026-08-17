.PHONY: help install run start stop restart status logs clean clean-data check \
        dev-start dev-stop dev-restart dev-status dev-logs dev-check prod-start

VENV := venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PORT := 8000
PORT_DEV := 8001
APP := main.py
SERVICE := bpspa
SERVICE_DEV := bpspa-dev

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	   awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

$(VENV):
	python3 -m venv $(VENV)

install: $(VENV)  ## поставить зависимости
	$(PIP) install -r requirements.txt

run:  ## запуск с логами в терминале
	$(PY) $(APP)


# --- prod (main, порт 8000) ---
start:  ## запустить prod в фоне
	sudo systemctl start $(SERVICE)

stop:  ## остановить prod
	sudo systemctl stop $(SERVICE)

restart:  ## перезапустить prod
	sudo systemctl restart $(SERVICE)

status:  ## состояние prod
	systemctl status $(SERVICE) --no-pager

logs:  ## живые логи prod
	journalctl -u $(SERVICE) -f

check:  ## отвечает ли prod
	@curl -fsS http://localhost:$(PORT)/viewer > /dev/null && \
	   echo "OK: prod отвечает" || echo "FAIL: нет ответа"

prod-start:  ## запустить prod, погасив dev (камера одна!)
	sudo systemctl stop $(SERVICE_DEV) || true
	sudo systemctl start $(SERVICE)
	@echo "PROD на :$(PORT), DEV остановлен"

# --- dev (develop, порт 8001) ---
dev-start:  ## запустить dev, погасив prod (камера одна!)
	sudo systemctl stop $(SERVICE) || true
	sudo systemctl start $(SERVICE_DEV)
	@echo "DEV на :$(PORT_DEV), PROD остановлен"

dev-stop:  ## остановить dev
	sudo systemctl stop $(SERVICE_DEV)

dev-restart:  ## перезапустить dev
	sudo systemctl restart $(SERVICE_DEV)

dev-status:  ## состояние dev
	systemctl status $(SERVICE_DEV) --no-pager

dev-logs:  ## живые логи dev
	journalctl -u $(SERVICE_DEV) -f

dev-check:  ## отвечает ли dev
	@curl -fsS http://localhost:$(PORT_DEV)/state > /dev/null && \
	   echo "OK: dev отвечает" || echo "FAIL: dev молчит"

clean:  ## удалить __pycache__ и .pyc
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete

clean-data:  ## удалить снимки и результаты прогонов
	rm -rf data/*.ply results/run_* results/*.json 2>/dev/null || true