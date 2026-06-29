.PHONY: help install run start stop restart status logs clean clean-data check

VENV := venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PORT := 8000
APP := main.py
SERVICE := bpspa

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

$(VENV):
	python3 -m venv $(VENV)

install: $(VENV)
	$(PIP) install -r requirements.txt

run:
	$(PY) $(APP)

start:
	sudo systemctl start $(SERVICE)

stop:
	sudo systemctl stop $(SERVICE)

restart:
	sudo systemctl restart $(SERVICE)

status:
	systemctl status $(SERVICE) --no-pager

logs:
	journalctl -u $(SERVICE) -f

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete

clean-data:
	rm -rf data/*.ply results/run_* results/*.json 2>/dev/null || true

check:
	@curl -fsS http://localhost:$(PORT)/viewer > /dev/null && \
		echo "OK: сервис отвечает" || echo "FAIL: нет ответа"