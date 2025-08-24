PY=python

setup:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

reproduce:
	$(PY) -m ws_lite.cli reproduce

ablation:
	$(PY) -m ws_lite.cli ablation

test:
	pytest -q
