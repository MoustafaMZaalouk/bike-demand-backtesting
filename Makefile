.PHONY: train eval test lint smoke reproduce
PYTHON ?= python

train:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $(PYTHON) -m bike_backtest.run train

eval:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $(PYTHON) -m bike_backtest.run eval
	$(PYTHON) -m bike_backtest.plot
	$(PYTHON) -m bike_backtest.report

test:
	OMP_NUM_THREADS=1 $(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

smoke:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $(PYTHON) -m bike_backtest.run all --smoke --output /tmp/bike-smoke

reproduce: train eval test lint
