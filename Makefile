PYTHON ?= python

.PHONY: test
test:
	pytest -v

.PHONY: autoformat
autoformat:
	ruff format .
	ruff check . --fix

.PHONY: lint
lint:
	$(PYTHON) -m flake8 .
	$(PYTHON) -m black . --check
	# Note that Bandit will look for .bandit file only if it's invoked with -r option.
	$(PYTHON) -m bandit -c pyproject.toml -r . --exit-zero

.PHONY: clean
clean:
	@find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

.PHONY: conda-osx-64.lock
conda-osx-64.lock:
	CONDA_SUBDIR=osx-64 conda-lock -f conda.yaml -p osx-64
	CONDA_SUBDIR=osx-64 conda-lock render -p osx-64

.PHONY: conda-linux-64.lock
conda-linux-64.lock:
	conda-lock -f conda.yaml -p linux-64
	conda-lock render -p linux-64

conda-lock.yml: conda-osx-64.lock conda-linux-64.lock

# Clear the cache and rebuild the lock.
.PHONY: poetry.lock
poetry.lock:
	poetry cache clear --all .
	poetry lock