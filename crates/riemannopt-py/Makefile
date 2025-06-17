.PHONY: help install dev test test-fast lint format type-check docs clean build publish

# Default target
help:
	@echo "RiemannOpt Python Module - Available commands:"
	@echo ""
	@echo "  make install      Install the package in the current environment"
	@echo "  make dev          Install for development with all dependencies"
	@echo "  make test         Run all tests"
	@echo "  make test-fast    Run tests excluding slow ones"
	@echo "  make lint         Run code linters"
	@echo "  make format       Format code automatically"
	@echo "  make type-check   Run type checking with mypy"
	@echo "  make docs         Build documentation"
	@echo "  make clean        Clean build artifacts"
	@echo "  make build        Build the package"
	@echo "  make publish      Publish to PyPI (requires credentials)"

# Installation targets
install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing targets
test:
	@echo "Running all tests..."
	pytest tests/ -v --tb=short

test-fast:
	@echo "Running fast tests..."
	pytest tests/ -v -m "not slow" --tb=short

test-unit:
	@echo "Running unit tests..."
	pytest tests/unit/ -v --tb=short

test-integration:
	@echo "Running integration tests..."
	pytest tests/integration/ -v -m integration --tb=short

test-benchmark:
	@echo "Running benchmarks..."
	pytest tests/benchmarks/ -v -m benchmark --tb=short

test-coverage:
	@echo "Running tests with coverage..."
	pytest tests/ --cov=riemannopt --cov-report=html --cov-report=term

# Code quality targets
lint:
	@echo "Running linters..."
	ruff check python/
	flake8 python/ || true

format:
	@echo "Formatting code..."
	ruff format python/
	isort python/

type-check:
	@echo "Running type checker..."
	mypy python/riemannopt

check: lint type-check test-fast
	@echo "All checks passed!"

# Documentation targets
docs:
	@echo "Building documentation..."
	cd docs && make clean && make html

docs-serve:
	@echo "Serving documentation..."
	cd docs/_build/html && python -m http.server

# Build targets
build:
	@echo "Building package..."
	maturin build --release

build-debug:
	@echo "Building debug package..."
	maturin build

develop:
	@echo "Installing in development mode..."
	maturin develop

# Clean targets
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info
	rm -rf target/wheels target/release
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.so" -delete
	find . -type f -name "*.dylib" -delete

clean-docs:
	@echo "Cleaning documentation..."
	cd docs && make clean

clean-all: clean clean-docs
	@echo "All clean!"

# Publishing targets
publish-test:
	@echo "Publishing to TestPyPI..."
	maturin publish --repository testpypi

publish:
	@echo "Publishing to PyPI..."
	@read -p "Are you sure you want to publish to PyPI? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		maturin publish; \
	else \
		echo "Publish cancelled."; \
	fi

# Development helpers
watch-tests:
	@echo "Watching tests..."
	watchmedo shell-command --patterns="*.py;*.rs" --recursive --command="make test-fast" .

profile:
	@echo "Profiling optimization..."
	python -m cProfile -o profile.stats examples/basic/sphere_optimization.py
	python -m pstats profile.stats

# Environment setup
venv:
	@echo "Creating virtual environment..."
	python -m venv venv
	@echo "Activate with: source venv/bin/activate"

setup-pre-commit:
	@echo "Setting up pre-commit hooks..."
	pre-commit install
	pre-commit run --all-files

# CI helpers
ci-test:
	@echo "Running CI tests..."
	pytest tests/ -v --tb=short --junitxml=junit.xml

ci-lint:
	@echo "Running CI linting..."
	ruff check python/ --format=github
	mypy python/riemannopt --no-error-summary