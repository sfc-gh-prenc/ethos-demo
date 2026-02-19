SHELL := /bin/bash
.PHONY: dev clean help

help:
	@echo "Available targets:"
	@echo "  dev    - Install package in dev mode (uses uv, fallback to pip)"
	@echo "  clean  - Remove __pycache__, build artifacts, and other generated files"

dev:
	@if command -v uv >/dev/null 2>&1; then \
		echo "Installing with uv..."; \
		uv pip install -e .[dev]; \
	else \
		echo -e "\033[33mWarning: uv not found, falling back to pip.\033[0m"; \
		pip install -e .[dev]; \
	fi

clean:
	@echo "Removing __pycache__ directories..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Removing .pyc and .pyo files..."
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "Removing .egg-info directories..."
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Removing build directories..."
	@rm -rf build/ dist/ .eggs/ 2>/dev/null || true
	@echo "Removing pytest cache..."
	@rm -rf .pytest_cache/ 2>/dev/null || true
	@echo "Removing ruff cache..."
	@rm -rf .ruff_cache/ 2>/dev/null || true
	@echo "Done."
