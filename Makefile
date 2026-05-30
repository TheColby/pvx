.PHONY: install install-dev lint format typecheck test coverage coverage-full docs docs-generated docs-html man pre-commit bench depsync alpha-check

install:
	python3 -m pip install -r requirements.txt
	python3 -m pip install -e .
	uv run python scripts/scripts_install_man_pages.py

install-dev:
	python3 -m pip install -r requirements.txt
	python3 -m pip install -e ".[dev]"
	python3 -m pip install pre-commit
	uv run python scripts/scripts_install_man_pages.py

lint:
	uv run ruff check src/pvx src/pvxalgorithms scripts tests

format:
	uv run ruff format src/pvx src/pvxalgorithms scripts tests

typecheck:
	uv run python -m mypy src/pvx/core/attribution.py src/pvx/core/control_bus.py src/pvx/core/streaming.py src/pvx/core/voc_console.py src/pvx/core/voc_parser.py src/pvx/core/voc_profiles.py src/pvx/voc_cli.py src/pvx/cli/pvxvoc.py src/pvxalgorithms scripts/scripts_alpha_check.py scripts/scripts_apply_attribution.py scripts/scripts_sync_homebrew_tap_formula.py

depsync:
	uv run python scripts/scripts_check_dependency_sync.py

test:
	uv run python -m unittest discover -s tests -p "test_*.py"

coverage:
	uv run python -m coverage run -m unittest discover -s tests -p "test_*.py"
	uv run python -m coverage report --include='src/pvx/core/voc_console.py,src/pvx/core/voc_parser.py,src/pvx/voc_cli.py,src/pvx/cli/pvxvoc.py,scripts/scripts_sync_homebrew_tap_formula.py,src/pvxalgorithms/*' --fail-under=100

coverage-full:
	uv run python -m coverage run -m unittest discover -s tests -p "test_*.py"
	uv run python -m coverage report

docs:
	uv run python scripts/scripts_generate_python_docs.py
	uv run python scripts/scripts_generate_theory_docs.py
	uv run python scripts/scripts_generate_docs_extras.py

docs-generated: docs docs-html man

docs-html:
	uv run python scripts/scripts_generate_html_docs.py

man:
	uv run python scripts/scripts_install_man_pages.py

pre-commit:
	pre-commit run --all-files

alpha-check:
	uv run python scripts/scripts_alpha_check.py

bench:
	uv run python benchmarks/run_bench.py --quick --out-dir benchmarks/out --strict-corpus --determinism-runs 2
