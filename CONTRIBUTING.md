# Contributing

Contributions are welcome. Keep changes focused, add or update tests for behavioral changes, and update the public documentation when the user-facing API changes.

## Local Setup

Create a virtual environment and install the project in editable mode with development dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Test And Coverage

Run the full test suite:

```bash
python -m unittest discover -s tests -v
```

Generate a local coverage report:

```bash
python -m coverage run -m unittest discover -s tests -v
python -m coverage report
python -m coverage xml
```

The default coverage target tracks the core package. The optional Spark backend
is validated separately because its runtime integration test is gated behind an
environment flag and extra dependencies.

The optional Spark integration test is disabled by default. To run it locally, install the Spark extra and set the integration-test flag:

```bash
python -m pip install -e .[dev,spark]
GLM_FACTOR_OPTIMIZER_RUN_SPARK_TESTS=1 python -m unittest tests.test_spark_backend -v
```

## Packaging Checks

Before opening a release-oriented pull request, verify the package builds cleanly:

```bash
python -m build
python -m twine check dist/*
```

## Pull Requests

- Keep each pull request scoped to one change set.
- Add regression tests for bug fixes and new features.
- Prefer neutral, domain-free examples in public docs and examples.
- Avoid committing generated artifacts, virtual environments, or local secrets.

## Release Process

Create and push a version tag like `v0.1.0` after the main branch is ready. The release workflow builds the distributions, runs a final Twine check, and publishes to PyPI via GitHub trusted publishing once the repository is configured as a trusted publisher in PyPI.