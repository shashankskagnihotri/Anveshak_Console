# Testing

This repository includes unit and smoke tests for the package, CLI, retrieval helpers, logging, and runtime state handling.

## Install for Testing

From a fresh environment:

```bash
python -m pip install -U pip
python -m pip install -e .[test]
```

If you prefer the repo's existing dependency file:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Run the Test Suite

Run all tests:

```bash
pytest -q
```

Run only the CLI and packaging tests:

```bash
pytest tests/test_cli.py -q
```

Run the original smoke tests:

```bash
pytest tests/test_smoke.py -q
```

## Extra Sanity Check

Compile the Python package to catch syntax issues:

```bash
python -m compileall main.py anveshak tests
```

## What the Tests Cover

- package import surface
- installable CLI argument parsing
- `python -m anveshak`
- runtime progress aggregation
- run logging
- session locking
- steering constraints
- memory reset behavior
- parser behavior for PDFs
- model catalog backend inference

## What the Tests Do Not Cover

- downloading the full 122B default checkpoint
- full GPU inference benchmarks
- browser visual regression testing
- external website stability

Those remain manual validation areas.

## CI / GitHub Usage

The same commands work on local machines and GitHub runners:

```bash
python -m pip install -e .[test]
pytest -q
```

For GitHub-hosted CPU runners, remember that the tests validate package behavior and lightweight logic only. They do not prove that heavyweight local GPU inference works on a hosted runner.
