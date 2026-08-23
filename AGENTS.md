# Repository Guidelines

## Project Structure & Module Organization

This repository contains the Multi-Dynamic Deep Image Prior implementation for cardiac MRI reconstruction. Core Python modules live in `dip/`, including model definitions (`models.py`), reconstruction flow (`mdip.py`), MRI/data helpers (`mri.py`, `dataset.py`), FFT utilities, losses, evaluation, and plotting. The primary runnable workflows are notebooks: `M-DIP.ipynb` for M-DIP and L+S reconstruction, and `LR-DIP.ipynb` for LR-DIP comparison. Dependency sets are split between `requirements.txt` and `requirements_lrdip.txt`. Raw input data is expected under `data/`, which should not be committed.

## Build, Test, and Development Commands

Create a Python 3.12 environment, then install the standard dependencies:

```bash
pip install -r requirements.txt
```

Install LR-DIP comparison dependencies in a separate Python 3.12 environment:

```bash
pip install -r requirements_lrdip.txt
```

Run notebook workflows interactively with Jupyter:

```bash
jupyter notebook
```

Run M-DIP from the command line with papermill:

```bash
papermill M-DIP.ipynb temp.ipynb -p n_bases 15 -p filename <file.mrd>
```

## Coding Style & Naming Conventions

Use Python 3.12 syntax and keep module code in `dip/`. Follow the existing style: 4-space indentation, type hints for public class/function signatures where practical, snake_case for functions, variables, and module names, and PascalCase for classes. Prefer explicit imports and keep tensor shape comments concise when they clarify MRI or PyTorch dimensions. Existing code commonly uses single-quoted strings; match surrounding style when editing.

## Testing Guidelines

There is currently no dedicated automated test suite. For changes to reconstruction logic, validate by running the affected notebook path or a small papermill job against representative local data. For pure utility changes, add focused tests if introducing a test framework, or provide a minimal reproducible validation script/output in the pull request. Avoid committing generated notebooks, large outputs, or raw MRI data.

## Commit & Pull Request Guidelines

Recent history uses short, imperative, lower-case commit messages such as `update README` and `include phase padding in mask`. Keep commits focused and describe the behavior changed. Pull requests should include a concise summary, the commands or notebooks used for validation, relevant parameter changes, linked issues when applicable, and screenshots or metrics for reconstruction-quality changes.

## Security & Configuration Tips

Keep patient, scanner, and unpublished research data outside git. Store local paths and experiment outputs in ignored folders such as `data/` or a separate results directory. Do not add credentials, private dataset links, or machine-specific absolute paths to notebooks or source files.
