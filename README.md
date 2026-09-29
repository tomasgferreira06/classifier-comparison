# Classifier Comparison

Session 8 practical assignment — **Classifiers**.

The notebook `session8_classifiers.ipynb` covers:

- a short scikit-learn example with the **Iris** dataset;
- the main study on the scikit-learn **Digits** dataset (8×8 images): EDA, a stratified 60/20/20
  train/validation/test split, and **KNN**, **SVM** and **MLP** classifiers (baseline → tuning on validation → final
  model on train + validation → single test evaluation);
- the extra work on **MNIST** (28×28 images, 70 000 samples): the same classifiers, an experimental analysis of **KNN
  scalability**, and a simple **CNN** in PyTorch.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync
```

On Windows/Linux, PyTorch is installed from the CUDA 13.0 wheel index (see `pyproject.toml`). The CNN uses an NVIDIA GPU
when one is available and falls back to the CPU otherwise.

## Running the notebook

Open `session8_classifiers.ipynb` in VS Code (or any Jupyter front-end) using the `.venv` kernel created by `uv sync`,
and run all cells. To execute it headless:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 session8_classifiers.ipynb
```

- MNIST is downloaded automatically by `torchvision` into `data/` on the first run.
- Trained models are saved in `checkpoints/` and reused on later runs. Set `FORCE_RETRAIN = True` (section 2.2) to
  retrain everything. A run from existing checkpoints takes a few minutes. Without checkpoints, training everything is
  much longer: the MNIST SVM alone needs several minutes on a CPU.
- `data/` and `checkpoints/` are not tracked by git.

## Project structure

| Path | Content |
|---|---|
| `session8_classifiers.ipynb` | the complete study (code, results and discussion) |
| `Session-8-Practical-Assignment-Classifiers.pdf` | assignment brief |
| `SESSION8_AUDIT.md` | audit of the notebook (methodology, results, code quality) |
| `checkpoints/` | saved models, tuning results and split indices (generated) |
| `data/` | MNIST files (downloaded) |
| `pyproject.toml`, `uv.lock` | dependencies |
