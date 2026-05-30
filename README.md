# gram-diff-mimo

Research code for GRAM-DIFF-style MIMO channel estimation experiments.

The package contains small MIMO channel simulation utilities, least-squares and diffusion-guided channel estimators, adapters for bundled pretrained diffusion checkpoints from the Fesl paper (https://github.com/benediktfesl/Diffusion_channel_est), and example scripts for evaluations.

## Installation

Copy and paste this block from a checkout of the repository:

```bash
# Install Pixi if it is not already available:
curl -fsSL https://pixi.sh/install.sh | bash

# Start a fresh shell after installing Pixi, then run:
cd gram-diff-mimo
pixi install
```

If you already have Pixi installed, start at `cd gram-diff-mimo`.

The Pixi environment installs this package in editable mode and includes the scientific Python dependencies used by the examples and tests. PyTorch is configured through the CUDA wheel index in `pyproject.toml`; CPU-only systems may need a local PyTorch dependency adjustment before running pretrained-model examples.

## Evaluation

```bash
pixi run python examples/paper_structured_eval.py \
  --model-dir best_models_fesl_dm_paper/3gpp_path=3 \
  --device auto \
  --snr-min-db -15 \
  --snr-max-db 15 \
  --n-snr-points 31 \
  --n-data "200,2000" \
  --n-trials 4096 \
  --batch-size 8 \
  --output-csv results/structured_eval.csv \
  --plot-dir results/figures
```

## What Is Included

- `src/gram_diff_mimo/mimo`: channel generation, pilots, metrics, and channel estimators.
- `src/gram_diff_mimo/diffusion`: diffusion schedules, denoisers, sampling, and pretrained checkpoint loading.
- `examples`: evaluation harnesses and small development checks.
- `best_models_fesl_dm_paper`: bundled pretrained model directories containing
  `sim_params.json`, checkpoint files, and saved train/test result metadata.
- `tests`: focused regression tests for MIMO estimators and diffusion guidance.
- `notebooks`: exploratory figure-generation notebooks.

## Development

Run the test suite:

```bash
pixi run pytest
```

Run a single debug-testing file:

```bash
pixi run pytest tests/test_mimo_estimators.py
```
