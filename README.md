# XQP: KV-Cache Saliency Prediction

XQP is a Python codebase for training and evaluating lightweight KV-cache
saliency predictors for LLM serving experiments. The repository keeps source
code, tests, configuration, benchmark helpers, and small deployable predictor
checkpoints. Paper drafts, review notes, logs, traces, and generated result
tables are intentionally kept out of the repository.

## Repository Layout

```text
xqp/                 Core predictor library and CLI entry points.
experiments/         Reproducible experiment, analysis, and plotting drivers.
experiments/predictors/
                     Small JSON predictor checkpoints kept for demos/evaluation.
scripts/             Trace collection, training, benchmark, and setup helpers.
configs/             Model configuration files.
benchmark/           Benchmark protocol, reference model, and submission tools.
tests/               CPU test suite for the library and experiment drivers.
```

Generated artifacts are ignored by Git:

```text
experiments/results/
experiments/logs/
experiments/traces/
experiments/**/results/
```

## Install

```bash
python -m pip install -e .[test]
```

For the full local conda environment:

```bash
conda env create -f environment.yml
conda activate csp-llm
```

## Test

```bash
pytest -q
```

## Common Entry Points

```bash
xqp-train --help
xqp-eval --help
python experiments/run_icdm_full.py --help
python experiments/generate_figures.py --help
```

Trace collection and larger GPU runs are available under `scripts/` and
`experiments/`. They write outputs to ignored artifact directories by default.

## License

Apache 2.0.
