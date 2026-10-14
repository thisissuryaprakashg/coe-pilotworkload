# Pilot Workload Analysis

This repository contains Python scripts for extracting multimodal physiological features and evaluating pilot workload models from the CogPilot dataset.

## Folder structure

- `src/data_pipeline/` - preprocessing and feature-generation scripts.
- `src/modeling/` - model training and evaluation experiments.
- `src/analysis/` - exploratory feature and TDA analysis.
- `docs/` - existing project notes and dataset documentation.
- `outputs/` - generated CSV and NPY data products.
- `experiments/` - accuracy-improvement experiments, preserved as-is.
- `fighter-pilot-synthetic/` - auxiliary project content, preserved as-is.

## Setup

Use Python 3. Install the packages used by the scripts, including `numpy`, `pandas`, `scipy`, and `scikit-learn`. Optional scripts may also require `pyriemann` or `ripser`.

Run commands from the repository root with the project virtual environment activated.

## Typical workflow

```text
python -m src.data_pipeline.new_data_pipeline
python -m src.data_pipeline.enhance_features
python -m src.modeling.improved_model
```

Run the other modules in `src/` when needed for alternative models, feature families, or exploratory analysis.

The raw dataset is expected separately, with `task-ils/`, `task-rest/`, and `starterCode/data_feats/devSubjsFeatMat.csv` available to the pipeline.
