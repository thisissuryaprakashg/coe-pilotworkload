# Modeling

This folder contains workload model training and evaluation scripts.

- `improved_model.py` - primary enhanced-feature model evaluation.
- `advanced_model.py` - additional subject-normalization model.
- `nonlinear_model.py` - nonlinear model comparisons.
- `step2_hmm_smoothing.py`, `step4_plsda.py`, and `step5_gmm_phenotyping.py` - model variants built around the feature matrix.

Run the corresponding module from the repository root, for example `python -m src.modeling.improved_model`.
