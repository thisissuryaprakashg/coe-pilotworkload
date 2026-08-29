# Data Pipeline

These scripts transform the raw CogPilot recordings into modeling-ready tables.

Typical order:

1. `build_cleaned_dataset.py` creates `data/processed/cleaned_multimodal_dataset.csv`.
2. `assemble_master_table.py` adds resting baselines, delta features, and official eye features.
3. The final table is written to `data/processed/master_feature_matrix.csv`.

Set `COGPILOT_DATASET_ROOT` to the extracted CogPilot dataset directory. If unset, the default is `data/raw/cogpilot/`.
