# Modeling

These scripts analyze the master feature matrix and evaluate workload models.

Recommended order:

1. `feature_screening.py`
2. `math_model_ordinal_regression.py`
3. `train_and_compare_models.py`
4. `advanced_workload_modeling.py`
5. `pilot_personalized_delta_math_model.py` when the personalized analysis is needed

Input data is read from `data/processed/`. Results are written to `reports/`.
