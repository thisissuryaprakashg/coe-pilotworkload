# Data Pipeline

This folder contains preprocessing and feature-generation scripts for the pilot workload dataset.

## Included scripts

- `new_data_pipeline.py` — per-window physiological feature construction
- `enhance_features.py` — subject normalization and temporal enhancement
- `step1_signature.py` — signature-based eye/pupil features
- `step2_hmm_smoothing.py` — HMM smoothing model variant
- `step3_riemannian.py` — covariance and tangent-space features
- `step4_plsda.py` — PLS-DA model variant
- `step5_gmm_phenotyping.py` — GMM-based phenotyping model variant
- `tda_features.py` — topology-driven feature extraction

## Typical run order

1. `python new_data_pipeline.py`
2. `python enhance_features.py`
3. Additional analysis or modeling scripts as needed
