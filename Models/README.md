# Model artifacts

`pitchiq_v25.joblib` is the active Version 2.5 artifact loaded by `src.engine.production.load_or_train_model()` and served by the FastAPI application (it retrains deterministically if the data or library versions change; saves are atomic).

`historical_predictor_v2.joblib` is the archived Version 2.4 model, kept for the like-for-like comparison and for `python -m src.train --legacy`.

The following files are retained from an earlier experimental pipeline for provenance only; the current API does not load them:

- `final_model.joblib`
- `preprocessing_pipeline.joblib`
- `feature_config.json`
- `model_metadata.json`

Their metrics and feature definitions refer to a different model generation and must not be presented as current results. Active-model metrics are in `reports/v25/` (v2.5) and `reports/model_evaluation_v2.json` (archived v2.4).
