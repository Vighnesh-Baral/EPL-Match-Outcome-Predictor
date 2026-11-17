# Model artifacts

`historical_predictor_v2.joblib` is the active Version 2.4 artifact loaded by `src.historical_model.load_or_train_model()` and served by the FastAPI application.

The following files are retained from the previous experimental pipeline for provenance only; the current API does not load them:

- `final_model.joblib`
- `preprocessing_pipeline.joblib`
- `feature_config.json`
- `model_metadata.json`

Their metrics and feature definitions refer to a different model generation and must not be presented as Version 2.4 results. The current active model metrics and metadata are in `reports/model_evaluation_v2.json`.
