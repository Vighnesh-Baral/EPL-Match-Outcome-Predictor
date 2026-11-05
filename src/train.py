"""Train and compare the Version 2 recency-weighted historical models."""
from __future__ import annotations
import json
from src.historical_model import train_and_save_model


def main() -> int:
    result = train_and_save_model()
    metadata = result["metadata"]
    print("Version 2 historical model training completed.")
    print(f"Selected classifier: {metadata['model_name']}")
    print(f"Training matches: {metadata['training_matches']}")
    print(f"Training period: {metadata['training_start']} to {metadata['training_end']}")
    print(f"Ensemble weights: classifier {metadata['ml_probability_weight']:.0%}, Poisson {metadata['poisson_probability_weight']:.0%}")
    print("\nValidation comparison:")
    for row in result["validation_comparison"]:
        print(f"- {row['model']}: log loss {row['validation_blend_log_loss']:.4f}")
    print("\nUntouched test-period comparison:")
    for row in result["test_comparison"]:
        print(f"- {row['model']}: log loss {row['log_loss']:.4f}, Brier {row['brier_score']:.4f}, accuracy {row['accuracy']:.3f}")
    print("\nSaved model: Models/historical_predictor_v2.joblib")
    print("Saved reports: reports/model_evaluation_v2.json and reports/model_comparison_v2.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
