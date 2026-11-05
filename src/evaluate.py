"""
Evaluation, Calibration Assessment, and Explainability Module
EPL Match Outcome Predictor 2026/27
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    brier_score_loss,
    precision_score,
    recall_score
)

from src.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    TEST_SEASONS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_final_model():
    """
    Run detailed evaluation of the final production model:
    - Confusion Matrix (Actual vs Predicted H/D/A)
    - Classification Report (Precision, Recall, F1 for H, D, A)
    - Draw prediction analysis (Probability distribution for actual draws)
    - Feature Importances (Permutation Importance on test set)
    - Probability Calibration Assessment
    """
    # 1. Load model and artifacts
    model_path = MODELS_DIR / "final_model.joblib"
    preprocessor_path = MODELS_DIR / "preprocessing_pipeline.joblib"
    config_path = MODELS_DIR / "feature_config.json"

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    features = config["features"]
    classes = list(model.classes_)
    opt_draw_tau = config.get("optimal_draw_threshold", 0.23)

    # 2. Load test set
    features_df = pd.read_csv(PROCESSED_DATA_DIR / "match_features.csv")
    test_df = features_df[features_df["Season"].isin(TEST_SEASONS)].copy()

    X_test = test_df[features].values
    y_test = test_df["FTR"].values

    X_test_proc = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_proc)
    y_prob = model.predict_proba(X_test_proc)

    # Draw-tuned predictions
    draw_idx = classes.index("D")
    h_idx = classes.index("H")
    a_idx = classes.index("A")

    y_pred_draw_tuned = []
    for p in y_prob:
        p_d = p[draw_idx]
        p_h = p[h_idx]
        p_a = p[a_idx]
        if p_d >= opt_draw_tau and abs(p_h - p_a) <= 0.20:
            y_pred_draw_tuned.append("D")
        elif p_h >= p_a:
            y_pred_draw_tuned.append("H")
        else:
            y_pred_draw_tuned.append("A")
    y_pred_draw_tuned = np.array(y_pred_draw_tuned)

    # 3. Overall Performance
    acc = accuracy_score(y_test, y_pred)
    acc_tuned = accuracy_score(y_test, y_pred_draw_tuned)
    ll = log_loss(y_test, y_prob, labels=classes)

    # Multiclass Brier score
    brier = 0.0
    for idx, c in enumerate(classes):
        y_binary = (y_test == c).astype(float)
        brier += np.mean((y_prob[:, idx] - y_binary) ** 2)

    logger.info(f"Standard Test Accuracy: {acc*100:.2f}% | Draw-Tuned Accuracy: {acc_tuned*100:.2f}% | Log Loss: {ll:.4f} | Brier: {brier:.4f}")

    # 4. Classification Reports
    clf_report_dict = classification_report(y_test, y_pred, labels=classes, output_dict=True)
    clf_report_text = classification_report(y_test, y_pred, labels=classes)

    clf_report_tuned_dict = classification_report(y_test, y_pred_draw_tuned, labels=classes, output_dict=True)
    clf_report_tuned_text = classification_report(y_test, y_pred_draw_tuned, labels=classes)

    # 5. Confusion Matrices
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=[f"Actual_{c}" for c in classes], columns=[f"Pred_{c}" for c in classes])

    cm_tuned = confusion_matrix(y_test, y_pred_draw_tuned, labels=classes)
    cm_tuned_df = pd.DataFrame(cm_tuned, index=[f"Actual_{c}" for c in classes], columns=[f"Pred_{c}" for c in classes])

    # 6. Draw Prediction Analysis
    actual_draw_mask = (y_test == "D")
    draw_probs_actual_draws = y_prob[actual_draw_mask, draw_idx]
    draw_probs_non_draws = y_prob[~actual_draw_mask, draw_idx]

    draw_analysis = {
        "actual_draw_count": int(actual_draw_mask.sum()),
        "predicted_draws_standard": int((y_pred == "D").sum()),
        "predicted_draws_tuned": int((y_pred_draw_tuned == "D").sum()),
        "correctly_predicted_draws_tuned": int(((y_test == "D") & (y_pred_draw_tuned == "D")).sum()),
        "draw_recall_tuned": round(float(recall_score(y_test == "D", y_pred_draw_tuned == "D")), 4),
        "draw_precision_tuned": round(float(precision_score(y_test == "D", y_pred_draw_tuned == "D", zero_division=0)), 4),
        "draw_f1_tuned": round(float(f1_score(y_test == "D", y_pred_draw_tuned == "D")), 4),
        "mean_draw_prob_when_draw": round(float(np.mean(draw_probs_actual_draws)), 4),
        "mean_draw_prob_when_not_draw": round(float(np.mean(draw_probs_non_draws)), 4),
        "max_draw_prob_observed": round(float(np.max(y_prob[:, draw_idx])), 4),
        "min_draw_prob_observed": round(float(np.min(y_prob[:, draw_idx])), 4)
    }

    # 7. Permutation Feature Importance
    logger.info("Computing permutation feature importance on test set...")
    perm_importance = permutation_importance(
        model, X_test_proc, y_test,
        n_repeats=5,
        random_state=RANDOM_SEED,
        scoring="accuracy"
    )

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance_Mean": perm_importance.importances_mean,
        "Importance_Std": perm_importance.importances_std
    }).sort_values("Importance_Mean", ascending=False).reset_index(drop=True)

    # Save importance
    importance_path = REPORTS_DIR / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    # 8. Calibration Assessment
    prob_stats = {}
    for idx, c in enumerate(classes):
        y_binary = (y_test == c).astype(int)
        p = y_prob[:, idx]
        # Binned calibration
        bins = np.linspace(0, 1, 6)
        bin_ids = np.digitize(p, bins) - 1
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_actuals = []
        bin_preds = []
        bin_counts = []
        for b in range(len(bins) - 1):
            mask = (bin_ids == b)
            if mask.sum() > 0:
                bin_actuals.append(round(float(y_binary[mask].mean()), 4))
                bin_preds.append(round(float(p[mask].mean()), 4))
                bin_counts.append(int(mask.sum()))
            else:
                bin_actuals.append(0.0)
                bin_preds.append(round(float(bin_centers[b]), 4))
                bin_counts.append(0)
        prob_stats[c] = {
            "mean_predicted": round(float(np.mean(p)), 4),
            "actual_frequency": round(float(np.mean(y_binary)), 4),
            "binned_preds": bin_preds,
            "binned_actuals": bin_actuals,
            "binned_counts": bin_counts
        }

    eval_summary = {
        "test_seasons": TEST_SEASONS,
        "test_samples": len(test_df),
        "standard_accuracy": round(acc * 100, 2),
        "draw_tuned_accuracy": round(acc_tuned * 100, 2),
        "log_loss": round(ll, 4),
        "brier_score": round(brier, 4),
        "classification_report_standard": clf_report_dict,
        "classification_report_draw_tuned": clf_report_tuned_dict,
        "confusion_matrix_standard": cm_df.to_dict(),
        "confusion_matrix_draw_tuned": cm_tuned_df.to_dict(),
        "draw_analysis": draw_analysis,
        "calibration": prob_stats,
        "top_10_features": importance_df.head(10).to_dict(orient="records")
    }

    eval_json_path = REPORTS_DIR / "evaluation_metrics.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    logger.info(f"Saved evaluation metrics to {eval_json_path}")
    return eval_summary, cm_df, cm_tuned_df, importance_df, clf_report_text, clf_report_tuned_text


if __name__ == "__main__":
    summary, cm, cm_tuned, feat_imp, rep_std, rep_tuned = evaluate_final_model()
    print("\n--- STANDARD PREDICTION CLASSIFICATION REPORT ---")
    print(rep_std)
    print("\n--- STANDARD CONFUSION MATRIX ---")
    print(cm)
    print("\n--- DRAW-TUNED CLASSIFICATION REPORT ---")
    print(rep_tuned)
    print("\n--- DRAW-TUNED CONFUSION MATRIX ---")
    print(cm_tuned)
    print("\n--- TOP 10 PERMUTATION FEATURES ---")
    print(feat_imp.head(10))
    print("\n--- DRAW PREDICTION ANALYSIS ---")
    print(json.dumps(summary["draw_analysis"], indent=2))
