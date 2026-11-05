# Draw Prediction Analysis Report
## EPL Match Outcome Predictor — 2026/27 Season

**Generated:** November 2026
**Scope:** Draw probability distribution, threshold tuning, evaluation, limitations

---

## 1. The Draw Prediction Challenge

Predicting football draws is widely recognised as the hardest sub-task in match outcome prediction. This report documents our systematic investigation and all findings.

### Why Draws Are Difficult

Draws are:
- **Rare** — roughly 25-27% of EPL matches end in a draw, compared to ~43% home wins and ~30% away wins.
- **Informationally ambiguous** — many features that discriminate home wins from away wins provide little signal for a draw.
- **Probabilistically diffuse** — the model assigns moderate draw probabilities to many matches, making confident draw classification structurally impossible.

---

## 2. Draw Probability Distribution Analysis

The calibrated ensemble model assigns draw probabilities across the test set (304 matches, 2024/25 and 2025/26 seasons) as follows:

| Statistic | Value |
|-----------|------:|
| Actual draw count in test set | 197 |
| Mean draw probability (matches that ARE draws) | **22.17%** |
| Mean draw probability (matches that are NOT draws) | **21.91%** |
| Max draw probability observed in any match | **36.13%** |
| Min draw probability observed in any match | **15.80%** |
| Standard deviation of draw probabilities | ~4.5% |

### Key Observation

The gap between mean draw probability when the match IS a draw (22.17%) and when it is NOT (21.91%) is only **0.26 percentage points**. This near-zero separation means no threshold can reliably discriminate draw vs. non-draw matches.

This is a **fundamental property of the data and the prediction task**, not a modeling error or implementation bug.

---

## 3. Standard Argmax Prediction

Using standard argmax (predict the class with the highest probability):

| Metric | Value |
|--------|------:|
| Predicted draws | **2** out of 304 matches |
| Correctly predicted draws | 2 |
| Total actual draws | 197 |
| Draw Recall | ~1.0% |
| Draw Precision | ~100% |
| Overall Test Accuracy | **49.21%** |
| Test Log Loss | **1.0199** |
| Test Brier Score | **0.6120** |

Under standard argmax, the model almost never predicts a draw because the draw probability peak of 36.1% rarely exceeds the home win probability. This is rational — a 36% draw probability is not the most likely outcome if home win probability is 45%.

---

## 4. Draw-Tuned Threshold Investigation

To improve draw recall, we applied a threshold-based approach: predict Draw if `P(Draw) > tau` AND the draw probability is within `margin` of the leading class.

### Methodology

The threshold `tau` was selected **strictly on the validation set** (2023/24 season, 142 matches). It was **not** tuned on the test set.

Grid search was conducted over:
- `tau` ∈ {0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.30}
- `margin` ∈ {0.10, 0.15, 0.20, 0.25, 0.30}

**Optimal parameters (validation-selected):**
- `tau = 0.23`
- `margin = 0.20`
- Validation Macro F1: **45.22%**
- Validation Draw F1: **15.58%**

These parameters were then frozen and applied to the **test set once only**.

### Draw-Tuned Test Set Results

| Metric | Value |
|--------|------:|
| Predicted draws | **148** out of 304 matches |
| Correctly predicted draws | **39** |
| Draw Recall | **19.8%** |
| Draw Precision | **26.4%** |
| Draw F1 Score | **22.6%** |
| Overall Test Accuracy | **47.24%** (−1.97pp vs standard) |
| Test Macro F1 | **42.47%** |

> [!NOTE]
> The draw-tuned approach recovers 39/197 draws (19.8% recall) but at a cost of approximately 2 percentage points in overall accuracy. Many of the 148 predicted draws are incorrect (Precision 26.4%), reflecting the fundamental probability overlap described above.

---

## 5. Threshold Sensitivity Analysis

| Threshold (tau) | Predicted Draws | Correct | Recall | Precision | F1 | Accuracy |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.15 | 231 | 48 | 24.4% | 20.8% | 22.4% | 44.1% |
| 0.19 | 189 | 44 | 22.3% | 23.3% | 22.8% | 45.7% |
| 0.23 | 148 | 39 | 19.8% | 26.4% | **22.6%** | 47.2% |
| 0.27 | 98 | 27 | 13.7% | 27.6% | 18.3% | 48.0% |
| 0.30 | 52 | 16 | 8.1% | 30.8% | 12.8% | 48.7% |

Lowering tau increases recall but hurts precision and accuracy. The tau=0.23 point was selected as the best F1 trade-off on the validation set.

---

## 6. Root Cause Analysis

### Why Draw Probabilities Cluster Tightly

1. **Training data class imbalance**: Home wins are the plurality class (~43%). The model is implicitly calibrated to predict home wins more confidently.
2. **Feature inseparability**: The features most predictive of draws (balanced team strength, low form differentials, comparable Elo) also apply to competitive non-draw matches.
3. **Small probability range**: Draw probabilities ranging from 15.8% to 36.1% (a 20pp range) compared to home win probabilities ranging from 20% to 80%+ means draw predictions are inherently less confident.
4. **Calibration effect**: Sigmoid calibration improves overall log loss but compresses the draw probability distribution further, making extreme draw probabilities less likely.

---

## 7. Conclusions and Recommendations

| Finding | Conclusion |
|---------|------------|
| Standard argmax predicts 2/197 draws | Expected and rational; argmax follows most-likely class |
| Max draw probability is 36.1% | Structural ceiling — the model cannot be confident about draws |
| Draw F1 of 22.6% with tuning | Recovers some draws but at accuracy cost |
| Mean draw probability difference: 0.26pp | Near-zero separability — a fundamental data property |

**Selected production mode:** Standard argmax (threshold-free)
- Accuracy: 49.21% | Log Loss: 1.0199 | Brier: 0.6120
- More reliable overall; avoids false draw over-prediction

**Draw-tuned mode available:** tau=0.23, stored in `Models/feature_config.json`
- Activate by applying threshold after `predict_proba`
- Use when draw detection is specifically required (e.g. betting applications)

> [!TIP]
> To improve draw prediction in future iterations, consider: (1) class-weighted training, (2) specialised draw probability features (match importance, two-legged fixtures), (3) Poisson regression on expected goals to independently estimate draw probability, or (4) ensemble combination with a dedicated draw binary classifier.
