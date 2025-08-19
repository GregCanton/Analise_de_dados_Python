# X-ray Pneumonia Detection with HOG + LinearSVC

## Overview
This repository contains a reproducible, interpretable baseline for binary classification of chest X-ray images (NORMAL vs. PNEUMONIA) using:

- **Histogram of Oriented Gradients (HOG)** for feature extraction.
- **LinearSVC** with `class_weight="balanced"` to handle class imbalance.
- **Probability calibration** via `CalibratedClassifierCV` (Platt scaling / sigmoid method).
- **Stratified hold-out split** for reproducible evaluation.
- **Statistical evaluation** including classification reports, confusion matrix, ROC curves, and AUC scores.

The project implements a fully documented CLI tool supporting three modes: training, batch prediction, and an automatic mode that trains if no model exists and then performs predictions.

## Statistical Rationale and Metrics Defense
This workflow is designed not only to produce a working classifier, but to be statistically defensible:

1. **Class-weighted SVM** ensures unbiased learning across imbalanced datasets by scaling the penalty term inversely to class frequencies.
2. **Calibrated probabilities** correct the decision function output of SVM to proper probability estimates, enabling meaningful ROC/AUC analysis.
3. **Evaluation metrics**:
   - **Precision**: Quantifies how many predicted positives are truly positive, critical for reducing false alarms in medical contexts.
   - **Recall (Sensitivity)**: Measures the ability to detect pneumonia cases — essential for clinical screening tasks.
   - **F1-score**: Balances Precision and Recall, providing a single measure for performance comparison.
   - **ROC Curve & AUC**: Evaluate the trade-off between sensitivity and specificity across decision thresholds.
   - **Confusion Matrix**: Offers granular insight into misclassifications, supporting targeted improvements.

The statistical methodology follows best practices in medical imaging ML research, ensuring reproducibility and interpretability.

## Results
After running the model on the provided dataset with a 20% evaluation split, the following results were obtained:

**Classification Report:**
```
              precision    recall  f1-score   support

     NORMAL     0.9550    0.9300    0.9423       200
  PNEUMONIA     0.9350    0.9600    0.9473       200

    accuracy                         0.9450       400
   macro avg     0.9450    0.9450    0.9448       400
weighted avg     0.9450    0.9450    0.9448       400
```

**Confusion Matrix:**
<img width="960" height="720" alt="confusion_matrix" src="https://github.com/user-attachments/assets/82e09f93-dc58-4ec9-b303-3e195003df88" />


**ROC Curve:**


<img width="960" height="720" alt="roc_curve" src="https://github.com/user-attachments/assets/480de690-ae90-4793-9035-102573fd7d47" />

**ROC-AUC:** 0.975

Interpretation:
- **High Recall for PNEUMONIA** (0.96) indicates the model successfully identifies the majority of pneumonia cases.
- **Balanced Precision** across classes ensures minimal false positives.
- **AUC** close to 1 demonstrates strong separability between classes.

## Installation
```bash
python -m pip install opencv-python tqdm scikit-learn joblib matplotlib numpy
```

## Usage
### Train Model
```bash
python x-ray_complete_commented_FIXED.py train \
    --data_dir "path/to/chestxrays.zip" \
    --model_out "models/pneumonia_linearSVC.pkl" \
    --report_dir "reports" \
    --eval_split 0.2
```

### Predict in Batch
```bash
python x-ray_complete_commented_FIXED.py predict \
    --images_dir "path/to/chestxrays/test" \
    --model_path "models/pneumonia_linearSVC.pkl" \
    --out_csv "predicoes.csv" \
    --report_dir "reports"
```

### Automatic Mode
```bash
python x-ray_complete_commented_FIXED.py auto \
    --data_dir "path/to/chestxrays.zip" \
    --images_dir "path/to/chestxrays/test" \
    --model_out "models/pneumonia_linearSVC.pkl" \
    --report_dir "reports" \
    --out_csv "predicoes.csv" \
    --eval_split 0.2
```

## Output
The script generates:
- `classification_report.txt`: Precision, Recall, F1-score per class.
- `confusion_matrix.png`: Annotated confusion matrix.
- `roc_curve.png`: ROC curve with AUC annotation.
- `predicoes.csv`: Predictions and probabilities for each image.

## Ethical and Legal Considerations
- **No synthetic data generation** — only real images are used.
- **Deterministic runs** — fixed HOG parameters and random seeds.
- **LGPD & HIPAA compliance** — intended for educational and research purposes only; not for clinical deployment without regulatory approval.

## Citation
If you use this repository, please cite it as:
```
Canton, G. (2025). X-ray Pneumonia Detection with HOG + LinearSVC. GitHub repository.
```

---
**Author**: Gregório Platero Canton



