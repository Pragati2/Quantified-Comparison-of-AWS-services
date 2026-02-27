# =============================================================================
# CSP-554: Credit Card Fraud Detection — AWS SageMaker (Python / Jupyter)
# Models  : Logistic Regression · SGD Classifier · Random Forest
# Metrics : Recall · F1-Score
# =============================================================================

# ── 0. Install dependencies ───────────────────────────────────────────────────
# !pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import StandardScaler
from sklearn.linear_model       import LogisticRegression, SGDClassifier
from sklearn.ensemble           import RandomForestClassifier
from sklearn.metrics            import (classification_report,
                                        confusion_matrix,
                                        f1_score, recall_score,
                                        ConfusionMatrixDisplay)
from imblearn.over_sampling     import SMOTE
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load Data ──────────────────────────────────────────────────────────────
# On SageMaker, data is typically stored in S3; update the path as needed
DATA_PATH = "data/train.csv"    # or "s3://your-bucket/train.csv"

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("\nClass distribution:\n", df["Class"].value_counts())
print(f"\nFraud rate: {df['Class'].mean()*100:.3f}%")

# ── 2. Preprocessing ──────────────────────────────────────────────────────────
# Drop duplicates
df.drop_duplicates(inplace=True)

# Features & target
X = df.drop(columns=["Class"])
y = df["Class"]

# Scale Amount and Time (V1-V28 are already PCA-transformed)
scaler = StandardScaler()
X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])

# ── 3. Train / Validation / Test Split ───────────────────────────────────────
X_temp, X_test,  y_temp, y_test  = train_test_split(X, y, test_size=0.15,
                                                      stratify=y, random_state=42)
X_train, X_val,  y_train, y_val  = train_test_split(X_temp, y_temp, test_size=0.176,
                                                      stratify=y_temp, random_state=42)
# 0.176 of 0.85 ≈ 0.15 of total → 70 / 15 / 15 split

print(f"\nTrain: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# ── 4. Handle Class Imbalance with SMOTE (train set only) ────────────────────
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"\nAfter SMOTE — Class distribution:\n{pd.Series(y_train_res).value_counts()}")

# ── 5. Helper: Evaluate a model ───────────────────────────────────────────────
def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    f1  = f1_score(y_te, preds, average="binary")
    rec = recall_score(y_te, preds, average="binary")
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(classification_report(y_te, preds,
                                 target_names=["Legit", "Fraud"]))
    print(f"  F1-Score : {f1:.4f}")
    print(f"  Recall   : {rec:.4f}")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_te, preds, display_labels=["Legit", "Fraud"],
        cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(f"output/cm_{name.replace(' ', '_')}.png", dpi=150)
    plt.show()
    return {"model": name, "f1": round(f1, 4), "recall": round(rec, 4)}

# ── 6. Model Training & Evaluation ───────────────────────────────────────────
results = []

# 6a. Logistic Regression
results.append(evaluate(
    "Logistic Regression",
    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    X_train_res, y_train_res, X_test, y_test
))

# 6b. Stochastic Gradient Descent
results.append(evaluate(
    "SGD Classifier",
    SGDClassifier(loss="log_loss", max_iter=1000,
                  class_weight="balanced", random_state=42),
    X_train_res, y_train_res, X_test, y_test
))

# 6c. Random Forest
results.append(evaluate(
    "Random Forest",
    RandomForestClassifier(n_estimators=500, max_depth=None,
                           class_weight="balanced", random_state=42, n_jobs=-1),
    X_train_res, y_train_res, X_test, y_test
))

# ── 7. Model Comparison Plot ──────────────────────────────────────────────────
results_df = pd.DataFrame(results)
print("\n=== Model Comparison ===")
print(results_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, metric in zip(axes, ["f1", "recall"]):
    bars = ax.barh(results_df["model"], results_df[metric],
                   color=["#58a6ff", "#3fb950", "#ffa657"])
    ax.set_xlim(0, 1.1)
    ax.set_xlabel(metric.upper() + " Score")
    ax.set_title(f"Model Comparison — {metric.upper()}")
    for bar, val in zip(bars, results_df[metric]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("output/sagemaker_model_comparison.png", dpi=150)
plt.show()

# ── 8. Feature Importance (Random Forest) ─────────────────────────────────────
rf_model = RandomForestClassifier(n_estimators=500, random_state=42,
                                   class_weight="balanced", n_jobs=-1)
rf_model.fit(X_train_res, y_train_res)

feat_imp = pd.Series(rf_model.feature_importances_,
                     index=X.columns).sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(8, 5))
feat_imp.plot(kind="barh", ax=ax, color="#f0883e")
ax.invert_yaxis()
ax.set_title("Top 10 Feature Importances — Random Forest")
ax.set_xlabel("Importance (Mean Decrease in Impurity)")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("output/feature_importance.png", dpi=150)
plt.show()

print("\nDone. All outputs saved to output/")
