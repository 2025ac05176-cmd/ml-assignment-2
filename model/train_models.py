"""
Student Dropout & Academic Success - Classification Model Training
==================================================================
BITS Pilani WILP | M.Tech (AIML/DSE) | Machine Learning | Assignment 2

Trains five classification models on the "Predict Students' Dropout and
Academic Success" dataset and evaluates each on a held-out test split using
six metrics: Accuracy, AUC, Precision, Recall, F1 and MCC.

Every model is wrapped in a scikit-learn Pipeline so that the exact same
preprocessing that was applied at training time is reapplied automatically
at inference time inside the Streamlit app. This removes the classic
"forgot to scale the test data" deployment bug.

Run:  python model/train_models.py
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_FRACTION = 0.20

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_FILE = os.path.join(ROOT, "data", "dataset.csv")
TEST_CSV_OUT = os.path.join(ROOT, "test_data.csv")
METRICS_OUT = os.path.join(HERE, "metrics_summary.json")

TARGET_COLUMN = "Target"


# --------------------------------------------------------------------------
# 1. Load and describe
# --------------------------------------------------------------------------
def load_dataset():
    frame = pd.read_csv(DATA_FILE)
    frame.columns = [c.strip() for c in frame.columns]

    print("=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)
    print(f"Rows              : {frame.shape[0]}")
    print(f"Columns           : {frame.shape[1]}  ({frame.shape[1] - 1} features + 1 target)")
    print(f"Missing values    : {int(frame.isnull().sum().sum())}")
    print(f"Duplicate rows    : {int(frame.duplicated().sum())}")
    print("\nTarget distribution:")
    counts = frame[TARGET_COLUMN].value_counts()
    for label, n in counts.items():
        print(f"  {label:<10} {n:>5}  ({n / len(frame) * 100:5.2f}%)")
    print(f"\nImbalance ratio (majority/minority): {counts.max() / counts.min():.2f}")
    return frame


# --------------------------------------------------------------------------
# 2. Split
# --------------------------------------------------------------------------
def split_dataset(frame):
    """Stratified split so the rare 'Enrolled' class keeps its proportion."""
    features = frame.drop(columns=[TARGET_COLUMN])
    labels_raw = frame[TARGET_COLUMN]

    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels_raw)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_FRACTION,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print("\n" + "=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)
    print(f"Training rows : {len(x_train)}")
    print(f"Test rows     : {len(x_test)}")
    print(f"Class order   : {list(encoder.classes_)}  ->  {list(range(len(encoder.classes_)))}")

    # test_data.csv is what the marker / user uploads into the Streamlit app.
    # It keeps the original string labels so the CSV stays human readable.
    holdout = x_test.copy()
    holdout[TARGET_COLUMN] = encoder.inverse_transform(y_test)
    holdout.to_csv(TEST_CSV_OUT, index=False)
    print(f"Wrote holdout -> {TEST_CSV_OUT}  ({holdout.shape[0]} rows)")

    return x_train, x_test, y_train, y_test, encoder


# --------------------------------------------------------------------------
# 3. Model definitions
# --------------------------------------------------------------------------
def build_models():
    """
    Five classifiers as required by the assignment.

    Distance- and probability-based learners (Logistic Regression, kNN,
    Gaussian Naive Bayes) are preceded by StandardScaler because their
    objective functions are sensitive to feature magnitude - 'Age at
    enrollment' spans ~17-70 while 'GDP' spans ~-4 to 4.

    Tree-based learners (Decision Tree, Random Forest) split on thresholds
    and are invariant to monotone rescaling, so they are left unscaled.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=3000,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            )),
        ]),

        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(
                criterion="entropy",
                max_depth=8,           # pruned: unbounded depth memorises the training set
                min_samples_leaf=12,
                min_samples_split=20,
                random_state=RANDOM_STATE,
            )),
        ]),

        "kNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=15,        # ~sqrt(n_train)/4; smoother boundary on noisy classes
                weights="distance",
                metric="minkowski",
                p=2,
            )),
        ]),

        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB(var_smoothing=1e-8)),
        ]),

        "Random Forest (Ensemble)": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }


# --------------------------------------------------------------------------
# 4. Metrics
# --------------------------------------------------------------------------
def evaluate(model, x_test, y_test):
    """
    Six metrics required by the assignment.

    This is a 3-class problem, so:
      * AUC uses one-vs-rest with weighted averaging over classes.
      * Precision / Recall / F1 use weighted averaging, which accounts for
        the class imbalance instead of over-rewarding the majority class.
      * MCC is naturally multi-class and needs no averaging scheme. It is
        the most trustworthy single number here because it only scores well
        when a model does well on ALL three classes.
    """
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def per_class_recall(model, x_test, y_test, encoder):
    """Recall per class - used to justify the written observations."""
    y_pred = model.predict(x_test)
    scores = recall_score(y_test, y_pred, average=None, zero_division=0)
    return {label: float(s) for label, s in zip(encoder.classes_, scores)}


# --------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------
def main():
    frame = load_dataset()
    x_train, x_test, y_train, y_test, encoder = split_dataset(frame)

    models = build_models()
    results, extras = {}, {}

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    for name, model in models.items():
        model.fit(x_train, y_train)

        scores = evaluate(model, x_test, y_test)
        results[name] = scores

        cv = cross_val_score(model, x_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
        train_acc = accuracy_score(y_train, model.predict(x_train))

        extras[name] = {
            "cv_mean": float(cv.mean()),
            "cv_std": float(cv.std()),
            "train_accuracy": float(train_acc),
            "overfit_gap": float(train_acc - scores["Accuracy"]),
            "per_class_recall": per_class_recall(model, x_test, y_test, encoder),
        }

        # compress=3 keeps the Random Forest under ~12 MB instead of ~37 MB,
        # which matters for repository size and Streamlit Cloud memory.
        filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(model, os.path.join(HERE, f"{filename}.pkl"), compress=3)

        print(f"\n{name}")
        print("-" * len(name))
        for metric, value in scores.items():
            print(f"  {metric:<10} {value:.4f}")
        print(f"  {'CV(5) acc':<10} {cv.mean():.4f} +/- {cv.std():.4f}")
        print(f"  {'Train acc':<10} {train_acc:.4f}   (gap {train_acc - scores['Accuracy']:+.4f})")
        print(f"  saved -> model/{filename}.pkl")

    joblib.dump(encoder, os.path.join(HERE, "label_encoder.pkl"))
    joblib.dump(list(x_train.columns), os.path.join(HERE, "feature_names.pkl"))

    # ---------------- comparison table ----------------
    table = pd.DataFrame(results).T[
        ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ].round(4)

    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(table.to_string())

    winner = table["MCC"].idxmax()
    print(f"\nBest by MCC : {winner}  ({table.loc[winner, 'MCC']:.4f})")
    print(f"Best by F1  : {table['F1'].idxmax()}")
    print(f"Best by AUC : {table['AUC'].idxmax()}")

    print("\n" + "=" * 70)
    print(f"CONFUSION MATRIX + REPORT - {winner}")
    print("=" * 70)
    best = models[winner]
    print(confusion_matrix(y_test, best.predict(x_test)))
    print()
    print(classification_report(
        y_test, best.predict(x_test),
        target_names=encoder.classes_, digits=4, zero_division=0,
    ))

    table.to_csv(os.path.join(HERE, "comparison_table.csv"))
    with open(METRICS_OUT, "w") as fh:
        json.dump(
            {"metrics": results, "diagnostics": extras,
             "winner": winner, "classes": list(encoder.classes_)},
            fh, indent=2,
        )
    print(f"Saved metrics -> {METRICS_OUT}")
    print("Done.")


if __name__ == "__main__":
    main()
