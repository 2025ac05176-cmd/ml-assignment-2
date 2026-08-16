"""
Student Dropout Prediction - Interactive Model Evaluation Dashboard
===================================================================

Streamlit application that lets a user upload a test CSV, pick one of five
trained classifiers, and inspect its performance through metrics, a
confusion matrix, a classification report and per-row predictions.
"""

import io
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Dropout Classifier",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#4C6EF5"
PALETTE = ["#4C6EF5", "#F59F00", "#12B886"]

st.markdown(
    f"""
    <style>
      .main-heading {{
          font-size: 2.05rem; font-weight: 700; color: {ACCENT};
          margin-bottom: 0.1rem; letter-spacing: -0.02em;
      }}
      .sub-heading {{
          font-size: 0.95rem; color: #868E96; margin-bottom: 1.6rem;
      }}
      .metric-card {{
          background: linear-gradient(140deg, #F8F9FE 0%, #EEF1FC 100%);
          border: 1px solid #DEE2F0; border-left: 4px solid {ACCENT};
          border-radius: 9px; padding: 0.85rem 1rem; text-align: center;
      }}
      .metric-card .label {{
          font-size: 0.72rem; color: #6B7280; text-transform: uppercase;
          letter-spacing: 0.09em; font-weight: 600;
      }}
      .metric-card .value {{
          font-size: 1.7rem; font-weight: 700; color: #1F2937; line-height: 1.25;
      }}
      .note {{
          background: #FFF9E6; border-left: 4px solid #F59F00;
          padding: 0.7rem 0.95rem; border-radius: 6px; font-size: 0.88rem;
      }}
      div[data-testid="stDataFrame"] {{ border-radius: 8px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
TARGET_COLUMN = "Target"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}

MODEL_NOTES = {
    "Logistic Regression": "Linear baseline on standardised features. Strongest of the "
                           "non-ensemble models here, which says the classes are close to "
                           "linearly separable.",
    "Decision Tree": "Single tree pruned to depth 8. Fully interpretable but higher variance "
                     "than the forest built from the same splits.",
    "kNN": "15 neighbours, distance-weighted, on standardised features. Struggles because "
           "34 dimensions dilute the notion of 'nearby'.",
    "Naive Bayes": "Gaussian NB. Its feature-independence assumption is badly violated here - "
                   "1st and 2nd semester results are strongly correlated.",
    "Random Forest (Ensemble)": "400 bagged trees with balanced subsampling. Best model on "
                                "every one of the six metrics.",
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    models, missing = {}, []
    for name, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            missing.append(filename)

    encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    features_path = os.path.join(MODEL_DIR, "feature_names.pkl")
    encoder = joblib.load(encoder_path) if os.path.exists(encoder_path) else None
    features = joblib.load(features_path) if os.path.exists(features_path) else None
    return models, encoder, features, missing


@st.cache_data(show_spinner=False)
def load_bundled_test_data():
    path = os.path.join(HERE, "test_data.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


MODELS, ENCODER, FEATURE_NAMES, MISSING = load_artifacts()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_proba):
    """Six assignment metrics. Weighted averaging handles the 3-class imbalance."""
    scores = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": np.nan,
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    try:
        if y_proba is not None and len(np.unique(y_true)) > 2:
            scores["AUC"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        elif y_proba is not None:
            scores["AUC"] = roc_auc_score(y_true, y_proba[:, 1])
    except ValueError:
        pass
    return scores


def metric_card(column, label, value):
    text = "n/a" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.4f}"
    column.markdown(
        f'<div class="metric-card"><div class="label">{label}</div>'
        f'<div class="value">{text}</div></div>',
        unsafe_allow_html=True,
    )


def draw_confusion_matrix(y_true, y_pred, class_names):
    matrix = confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))

    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=1.2, linecolor="white", ax=axes[0],
                annot_kws={"size": 12, "weight": "bold"})
    axes[0].set_title("Confusion Matrix (counts)", fontsize=11, weight="bold", pad=10)
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

    with np.errstate(divide="ignore", invalid="ignore"):
        normalised = matrix / matrix.sum(axis=1, keepdims=True)
    normalised = np.nan_to_num(normalised)

    sns.heatmap(normalised, annot=True, fmt=".2f", cmap="Oranges", cbar=False,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=1.2, linecolor="white", ax=axes[1],
                annot_kws={"size": 12, "weight": "bold"})
    axes[1].set_title("Row-normalised (recall per class)", fontsize=11, weight="bold", pad=10)
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")

    fig.tight_layout()
    return fig


def draw_roc_curves(y_true, y_proba, class_names):
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for index, label in enumerate(class_names):
        binary_truth = (np.asarray(y_true) == index).astype(int)
        if binary_truth.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(binary_truth, y_proba[:, index])
        area = roc_auc_score(binary_truth, y_proba[:, index])
        ax.plot(fpr, tpr, lw=2, color=PALETTE[index % len(PALETTE)],
                label=f"{label}  (AUC {area:.3f})")
    ax.plot([0, 1], [0, 1], "--", lw=1, color="#ADB5BD", label="Random")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-Rest ROC", fontsize=11, weight="bold")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Controls")

    if MISSING:
        st.error("Missing model files:\n\n" + "\n".join(f"- {m}" for m in MISSING))
        st.stop()
    if not MODELS:
        st.error("No trained models found in `model/`. Run `python model/train_models.py` first.")
        st.stop()

    # --- (b) model selection dropdown ---
    selected_model = st.selectbox("Classification model", list(MODELS.keys()), index=4)
    st.caption(MODEL_NOTES.get(selected_model, ""))

    st.divider()

    # --- (a) dataset upload option (CSV) ---
    # Upload is the primary path, per the assignment requirement. The bundled
    # fallback exists only so the app is still demonstrable without a download.
    st.markdown("### 📁 Test data")
    source_mode = st.radio(
        "Data source",
        ["Upload a CSV", "Use bundled test_data.csv"],
        index=0,
        help="Upload the 885-row test_data.csv from the repository, or any CSV "
             "carrying the same 34 feature columns.",
    )

    uploaded = None
    if source_mode == "Upload a CSV":
        uploaded = st.file_uploader(
            "Choose test CSV",
            type=["csv"],
            help="Must contain the same 34 feature columns used in training. "
                 "Include a 'Target' column to see evaluation metrics.",
        )

    use_bundled = source_mode == "Use bundled test_data.csv"

    st.divider()
    compare_all = st.checkbox("Compare all 5 models", value=False)
    show_roc = st.checkbox("Show ROC curves", value=True)

    st.divider()
    st.caption(
        "Assignment 2 · Machine Learning\n\n"
        "BITS Pilani WILP · M.Tech (AIML/DSE)\n\n"
        "Dataset: Predict Students' Dropout and Academic Success"
    )


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown('<div class="main-heading">🎓 Student Dropout &amp; Academic Success</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-heading">Upload test data, choose a classifier, and compare six '
    'evaluation metrics across five machine learning models.</div>',
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Resolve input data
# --------------------------------------------------------------------------
data, source_label = None, ""

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
        source_label = f"uploaded file · **{uploaded.name}**"
    except Exception as exc:
        st.error(f"Could not read that CSV: {exc}")
        st.stop()
elif use_bundled:
    data = load_bundled_test_data()
    source_label = "bundled **test_data.csv**"

if data is None:
    st.info(
        "**👈 Upload a test CSV in the sidebar to begin.**\n\n"
        "Use `test_data.csv` from this project's repository — it is the 885-row "
        "held-out split that none of the models saw during training. Any CSV with "
        "the same 34 feature columns will work; include a `Target` column to see "
        "evaluation metrics as well as predictions."
    )
    st.caption(
        "No file to hand? Switch the sidebar to **Use bundled test_data.csv** to "
        "load the same file straight from the repository."
    )
    st.stop()

data.columns = [c.strip() for c in data.columns]

if FEATURE_NAMES:
    absent = [c for c in FEATURE_NAMES if c not in data.columns]
    if absent:
        st.error(
            f"The uploaded file is missing {len(absent)} required feature column(s). "
            f"First few: {', '.join(absent[:6])}"
        )
        st.stop()

has_labels = TARGET_COLUMN in data.columns
features = data[FEATURE_NAMES] if FEATURE_NAMES else data.drop(columns=[TARGET_COLUMN], errors="ignore")

st.success(f"Loaded {len(data):,} rows × {features.shape[1]} features from {source_label}.")
if not has_labels:
    st.markdown(
        '<div class="note">No <code>Target</code> column found, so this file is treated as '
        'unlabelled. Predictions are shown, but evaluation metrics need ground truth.</div>',
        unsafe_allow_html=True,
    )

y_true = None
if has_labels and ENCODER is not None:
    try:
        y_true = ENCODER.transform(data[TARGET_COLUMN].astype(str))
    except ValueError:
        st.warning(
            f"The Target column contains labels outside the trained set "
            f"{list(ENCODER.classes_)}. Metrics disabled."
        )
        has_labels = False

class_names = list(ENCODER.classes_) if ENCODER is not None else []


# --------------------------------------------------------------------------
# Predict
# --------------------------------------------------------------------------
model = MODELS[selected_model]
y_pred = model.predict(features)
y_proba = model.predict_proba(features) if hasattr(model, "predict_proba") else None

tab_metrics, tab_matrix, tab_predictions, tab_data = st.tabs(
    ["📊 Metrics", "🔢 Confusion Matrix", "🔮 Predictions", "🗂️ Data Preview"]
)


# --- (c) display of evaluation metrics ---
with tab_metrics:
    if has_labels and y_true is not None:
        scores = compute_metrics(y_true, y_pred, y_proba)

        st.markdown(f"#### {selected_model}")
        row_one = st.columns(3)
        for column, key in zip(row_one, ["Accuracy", "AUC", "Precision"]):
            metric_card(column, key, scores[key])
        row_two = st.columns(3)
        for column, key in zip(row_two, ["Recall", "F1", "MCC"]):
            metric_card(column, key, scores[key])

        st.caption(
            "AUC is one-vs-rest, weighted across the three classes. Precision, Recall and F1 "
            "use weighted averaging so the minority *Enrolled* class is not drowned out. "
            "MCC is inherently multi-class."
        )

        if show_roc and y_proba is not None and len(class_names) == y_proba.shape[1]:
            st.divider()
            left, right = st.columns([1, 1])
            with left:
                st.pyplot(draw_roc_curves(y_true, y_proba, class_names))
            with right:
                st.markdown("##### Per-class recall")
                per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
                st.dataframe(
                    pd.DataFrame({"Class": class_names, "Recall": per_class.round(4)}),
                    hide_index=True, width="stretch",
                )

        if compare_all:
            st.divider()
            st.markdown("#### All five models on this data")
            rows = {}
            progress = st.progress(0.0, text="Scoring models…")
            for i, (name, candidate) in enumerate(MODELS.items(), start=1):
                pred = candidate.predict(features)
                proba = candidate.predict_proba(features) if hasattr(candidate, "predict_proba") else None
                rows[name] = compute_metrics(y_true, pred, proba)
                progress.progress(i / len(MODELS), text=f"Scoring {name}…")
            progress.empty()

            table = pd.DataFrame(rows).T[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
            st.dataframe(
                table.style.format("{:.4f}").highlight_max(axis=0, color="#D3F9D8"),
                width="stretch",
            )
            st.caption(f"Best by MCC: **{table['MCC'].idxmax()}** · green cells mark the best score per metric.")

            chart, axis = plt.subplots(figsize=(9, 3.6))
            table[["Accuracy", "F1", "MCC"]].plot(
                kind="bar", ax=axis, color=PALETTE, width=0.75, edgecolor="white",
            )
            axis.set_ylabel("Score"); axis.set_ylim(0, 1)
            axis.set_title("Model comparison", fontsize=11, weight="bold")
            axis.legend(frameon=False, fontsize=9)
            axis.spines[["top", "right"]].set_visible(False)
            plt.xticks(rotation=18, ha="right", fontsize=8)
            chart.tight_layout()
            st.pyplot(chart)
    else:
        st.info("Evaluation metrics require a `Target` column in the uploaded CSV.")


# --- (d) confusion matrix / classification report ---
with tab_matrix:
    if has_labels and y_true is not None:
        st.pyplot(draw_confusion_matrix(y_true, y_pred, class_names))
        st.markdown("#### Classification report")
        report = classification_report(
            y_true, y_pred, target_names=class_names,
            output_dict=True, zero_division=0,
        )
        st.dataframe(
            pd.DataFrame(report).T.style.format("{:.4f}"),
            width="stretch",
        )
    else:
        st.info("A confusion matrix needs ground-truth labels. Upload a CSV with a `Target` column.")


with tab_predictions:
    output = data.copy()
    output["Predicted"] = ENCODER.inverse_transform(y_pred) if ENCODER is not None else y_pred
    if y_proba is not None:
        output["Confidence"] = y_proba.max(axis=1).round(4)
        for index, label in enumerate(class_names):
            output[f"P({label})"] = y_proba[:, index].round(4)
    if has_labels:
        output["Correct"] = np.where(output[TARGET_COLUMN] == output["Predicted"], "✓", "✗")

    left, right = st.columns([1, 1])
    with left:
        distribution = pd.Series(output["Predicted"]).value_counts()
        fig, axis = plt.subplots(figsize=(4.6, 3.3))
        axis.bar(distribution.index.astype(str), distribution.values,
                 color=PALETTE[: len(distribution)], edgecolor="white")
        axis.set_title("Predicted class distribution", fontsize=11, weight="bold")
        axis.set_ylabel("Rows")
        axis.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
    with right:
        if has_labels:
            correct = int((output["Correct"] == "✓").sum())
            st.metric("Correctly classified", f"{correct:,} / {len(output):,}",
                      f"{correct / len(output) * 100:.2f}%")
        if y_proba is not None:
            st.metric("Mean confidence", f"{y_proba.max(axis=1).mean():.4f}")
            low = int((y_proba.max(axis=1) < 0.5).sum())
            st.metric("Low-confidence rows (<0.50)", f"{low:,}")

    st.dataframe(output.head(200), width="stretch", height=330)
    st.caption(f"Showing first 200 of {len(output):,} rows.")

    buffer = io.StringIO()
    output.to_csv(buffer, index=False)
    st.download_button(
        "⬇️ Download predictions as CSV",
        buffer.getvalue(),
        file_name=f"predictions_{selected_model.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )


with tab_data:
    st.markdown("#### Uploaded data")
    st.dataframe(data.head(100), width="stretch", height=300)
    st.markdown("#### Numeric summary")
    st.dataframe(data.describe().T.style.format("{:.3f}"), width="stretch", height=300)
    if has_labels:
        st.markdown("#### Actual class balance")
        st.bar_chart(data[TARGET_COLUMN].value_counts())