# Predicting Student Dropout and Academic Success

**BITS Pilani — Work Integrated Learning Programmes Division**

A comparative study of five supervised classification algorithms on a three-class
student-outcome prediction problem, delivered as an interactive Streamlit application.

**🔗 GitHub Repository:** https://github.com/2025ac05176-cmd/ml-assignment-2
** Live Streamlit App:** https://ml-assignment-2-gxwbvc46spywyqddb8vnnf.streamlit.app/

---

## a. Problem Statement

Higher-education institutions lose a substantial share of every intake to dropout.
The cost is borne twice: the student forfeits time, tuition and momentum, and the
institution loses funding and reputation. Retention teams can intervene — tutoring,
fee relief, counselling, course transfer — but intervention capacity is finite, so
it has to be aimed at the students who actually need it.

The difficulty is that the signal is spread thinly across many weak indicators.
No single field says "this student will leave." Dropout risk emerges from the
interaction of academic performance, financial standing, family background,
admission route and the macroeconomic conditions of the enrolment year.

**Task formulation.** Given 34 attributes recorded at or shortly after enrolment,
predict a student's academic outcome at the end of the normal course duration as one
of three classes:

| Class | Meaning |
|---|---|
| `Dropout` | Left the programme without completing |
| `Enrolled` | Still enrolled, has not yet graduated |
| `Graduate` | Successfully completed the programme |

This is a **multi-class (3-class) supervised classification** problem. The
practical objective is to flag at-risk students early enough for intervention to
matter, which makes **recall on the `Dropout` class** the metric that carries real
operational weight — a missed at-risk student is a far more expensive error than a
false alarm that costs one advisory meeting.

---

## b. Dataset Description

| Property | Value |
|---|---|
| **Name** | Predict Students' Dropout and Academic Success |
| **Source** | [Kaggle — Higher Education Predictors of Student Retention](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention) |
| **Original provider** | Polytechnic Institute of Portalegre, Portugal (also UCI ML Repository, ID 697) |
| **Instances** | **4,424** (requirement: ≥ 500 ✔) |
| **Features** | **34** (requirement: ≥ 12 ✔) |
| **Target** | `Target` — 3 classes |
| **Missing values** | 0 |
| **Duplicate rows** | 0 |
| **File size** | ~460 KB |
| **Licence** | CC BY 4.0 |

### Class distribution

| Class | Count | Share |
|---|---:|---:|
| Graduate | 2,209 | 49.93 % |
| Dropout | 1,421 | 32.12 % |
| Enrolled | 794 | 17.95 % |

The imbalance ratio between the largest and smallest class is **2.78 : 1**. This is
moderate rather than extreme, but it is enough that raw accuracy is a misleading
score — a model that never predicts `Enrolled` at all still reaches ~82 % accuracy
on the other two classes. This is precisely why the assignment's inclusion of **MCC**
matters, and why the averaging strategy below was chosen deliberately.

### Feature groups

The 34 predictors fall into five natural groups:

| Group | Count | Examples |
|---|---:|---|
| **Demographic** | 7 | `Age at enrollment`, `Gender`, `Marital status`, `Nacionality`, `International`, `Displaced`, `Educational special needs` |
| **Socio-economic (family)** | 4 | `Mother's qualification`, `Father's qualification`, `Mother's occupation`, `Father's occupation` |
| **Admission / enrolment** | 6 | `Application mode`, `Application order`, `Course`, `Previous qualification`, `Daytime/evening attendance`, `Scholarship holder` |
| **Financial** | 2 | `Debtor`, `Tuition fees up to date` |
| **Academic performance** | 12 | `Curricular units 1st/2nd sem (credited / enrolled / evaluations / approved / grade / without evaluations)` |
| **Macroeconomic** | 3 | `Unemployment rate`, `Inflation rate`, `GDP` |

Note that the categorical fields (`Course`, `Marital status`, occupations,
qualifications) arrive **pre-encoded as integer codes** in the source file. No
one-hot expansion was applied — the tree-based models handle integer codes natively,
and expanding `Father's occupation` alone (46 levels) would have added 46 sparse
columns for negligible gain on the linear models.

### Preprocessing

1. **Column names stripped** of stray whitespace.
2. **Target label-encoded** with `LabelEncoder` → `Dropout=0, Enrolled=1, Graduate=2`.
3. **Stratified 80 / 20 train-test split** (`random_state=42`) → **3,539 train / 885 test** rows.
   Stratification is essential here: a random split could easily under-represent the
   794-row `Enrolled` class in the test set and destabilise its metrics.
4. **Feature scaling applied selectively**, inside each model's `Pipeline`:
   - **Scaled** (`StandardScaler`) — Logistic Regression, kNN, Naive Bayes. These are
     sensitive to feature magnitude, and the raw ranges are wildly different
     (`Age at enrollment` spans 17–70; `GDP` spans −4.06 to +3.51).
   - **Not scaled** — Decision Tree, Random Forest. Trees split on thresholds and are
     invariant to any monotone rescaling, so scaling would add cost without effect.

Wrapping the scaler and the estimator in a single `sklearn.Pipeline` means the fitted
scaler is serialised **with** the model. The Streamlit app therefore cannot apply the
wrong preprocessing to uploaded data — the most common cause of a model that scores
well in a notebook and badly in deployment.

---

## c. Repository

**🔗 https://github.com/2025ac05176-cmd/ml-assignment-2**

```
ml-assignment-2/
├── app.py                      # Streamlit application (the deployed entry point)
├── requirements.txt            # pinned dependencies
├── README.md                   # this file
├── test_data.csv               # 885-row held-out test split, for upload into the app
├── runtime.txt                 # Python version hint for Streamlit Cloud
├── .streamlit/config.toml      # theme + upload size configuration
├── data/
│   └── dataset.csv             # full 4,424-row source dataset
└── model/
    ├── train_models.py         # end-to-end training + evaluation script
    ├── ML_Assignment_2.ipynb   # notebook walkthrough with EDA
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest_ensemble.pkl
    ├── label_encoder.pkl
    ├── feature_names.pkl
    ├── comparison_table.csv
    └── metrics_summary.json
```

### Reproducing the results

```bash
git clone https://github.com/2025ac05176-cmd/ml-assignment-2.git
cd ml-assignment-2
pip install -r requirements.txt

python model/train_models.py     # retrains all 5 models, regenerates test_data.csv
streamlit run app.py             # launches the dashboard at localhost:8501
```

`random_state=42` is fixed throughout, so a rerun reproduces every number in the
table below exactly.

---

## d. Models Used

Five classification algorithms, all trained and evaluated on the identical 80/20
stratified split of the same dataset.

| # | Model | Key hyperparameters | Rationale |
|---|---|---|---|
| 1 | Logistic Regression | `C=1.0`, `lbfgs`, `max_iter=3000`, scaled | Linear baseline; establishes how much of the signal is linearly separable |
| 2 | Decision Tree | `criterion='entropy'`, `max_depth=8`, `min_samples_leaf=12` | Pruned deliberately — unbounded depth reached 100 % training accuracy and generalised worse |
| 3 | kNN | `n_neighbors=15`, `weights='distance'`, scaled | k chosen well above 1 to smooth the boundary around the noisy `Enrolled` class |
| 4 | Naive Bayes | `GaussianNB`, `var_smoothing=1e-8`, scaled | Gaussian chosen over Multinomial because the strongest predictors are continuous grades, and the data contains negative values (`GDP`, `Inflation rate`) which Multinomial NB cannot accept |
| 5 | Random Forest (Ensemble) | `n_estimators=400`, `max_features='sqrt'`, `min_samples_leaf=2`, `class_weight='balanced_subsample'` | Bagged ensemble; `balanced_subsample` directly targets the minority-class problem |

> **Note on model count.** The assignment text lists five models (Logistic Regression,
> Decision Tree, kNN, Naive Bayes, Random Forest) while the table header refers to
> "all the 6 models". This submission implements the **five explicitly named models**,
> which is the enumerated requirement in Step 2.

### Metric definitions used

Because this is a three-class problem, each metric needs an explicit averaging scheme:

- **AUC** — `roc_auc_score(..., multi_class='ovr', average='weighted')`. One-vs-rest:
  each class is scored against all others, then averaged by class support.
- **Precision / Recall / F1** — `average='weighted'`. Weighted rather than macro so
  the scores reflect performance on the population as it actually is; macro averaging
  would let the 794-row `Enrolled` class dominate a third of the score.
- **MCC** — natively multi-class, requires no averaging. It is the most honest single
  number in the table: MCC only rises when a model performs well across **all three**
  classes, and stays near zero for a model that simply predicts the majority class.

### Comparison Table

Evaluated on the held-out test set (885 rows), none of which was seen during training.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7582 | 0.8980 | 0.7401 | 0.7582 | 0.7432 | 0.5977 |
| Decision Tree | 0.7458 | 0.8721 | 0.7353 | 0.7458 | 0.7348 | 0.5781 |
| kNN | 0.7119 | 0.8457 | 0.7017 | 0.7119 | 0.6839 | 0.5209 |
| Naive Bayes | 0.6667 | 0.8133 | 0.6467 | 0.6667 | 0.6527 | 0.4431 |
| **Random Forest (Ensemble)** | **0.7695** | **0.9077** | **0.7674** | **0.7695** | **0.7663** | **0.6214** |

### Supporting diagnostics

Generalisation evidence behind the observations below. The **overfit gap** is
training accuracy minus test accuracy; **CV(5)** is 5-fold cross-validated accuracy
on the training set alone.

| Model | Train Acc | Test Acc | Overfit gap | CV(5) accuracy | Recall: Dropout | Recall: Enrolled | Recall: Graduate |
|---|---:|---:|---:|---|---:|---:|---:|
| Logistic Regression | 0.7765 | 0.7582 | +0.018 | 0.7666 ± 0.011 | 0.750 | 0.327 | 0.919 |
| Decision Tree | 0.7903 | 0.7458 | +0.045 | 0.7330 ± 0.012 | 0.683 | 0.384 | 0.916 |
| kNN | 1.0000 | 0.7119 | **+0.288** | 0.7143 ± 0.014 | 0.620 | 0.214 | 0.950 |
| Naive Bayes | 0.6861 | 0.6667 | +0.019 | 0.6815 ± 0.012 | 0.655 | 0.214 | 0.837 |
| Random Forest | 0.9768 | 0.7695 | +0.207 | **0.7742 ± 0.007** | 0.729 | **0.484** | 0.898 |

Confusion matrix for the winning Random Forest (rows = actual, columns = predicted):

|  | → Dropout | → Enrolled | → Graduate |
|---|---:|---:|---:|
| **Dropout** (284) | **207** | 40 | 37 |
| **Enrolled** (159) | 30 | **77** | 52 |
| **Graduate** (442) | 11 | 34 | **397** |

---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | The strongest non-ensemble model, and the most surprising result in the study — a linear decision boundary reaches an MCC of 0.5977, within 0.024 of the 400-tree Random Forest. Its overfit gap is the second-smallest at +0.018, so almost none of its training performance is memorisation. What this tells us is that the dropout signal is largely **additive**: failing to pay tuition, low approval counts and a poor 2nd-semester grade each push risk up roughly independently, and little is gained from modelling their interactions. Its weakness is decisiveness — it recovers only 32.7 % of `Enrolled` students, because that class sits geometrically between the other two and a linear boundary cannot carve out a middle region. Best choice if the model must be explained to a retention committee, since the coefficients read directly as risk factors. |
| **Decision Tree** | Middle of the field on every metric (MCC 0.5781). The `max_depth=8` and `min_samples_leaf=12` constraints were deliberate: an unpruned tree hit 100 % training accuracy and generalised worse, a textbook demonstration of variance in a single high-depth tree. Even pruned, its CV standard deviation (± 0.012) and its gap of +0.045 are larger than Logistic Regression's, confirming that its splits are partly fitting noise. It is the only model here that produces a human-readable decision path, which has genuine operational value — a counsellor can be shown *why* a student was flagged. Its `Enrolled` recall of 0.384 actually beats Logistic Regression's, because axis-aligned splits *can* isolate a middle region where a linear boundary cannot. |
| **kNN** | Second-weakest overall (MCC 0.5209) and the clearest failure case in the study. The train/test gap of **+0.288** is by far the largest — training accuracy is a perfect 1.0000, which is an artefact of `weights='distance'` making every training point its own nearest neighbour at zero distance. The real problem is dimensionality: with 34 features, Euclidean distances concentrate, so the 15 "nearest" neighbours are barely nearer than any random 15 points, and the local neighbourhood stops being informative. Scaling was essential and still insufficient. Its `Enrolled` recall collapses to 0.214 while `Graduate` recall is the highest of any model at 0.950 — it has effectively learned to answer "Graduate" whenever it is uncertain, which the imbalanced neighbourhoods reward. |
| **Naive Bayes** | Weakest on all six metrics (MCC 0.4431, accuracy 0.6667). The cause is structural, not a tuning failure: Gaussian NB assumes features are conditionally independent given the class, and this dataset violates that assumption comprehensively. `Curricular units 1st sem (approved)` and `Curricular units 2nd sem (approved)` correlate strongly with each other and with their respective grades, so the model counts what is essentially the same piece of evidence four or five times and becomes overconfident. It also assumes each feature is normally distributed within a class, which is plainly false for binary flags like `Debtor` and for the integer-coded categoricals. Its one redeeming property is honesty — a gap of just +0.019 means it is not overfitting at all; it is simply **underfitting** because its assumptions are wrong. Fastest model to train by a wide margin, which is its only practical argument. |
| **Random Forest (Ensemble)** | **Best model on all six metrics** — accuracy 0.7695, AUC 0.9077, MCC 0.6214 — and the only one to exceed 0.6 MCC. Bagging 400 trees over `sqrt(34) ≈ 6` random features per split cancels the variance that hurt the single Decision Tree, converting the field's most unstable learner into its most reliable one; its CV standard deviation of ± 0.007 is the tightest in the study. `class_weight='balanced_subsample'` is what produced the decisive win: `Enrolled` recall of **0.484** is more than double kNN's and Naive Bayes' 0.214, and it is the only model that predicts the minority class at a useful rate. Its +0.207 train/test gap looks alarming next to Logistic Regression's, but for a bagged ensemble a high training accuracy is expected and largely benign — the CV score confirms the test result is real, not luck of the split. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble).** It wins every one of the six required metrics outright, and its margin is widest exactly where it matters: MCC (0.6214 vs 0.5977 for the runner-up) is the metric that penalises ignoring the minority class, and `Enrolled` recall (0.484) is where every other model fails. It also has the most stable cross-validation (± 0.007). The honest caveat is cost — it is ~400× the inference work of Logistic Regression for a 2.4-point MCC gain, and it cannot explain a single prediction to a student. **If this system were actually deployed for retention triage, Random Forest is the right choice**, because the 77 `Enrolled` students it correctly identifies (versus 52 for Logistic Regression) are precisely the borderline cases where intervention changes outcomes. Where the output must be defended or audited, Logistic Regression at 98 % of the performance would be the more sensible engineering trade. |

## Streamlit Application

**🚀 Live app:** https://ml-assignment-2-gxwbvc46spywyqddb8vnnf.streamlit.app/

The dashboard implements all four required features:

| # | Required feature | Implementation |
|---|---|---|
| **a** | Dataset upload option (CSV) | Sidebar file uploader accepting the 885-row `test_data.csv`, or any CSV carrying the same 34 feature columns. **Upload is the default data source** — the app opens asking for a file rather than pre-loading one. Column presence is validated before scoring, with a named error listing anything missing, and the source banner names the uploaded file so it is unambiguous which data produced the results. A bundled fallback is available from the same control for anyone who would rather not download the CSV first. |
| **b** | Model selection dropdown | Sidebar `selectbox` across all five trained models, each with a short contextual note. An optional **Compare all 5 models** mode scores every model at once into a single ranked table with the best value per metric highlighted. |
| **c** | Display of evaluation metrics | All six metrics — Accuracy, AUC, Precision, Recall, F1, MCC — rendered as cards, plus one-vs-rest ROC curves per class and a per-class recall breakdown. |
| **d** | Confusion matrix / classification report | Paired heatmaps (raw counts and row-normalised recall) alongside the full `classification_report` with per-class precision, recall, F1 and support. |



