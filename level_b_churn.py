# === cell b1
import time
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

df = pd.read_csv("data/o_churn_synthetic.csv")
X = df.drop(columns=["churn", "subscriber_code"])
y = df["churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

prep = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), ["region", "tariff"])],
    remainder=StandardScaler())
svm = make_pipeline(prep, SVC(kernel="rbf"))
# === cell b2
grid = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [0.001, 0.01, 0.1, 1]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(svm, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
t0 = time.perf_counter()
search.fit(X_train, y_train)
print(f"лучшие: {search.best_params_}  AUC CV={search.best_score_:.3f}"
      f"  поиск: {time.perf_counter() - t0:.0f} с")
# === cell b3
table = pd.DataFrame(search.cv_results_).pivot(
    index="param_svc__C", columns="param_svc__gamma", values="mean_test_score")
print(table.round(3))
# === cell b4
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

tree_prep = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), ["region", "tariff"])],
    remainder="passthrough")
models = {
    "SVM rbf (лучшая)": make_pipeline(prep, SVC(kernel="rbf", **{
        k.replace("svc__", ""): v for k, v in search.best_params_.items()})),
    "SVM linear": make_pipeline(prep, SVC(kernel="linear", C=1)),
    "Логрегрессия": make_pipeline(prep, LogisticRegression(max_iter=1000)),
    "Случайный лес": make_pipeline(tree_prep, RandomForestClassifier(
        n_estimators=100, random_state=42)),
    "XGBoost": make_pipeline(tree_prep, XGBClassifier(
        n_estimators=127, learning_rate=0.05, max_depth=3, random_state=42)),
}
for name, pipe in models.items():
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    sec = time.perf_counter() - t0
    f1 = f1_score(y_test, pipe.predict(X_test))
    scores = pipe.decision_function(X_test) if hasattr(pipe, "decision_function") \
        else pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, scores)
    print(f"{name:<18} F1 test={f1:.3f}  AUC test={auc:.3f}  {sec:.2f} с")
