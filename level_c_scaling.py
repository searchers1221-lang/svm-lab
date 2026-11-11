# === cell c3
import time
from sklearn.datasets import make_classification
from sklearn.svm import SVC, LinearSVC

for n in [2000, 4000, 8000, 16000]:
    Xn, yn = make_classification(n_samples=n, n_features=20, random_state=42)
    t0 = time.perf_counter(); SVC(kernel="rbf").fit(Xn, yn); rbf = time.perf_counter() - t0
    t0 = time.perf_counter(); LinearSVC().fit(Xn, yn); lin = time.perf_counter() - t0
    print(f"n={n:<6} rbf SVC: {rbf:5.2f} с   LinearSVC: {lin:5.2f} с")
