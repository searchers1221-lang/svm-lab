# === cell a1
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X2, y2 = make_moons(n_samples=300, noise=0.25, random_state=42)
X2 = StandardScaler().fit_transform(X2)
print("объектов:", len(X2), " классы:", np.bincount(y2))
# === cell a2
def plot_boundary(model, X, y, title, fname):
    xx, yy = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(5, 4))
    plt.contourf(xx, yy, zz, alpha=0.25, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=18, edgecolor="k")
    sv = model.support_vectors_
    plt.scatter(sv[:, 0], sv[:, 1], s=70, facecolors="none", edgecolors="k")
    plt.title(f"{title}: train acc={model.score(X, y):.3f}, SV={len(sv)}")
    plt.tight_layout(); plt.savefig(fname, dpi=120); plt.close()

# === cell a2b
lin = SVC(kernel="linear", C=1).fit(X2, y2)
plot_boundary(lin, X2, y2, "linear, C=1", "images/svm_linear.png")
print("linear: acc=%.3f  опорных=%d" % (lin.score(X2, y2), len(lin.support_vectors_)))
# === cell a3
rbf = SVC(kernel="rbf", C=1, gamma="scale").fit(X2, y2)
plot_boundary(rbf, X2, y2, "rbf, C=1", "images/svm_rbf.png")
print("rbf:    acc=%.3f  опорных=%d" % (rbf.score(X2, y2), len(rbf.support_vectors_)))
# === cell a5
for C in [0.01, 1, 100]:
    m = SVC(kernel="rbf", C=C, gamma="scale").fit(X2, y2)
    plot_boundary(m, X2, y2, f"rbf, C={C}", f"images/svm_rbf_C{C}.png")
    print(f"C={C:<5} acc={m.score(X2, y2):.3f}  опорных={len(m.support_vectors_)}")
