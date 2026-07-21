"""A tiny deterministic learner — CART + bagged random forest, numpy only.

Additive, experimental. Nothing in the shipped chart pipeline imports
this; it exists so experiments/learn_components.py can ask whether a
model TRAINED on the aligned Match Charting Project corpus beats the
hand-written rules on the two weakest chart components (serve placement
zone 4/5/6, ending type winner/net/wide/deep).

sklearn is not installed in this env (checked), so this is a small,
readable, fully deterministic multiclass classifier:

  Tree     CART with Gini impurity, greedy axis-aligned splits, a max
           depth and a min-leaf-size stop. Ties in the split search are
           broken by (feature index, threshold) order -> deterministic.
  Forest   n_trees CARTs on bootstrap resamples with sqrt-feature
           subsampling per split; a fixed integer seed drives every
           draw, so the whole fit is reproducible bit-for-bit. Class
           probability = vote fraction across trees -> a usable
           confidence for the precision/coverage curve.

No target can leak through a feature here: the harness chooses the
feature matrix and never puts the label (or the heuristic we are
replacing) into it.
"""

import numpy as np


class _Tree:
    def __init__(self, max_depth=6, min_leaf=5, n_feat=None, rng=None):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.n_feat = n_feat
        self.rng = rng or np.random.default_rng(0)
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self._cls_index = {c: i for i, c in enumerate(self.classes_)}
        yi = np.array([self._cls_index[v] for v in y])
        self.root = self._grow(X, yi, 0)
        return self

    def _leaf(self, yi):
        counts = np.bincount(yi, minlength=len(self.classes_)).astype(float)
        return {"leaf": True, "p": counts / counts.sum()}

    def _grow(self, X, yi, depth):
        if (depth >= self.max_depth or len(yi) < 2 * self.min_leaf
                or len(np.unique(yi)) == 1):
            return self._leaf(yi)
        feat, thr, gain = self._best_split(X, yi)
        if feat is None or gain <= 0:
            return self._leaf(yi)
        left = X[:, feat] <= thr
        return {"leaf": False, "feat": feat, "thr": thr,
                "L": self._grow(X[left], yi[left], depth + 1),
                "R": self._grow(X[~left], yi[~left], depth + 1)}

    def _gini(self, yi, k):
        if len(yi) == 0:
            return 0.0
        c = np.bincount(yi, minlength=k).astype(float)
        p = c / c.sum()
        return 1.0 - np.sum(p * p)

    def _best_split(self, X, yi):
        n, d = X.shape
        k = len(self.classes_)
        parent = self._gini(yi, k)
        m = self.n_feat or d
        feats = np.sort(self.rng.choice(d, size=min(m, d), replace=False))
        best = (None, None, 0.0)
        for f in feats:
            vals = np.unique(X[:, f])
            if len(vals) < 2:
                continue
            thrs = (vals[:-1] + vals[1:]) / 2.0
            for t in thrs:
                left = X[:, f] <= t
                nl = left.sum()
                if nl < self.min_leaf or n - nl < self.min_leaf:
                    continue
                gl = self._gini(yi[left], k)
                gr = self._gini(yi[~left], k)
                gain = parent - (nl * gl + (n - nl) * gr) / n
                if gain > best[2] + 1e-12:
                    best = (int(f), float(t), float(gain))
        return best

    def predict_proba(self, X):
        out = np.zeros((len(X), len(self.classes_)))
        for i, row in enumerate(X):
            node = self.root
            while not node["leaf"]:
                node = node["L"] if row[node["feat"]] <= node["thr"] else node["R"]
            out[i] = node["p"]
        return out


class RandomForest:
    """Deterministic bagged CART forest for small multiclass problems."""

    def __init__(self, n_trees=200, max_depth=6, min_leaf=4, seed=0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n, d = X.shape
        n_feat = max(1, int(np.sqrt(d)))
        rng = np.random.default_rng(self.seed)
        self.trees = []
        for _ in range(self.n_trees):
            idx = rng.integers(0, n, size=n)          # bootstrap
            t = _Tree(self.max_depth, self.min_leaf, n_feat,
                      np.random.default_rng(rng.integers(0, 2**32)))
            t.fit(X[idx], y[idx])
            self.trees.append(t)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        agg = np.zeros((len(X), len(self.classes_)))
        cls_i = {c: i for i, c in enumerate(self.classes_)}
        for t in self.trees:
            tp = t.predict_proba(X)                    # over t.classes_
            for j, c in enumerate(t.classes_):
                agg[:, cls_i[c]] += tp[:, j]
        return agg / self.n_trees

    def predict(self, X):
        p = self.predict_proba(X)
        return self.classes_[p.argmax(axis=1)]
