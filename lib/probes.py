"""Linear probes with leave-one-rule-set-out scoring (from the study's detect.py)."""
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEEDS = range(5)  # rule sets h3L0-4


def clf():
    return make_pipeline(StandardScaler(), RidgeClassifier(alpha=1e4))

def loro(X, y, seed, fit=clf, Xt=None, yt=None, seed_t=None):
    """Leave-one-rule-set-out scores; with Xt, train on X (other rule sets) and score Xt (held-out rule set)."""
    if Xt is None:
        Xt, yt, seed_t = X, y, seed
    s = np.zeros(len(yt))
    for k in SEEDS:
        tr, te = seed != k, seed_t == k
        if te.any():
            s[te] = fit().fit(X[tr], y[tr]).decision_function(Xt[te])
    return roc_auc_score(yt, s)
