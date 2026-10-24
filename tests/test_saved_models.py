"""Saved defenses give the same scores as the models that were just trained (PRETRAINED=1 reproduces training)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from gencpm.defenses.baselines import load_gbdt, save_gbdt  # noqa: E402
from gencpm.defenses.pild import load_pild, save_pild, train_pild  # noqa: E402


def _data(n, rng):
    X = rng.normal(size=(n, 20)).astype(np.float32)
    return X, (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float32)


def test_pild_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    (Xtr, ytr), (Xva, yva) = _data(3000, rng), _data(500, rng)
    net, _ = train_pild(Xtr, ytr, Xva, yva, seed=0, device=torch.device("cpu"), epochs=3)
    head = HistGradientBoostingClassifier(max_iter=20, random_state=0).fit(Xtr[:, :4], ytr)
    save_pild(tmp_path, 2, 1, net, head)
    net2, head2 = load_pild(tmp_path, 2, 1, torch.device("cpu"))
    np.testing.assert_array_equal(net(Xva), net2(Xva))
    np.testing.assert_array_equal(head.predict_proba(Xva[:, :4]), head2.predict_proba(Xva[:, :4]))


def test_gbdt_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    X, y = _data(2000, rng)
    clf = HistGradientBoostingClassifier(max_iter=30, early_stopping=True, random_state=0).fit(X, y)
    save_gbdt(tmp_path, 0, 2, clf)
    np.testing.assert_array_equal(clf.predict_proba(X), load_gbdt(tmp_path, 0, 2).predict_proba(X))
