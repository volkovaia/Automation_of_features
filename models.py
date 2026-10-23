"""Обёртки над XGBoost (CUDA) + оценка"""
from __future__ import annotations

import numpy as np
import xgboost as xgb

from data import full_metrics
from features import FeatureStore, Spec

DEVICE = "cuda"


def set_device(d: str):
    global DEVICE
    DEVICE = d


def fit_xgb(Xtr, ytr, Xva, yva, rounds=3000, es=60, eta=0.05, depth=8, max_rows=None, seed=0):
    if max_rows and len(Xtr) > max_rows:
        sel = np.random.default_rng(seed).choice(len(Xtr), max_rows, replace=False)
        Xtr, ytr = Xtr[sel], ytr[sel]
    dtr = xgb.QuantileDMatrix(Xtr, ytr, max_bin=256)
    dva = xgb.QuantileDMatrix(Xva, yva, ref=dtr)
    p = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", device=DEVICE,
             eta=eta, max_depth=depth, min_child_weight=5, subsample=0.8, colsample_bytree=0.6,
             reg_lambda=2.0, seed=seed)
    return xgb.train(p, dtr, rounds, evals=[(dva, "val")], early_stopping_rounds=es, verbose_eval=False)


def predict(bst, X) -> np.ndarray:
    bst.set_param({"device": DEVICE})
    return bst.predict(xgb.DMatrix(X), iteration_range=(0, bst.best_iteration + 1))


def _mean_dict(ds: list[dict]) -> dict:
    return {k: float(np.nanmean([d.get(k, np.nan) for d in ds])) for k in ds[0]}


def run_set(store: FeatureStore, specs: list[Spec], train_ids, val_id, test_ids, quick=False, **kw):
    """Бустинг на наборе признаков. test_ids - список тестовых дней (или None). Возвращает метрики val, test по дням, среднее по дням и per-user таблицы (_per_user)."""
    Xtr = store.matrix(specs, train_ids); ytr = store.labels(train_ids)
    Xva = store.matrix(specs, [val_id]); yva = store.labels([val_id])
    if quick:
        kw = dict(rounds=300, es=20, eta=0.15, depth=7, max_rows=1_500_000, **kw)
    bst = fit_xgb(Xtr, ytr, Xva, yva, **kw)
    out = {"n_features": len(specs), "best_iter": int(bst.best_iteration),
           "val": full_metrics(store.snaps[val_id], predict(bst, Xva))}
    if test_ids:
        days, tabs = [], []
        for ti in test_ids:
            m_, tab = full_metrics(store.snaps[ti], predict(bst, store.matrix(specs, [ti])), return_table=True)
            days.append(m_); tabs.append(tab)
        out["test_days"], out["test"], out["_per_user"] = days, _mean_dict(days), tabs
    imp = bst.get_score(importance_type="total_gain")
    out["importance"] = {specs[int(k[1:])].name: v for k, v in imp.items()}
    return out