"""Теория до обучения бустинга: какое ядро памяти и какой набор признаков лучше.

Модель. Событие канала c в момент t_k пары/группы g даёт вклад в целевую переменную через неизвестное ядро памяти g(s), s = T - t_k. Любой признак-"сумма по ядру"
    f_w = sum_k w(T - t_k) линеен по гистограмме лагов X_b (число событий в лаг-бине b). Поэтому всё считается из достаточных статистик по лаг-бинам:
    G = X~' X~   (ковариации бинов),   g = X~' y~,   syy = y~' y~,
где ~ — центрирование внутри пользователя (нас интересует ранжирование внутри юзера).
Тогда для любого набора признаков W (строки = веса по бинам)
    R2(W) = b' A^{-1} b / syy,   A = W G W',  b = W g - доля объяснённой (линейно) дисперсии, считается быстро, без бустинга.
  * ядро Винера–Хопфа = все бины как признаки (верхняя граница для сумм по ядрам);
  * оценка полупериода: argmax R2 по сетке экспонент, CI - бутстрэп по пользователям
    (пуассоновы веса);
  * честная проверка: коэффициенты по train-снапшотам, R2 считается на val (out-of-sample);
  * жадный детерминированный отбор ядер по приросту R2 (замена agent_search).
Аналитика для OU/Пуассона: rho^2 между ядрами e^{-lam s} и e^{-gam s} равно
4 lam gam/(lam+gam)^2, между окном w и e^{-gam s}: 2(1-e^{-gam w})^2/(gam w).

Запуск:  python theory.py --dataset synthetic --device cpu --users 300 --boot 4
         python theory.py --dataset yambda --size 50m --users 3000 --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from data import CHANNELS, build_splits, load_generic, load_yambda, make_synthetic
from features import (CH_IDX, FAMS, HL_GRID, LN2, WIN_GRID, FeatureStore, Spec, manual_set, mk,
                      rect8_set)

C_ = len(CHANNELS)


def rho2_exp_exp(lam, gam):
    return 4 * lam * gam / (lam + gam) ** 2


def rho2_rect_exp(w, gam):
    return 2 * (1 - np.exp(-gam * w)) ** 2 / (gam * w)


def analytic_tables(true_hl=10.0):
    gam = LN2 / true_hl
    t = {"true_half_life": true_hl,
         "exp_mismatch": {str(h): float(rho2_exp_exp(LN2 / h, gam)) for h in (3, 5, 7, 10, 14, 20, 30)},
         "rect_window": {str(w): float(rho2_rect_exp(w, gam)) for w in (1, 3, 7, 10, 14, 18, 30, 60)}}
    best_w = 1.26 / gam
    t["rect_best_window"] = (float(best_w), float(rho2_rect_exp(best_w, gam)))
    return t


# лаг-бины и векторы ядер
def make_edges(B: int, max_hist: float, lo: float = 0.02) -> np.ndarray:
    return np.concatenate([[0.0], np.geomspace(lo, max_hist, B)])          # B+1 границ


def kernel_vec(kind: str, p: float, edges: np.ndarray) -> np.ndarray:
    """Среднее значение ядра по каждому лаг-бину (события равномерно внутри бина)."""
    a, b = edges[:-1], edges[1:]; d = b - a
    if kind == "exp":
        lam = LN2 / p
        return (np.exp(-lam * a) - np.exp(-lam * b)) / (lam * d)
    if kind == "rect":
        return np.clip((p - a) / d, 0.0, 1.0)
    raise ValueError(kind)


def spec_row(spec: Spec, edges: np.ndarray) -> np.ndarray:
    B = len(edges) - 1
    v = np.zeros(len(FAMS) * C_ * B)
    o = (FAMS.index(spec.fam) * C_ + CH_IDX[spec.ch]) * B
    v[o:o + B] = kernel_vec(spec.kind, spec.p, edges)
    return v


def rows(specs, edges) -> np.ndarray:
    specs = [s for s in specs if s.kind in ("rect", "exp")]
    return np.stack([spec_row(s, edges) for s in specs])


# достаточные статистики
class Stats:
    pass


def _design(store: FeatureStore, si: int, edges_in: torch.Tensor, B: int) -> torch.Tensor:
    """[N, F*C*B] - гистограммы лагов по всем семействам и каналам для кандидатов снапшота."""
    dev, parts = store.dev, []
    for fam in FAMS:
        m = store.build_matched(si, fam)
        bins = torch.bucketize(m.dt, edges_in)
        flat = torch.zeros(m.nq * C_ * B, device=dev)
        for c in range(C_):
            mk_ = m.val[:, c] > 0
            idx = m.q[mk_] * (C_ * B) + c * B + bins[mk_]
            flat.index_add_(0, idx, torch.ones(idx.shape[0], device=dev))
        parts.append(flat.view(m.nq, C_ * B)[m.cand_q])
        del m, flat
    return torch.cat(parts, 1)


def _demean(X, y, cu, dev):
    _, inv = np.unique(cu, return_inverse=True)
    gid = torch.from_numpy(inv).to(dev)
    cnt = torch.bincount(gid).float()
    mx = torch.zeros(len(cnt), X.shape[1], device=dev).index_add_(0, gid, X) / cnt[:, None]
    my = torch.zeros(len(cnt), device=dev).index_add_(0, gid, y) / cnt
    return X - mx[gid], y - my[gid]


def compute_stats(store: FeatureStore, tr_ids, va_id, B=20, M=16, seed=0, check=True, log=print,
                  n_strata: int = 0) -> Stats:
    dev = store.dev
    edges = make_edges(B, store.max_hist)
    edges_in = torch.tensor(edges[1:-1], dtype=torch.float32, device=dev)
    D = len(FAMS) * C_ * B
    rng = np.random.default_rng(seed)
    Wb = np.ones((M + 1, store.ev.n_users), dtype=np.float32)
    Wb[1:] = rng.poisson(1.0, size=(M, store.ev.n_users))
    Wb_t = torch.from_numpy(Wb).to(dev)
    S = Stats(); S.edges, S.B, S.D, S.M = edges, B, D, M
    S.G = np.zeros((M + 1, D, D)); S.g = np.zeros((M + 1, D)); S.syy = np.zeros(M + 1)
    S.Gv = np.zeros((D, D)); S.gv = np.zeros(D); S.syyv = 0.0
    S.n_rows = 0
    S.n_strata = n_strata
    if n_strata > 0:       # страты пользователей по активности (число событий за 30 дней до снапшотов train)
        act = np.zeros(store.ev.n_users)
        for si in tr_ids:
            T_ = store.snaps[si].T
            lo_ = np.searchsorted(store.t_np, T_ - 30.0); hi_ = np.searchsorted(store.t_np, T_)
            act += np.bincount(store.ev.u[lo_:hi_], minlength=store.ev.n_users)
        jit = act + np.random.default_rng(seed).random(len(act)) * 1e-6      # ранги без пустых страт при совпадениях
        rank = np.argsort(np.argsort(jit))
        S.strata = np.minimum((rank * n_strata) // len(rank), n_strata - 1)
        S.Gs = np.zeros((n_strata, M + 1, D, D)); S.gs = np.zeros((n_strata, M + 1, D))
        S.syys = np.zeros((n_strata, M + 1))
    t0 = time.time()
    for si in list(tr_ids) + [va_id]:
        snap = store.snaps[si]
        X = _design(store, si, edges_in, B)
        if check and si == tr_ids[0]:
            _selfcheck(store, X, edges, si, log)
        y = torch.from_numpy(snap.y).to(dev)
        X, y = _demean(X, y, snap.cu, dev)
        if si == va_id:
            S.Gv = (X.T @ X).double().cpu().numpy(); S.gv = (X.T @ y).double().cpu().numpy()
            S.syyv = float((y * y).sum())
        else:
            cu_t = torch.from_numpy(snap.cu.astype(np.int64)).to(dev)
            for m in range(M + 1):
                w = Wb_t[m][cu_t]
                Xw = X * w[:, None]
                S.G[m] += (Xw.T @ X).double().cpu().numpy()
                S.g[m] += (Xw.T @ y).double().cpu().numpy()
                S.syy[m] += float((w * y * y).sum())
            S.n_rows += len(y)
            if n_strata > 0:
                st_t = torch.from_numpy(S.strata[snap.cu].astype(np.int64)).to(dev)
                parts = []
                for s_ in range(n_strata):
                    mk_ = st_t == s_
                    parts.append((X[mk_], y[mk_], Wb_t[:, cu_t[mk_]]))
                for s_, (Xs, ys, Ws) in enumerate(parts):
                    for m in range(M + 1):
                        Xw = Xs * Ws[m][:, None]
                        S.Gs[s_, m] += (Xw.T @ Xs).double().cpu().numpy()
                        S.gs[s_, m] += (Xw.T @ ys).double().cpu().numpy()
                        S.syys[s_, m] += float((Ws[m] * ys * ys).sum())
                del parts
        del X, y
        log(f"[theory] снапшот {si}: статистики готовы ({time.time()-t0:.0f}s)")
    return S


def _selfcheck(store, X, edges, si, log):
    """Проверка аппроксимации "события равномерны внутри бина": бин-признак vs точный признак."""
    for sp in (mk("pair", "good", "exp", 7), mk("uart", "good", "rect", 14)):
        store.ensure([sp])
        exact = store.cols[si][sp]
        approx = (X.cpu().numpy() @ spec_row(sp, edges)).astype(np.float64)
        r = np.corrcoef(exact, approx)[0, 1] if exact.std() > 0 and approx.std() > 0 else float("nan")
        log(f"[theory] selfcheck {sp.name}: corr(точный, по бинам) = {r:.4f}")


# R2 для наборов признаков
def _solve(A, b, ridge_rel=1e-8):
    r = ridge_rel * max(np.trace(A) / len(A), 1e-12)
    return np.linalg.solve(A + r * np.eye(len(A)), b)


def r2_in(W, G, g, syy, ridge_rel=1e-8):
    A, b = W @ G @ W.T, W @ g
    return float(b @ _solve(A, b, ridge_rel) / syy)


def r2_oos(W, S: Stats, ridge_rel=1e-8):
    coef = _solve(W @ S.G[0] @ W.T, W @ S.g[0], ridge_rel)
    sse = S.syyv - 2 * coef @ (W @ S.gv) + coef @ (W @ S.Gv @ W.T) @ coef
    return float(1 - sse / S.syyv)


def single_curve(S: Stats, fam: str, ch: str, kind: str, grid):
    """R2 одного признака-ядра по сетке параметров; [M+1, len(grid)] (m=0 полные данные)."""
    out = np.zeros((S.M + 1, len(grid)))
    rowsw = np.stack([spec_row(mk(fam, ch, kind, p), S.edges) for p in grid])
    for m in range(S.M + 1):
        A = np.einsum("kd,de,ke->k", rowsw, S.G[m], rowsw)
        b = rowsw @ S.g[m]
        out[m] = b ** 2 / np.maximum(A, 1e-12) / S.syy[m]
    return out


# жадный отбор
def candidate_pool(hl_pool=None, win_pool=None) -> list[Spec]:
    hl_pool = hl_pool if hl_pool is not None else np.round(np.geomspace(0.05, 150, 30), 3)
    win_pool = win_pool if win_pool is not None else np.round(np.geomspace(0.1, 110, 20), 3)
    return ([mk(f, c, "exp", h) for f in FAMS for c in CHANNELS for h in hl_pool]
            + [mk(f, c, "rect", w) for f in FAMS for c in CHANNELS for w in win_pool])


def random_specs(rng: np.random.Generator, k: int, exclude_fams=("user",)) -> list[Spec]:
    """Контроль: k случайных ядер из того же пула (без заведомо бесполезного семейства user)."""
    pool = [s for s in candidate_pool() if s.fam not in exclude_fams]
    return [pool[i] for i in rng.choice(len(pool), size=k, replace=False)]


def predict_set(S: Stats, specs: list[Spec]) -> dict:
    """Предсказанный (линейный) R2 набора признаков — только ядерные признаки rect/exp."""
    W = rows(specs, S.edges)
    return {"r2_in": r2_in(W, S.G[0], S.g[0], S.syy[0]), "r2_oos": r2_oos(W, S)}


def topk_single(S: Stats, k: int, exclude_fams=("user",)) -> list[Spec]:
    """Простой отбор: k ядер с наибольшим одиночным R2 из пула, без жадной коррекции на избыточность."""
    cand = [s for s in candidate_pool() if s.fam not in exclude_fams]
    W = rows(cand, S.edges)
    A = np.einsum("kd,de,ke->k", W, S.G[0], W); b = W @ S.g[0]
    r2 = b ** 2 / np.maximum(A, 1e-12) / S.syy[0]
    return [cand[i] for i in np.argsort(-r2)[:k]]


def resample_scales(rng: np.random.Generator, specs: list[Spec]) -> list[Spec]:
    """Те же (семейство, канал, тип ядра), но масштаб p случайный (логарифмически равномерно)."""
    out = []
    for s in specs:
        lo, hi = (0.05, 150.0) if s.kind == "exp" else (0.1, 110.0)
        out.append(mk(s.fam, s.ch, s.kind, round(float(np.exp(rng.uniform(np.log(lo), np.log(hi)))), 3)))
    return out


def greedy_select(S: Stats, kmax=24, min_gain=5e-4, hl_pool=None, win_pool=None, log=print):
    cand = candidate_pool(hl_pool, win_pool)
    Wc = rows(cand, S.edges)
    A = Wc @ S.G[0] @ Wc.T; b = Wc @ S.g[0]; dA = np.diag(A).copy()
    sel, gains, hist = [], [], []
    for _ in range(kmax):
        if sel:
            Ass = A[np.ix_(sel, sel)]
            Ainv = np.linalg.inv(Ass + 1e-9 * np.trace(Ass) / len(sel) * np.eye(len(sel)))
            u = A[:, sel] @ Ainv
            num = b - u @ b[sel]; den = dA - np.einsum("ks,ks->k", u, A[:, sel])
        else:
            num, den = b, dA
        gain = num ** 2 / np.maximum(den, 1e-9 * dA.max()) / S.syy[0]
        gain[sel] = -1; gain[den <= 1e-9 * dA] = -1
        j = int(np.argmax(gain))
        if gain[j] < min_gain:
            break
        sel.append(j); gains.append(float(gain[j]))
        Wsel = Wc[sel]
        hist.append({"k": len(sel), "added": cand[j].name, "gain": float(gain[j]),
                     "r2_in": r2_in(Wsel, S.G[0], S.g[0], S.syy[0]), "r2_oos": r2_oos(Wsel, S)})
        log(f"[select] {len(sel):2d} {cand[j].name:28s} +R2={gain[j]:.5f}  R2_in={hist[-1]['r2_in']:.4f} "
            f"R2_oos={hist[-1]['r2_oos']:.4f}")
    return [cand[j] for j in sel], hist


#  проверка двух временных шкал
def _single_r2(W, G, g, syy):
    """R2 каждого одиночного признака (строки W) для всех бутстрэп-реплик: [M+1, K]."""
    out = np.zeros((G.shape[0], len(W)))
    for m in range(G.shape[0]):
        out[m] = (W @ g[m]) ** 2 / np.maximum(np.einsum("kd,de,ke->k", W, G[m], W), 1e-12) / syy[m]
    return out


def two_scale_test(S: Stats, fam: str, ch: str, hl_grid=None) -> dict:
    """Одна экспонента против смеси двух (по сетке пар), для одного (семейство, канал).
    Вложенность -> R2_in смеси >= одиночной всегда, поэтому главная мера — прирост вне выборки (val) и его сравнение с «второй шкалой наугад»; плюс доля зазора до ядра Винера этого блока."""
    hl = np.geomspace(0.05, 200, 40) if hl_grid is None else np.asarray(hl_grid)
    W = np.stack([spec_row(mk(fam, ch, "exp", h), S.edges) for h in hl])
    G, g, syy = S.G[0], S.g[0], S.syy[0]
    A = W @ G @ W.T; b = W @ g; d = np.diag(A)
    single = b ** 2 / np.maximum(d, 1e-12) / syy
    i1 = int(np.argmax(single))
    aii, ajj = d[:, None], d[None, :]
    den = aii * ajj - A ** 2
    with np.errstate(all="ignore"):
        r2p = (ajj * b[:, None] ** 2 - 2 * A * b[:, None] * b[None, :] + aii * b[None, :] ** 2) / den / syy
    r2p[~np.isfinite(r2p)] = np.nan
    r2p[den < 1e-9 * aii * ajj] = np.nan
    np.fill_diagonal(r2p, np.nan)
    i2, j2 = np.unravel_index(np.nanargmax(r2p), r2p.shape)
    ia, ib = sorted((int(i2), int(j2)))
    r2_single_in, r2_pair_in = float(single[i1]), float(r2p[ia, ib])
    B = S.B; o = (FAMS.index(fam) * C_ + CH_IDX[ch]) * B
    Wblk = np.zeros((B, S.D)); Wblk[np.arange(B), o + np.arange(B)] = 1.0
    r2_wien_in = r2_in(Wblk, G, g, syy, 1e-10)
    oos1 = r2_oos(W[[i1]], S)
    oos_pair = r2_oos(W[[ia, ib]], S)
    ctrl = [r2_oos(W[[i1, j]], S) - oos1 for j in range(len(hl)) if j != i1]
    # бутстрэп прироста при фиксированном выборе (i1; ia, ib): in-sample на пересэмплированных пользователях
    boot = []
    for m in range(1, S.M + 1):
        a1 = (W[i1] @ S.g[m]) ** 2 / (W[i1] @ S.G[m] @ W[i1]) / S.syy[m]
        boot.append(r2_in(W[[ia, ib]], S.G[m], S.g[m], S.syy[m]) - a1)
    Wp = W[[ia, ib]]
    coef = np.linalg.solve(Wp @ G @ Wp.T + 1e-9 * np.eye(2), Wp @ g)
    gap = r2_wien_in - r2_single_in
    return {"h_single": float(hl[i1]), "h_pair": [float(hl[ia]), float(hl[ib])],
            "coef_pair": [float(c) for c in coef],
            "r2_single_in": r2_single_in, "r2_pair_in": r2_pair_in, "r2_wiener_in": float(r2_wien_in),
            "gain_in": r2_pair_in - r2_single_in,
            "gain_in_boot_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot else None,
            "gain_oos": float(oos_pair - oos1), "gain_oos_random2": float(np.mean(ctrl)),
            "gap_closed": float((r2_pair_in - r2_single_in) / gap) if gap > 1e-9 else None}


def strata_scales(S: Stats, fam: str, ch: str, hl_grid=None) -> list:
    """Оптимум одной экспоненты (и CI) отдельно по стратам активности."""
    if not getattr(S, "n_strata", 0):
        return []
    hl = np.geomspace(0.05, 200, 40) if hl_grid is None else np.asarray(hl_grid)
    W = np.stack([spec_row(mk(fam, ch, "exp", h), S.edges) for h in hl])
    out = []
    for s_ in range(S.n_strata):
        if S.syys[s_, 0] < 1e-9:
            out.append({"stratum": s_, "h_hat": float("nan"), "ci": [float("nan")] * 2, "r2": float("nan"),
                        "curve": [float("nan")] * len(hl)}); continue
        c = _single_r2(W, S.Gs[s_], S.gs[s_], S.syys[s_])
        i = int(np.argmax(c[0]))
        boots = hl[np.argmax(c[1:], axis=1)] if S.M > 0 else np.array([np.nan])
        out.append({"stratum": s_, "h_hat": float(hl[i]), "ci": [float(x) for x in np.percentile(boots, [2.5, 97.5])],
                    "r2": float(c[0, i]), "curve": (c[0] / c[0].max()).tolist()})
    return out


def run_two_scale(S: Stats, out_dir: str, fams=("pair", "uart", "ualb", "item", "art"), log=print) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    res = {}
    log("\n[scales] одна экспонента против смеси двух (прирост R² вне выборки, vs «вторая шкала наугад»):")
    for fam in fams:
        for ch in CHANNELS:
            r = two_scale_test(S, fam, ch); r["strata"] = strata_scales(S, fam, ch)
            res[f"{fam}.{ch}"] = r
            st = " | ".join(f"s{x['stratum']}: {x['h_hat']:.2f}" for x in r["strata"])
            log(f"  {fam:5s}.{ch:6s} h1={r['h_single']:7.2f}  пара={r['h_pair'][0]:.2f}/{r['h_pair'][1]:.1f}  "
                f"gain_oos={r['gain_oos']:+.5f} (наугад {r['gain_oos_random2']:+.5f})  "
                f"закрыто зазора={r['gap_closed'] if r['gap_closed'] is None else round(r['gap_closed'], 2)}  {st}")
    json.dump(res, open(f"{out_dir}/two_scale.json", "w"), indent=1)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        hl = np.geomspace(0.05, 200, 40); colors = ["tab:blue", "tab:orange", "tab:red", "tab:green"]
        keys = [k for k in ("uart.good", "uart.listen", "pair.good", "ualb.good") if k in res and res[k]["strata"]]
        if keys:
            fig, axes = plt.subplots(1, len(keys), figsize=(4.5 * len(keys), 3.8), sharey=True)
            for ax, k in zip(np.atleast_1d(axes), keys):
                for st in res[k]["strata"]:
                    ax.plot(hl, st["curve"], color=colors[st["stratum"] % 4], label=f"активность: страта {st['stratum']}")
                ax.set_xscale("log"); ax.set_title(k); ax.set_xlabel("полупериод, дни"); ax.grid(alpha=.3)
            np.atleast_1d(axes)[0].set_ylabel("R² / max R²"); np.atleast_1d(axes)[0].legend(fontsize=7)
            plt.tight_layout(); plt.savefig(f"{out_dir}/strata_curves.png", dpi=130); plt.close()
    except Exception as e:  # noqa: BLE001
        log(f"[scales] график пропущен: {e}")
    return res


# полный отчёт
def run_theory(store: FeatureStore, tr_ids, va_id, out_dir: str, B=20, M=16, seed=0, kmax=24,
               truth_hl: float | None = None, log=print, min_gain: float = 5e-4, return_stats: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    S = compute_stats(store, tr_ids, va_id, B, M, seed, log=log)
    rep: dict = {"B": B, "M": M, "n_train_rows": S.n_rows, "analytic": analytic_tables(truth_hl or 10.0)}

    hl_grid = np.geomspace(0.05, 200, 60); win_grid = np.geomspace(0.1, 110, 40)
    curves = {}
    log("\n[theory] оценка полупериода: argmax R2 по экспонентам (CI - бутстрэп по юзерам)")
    for fam in FAMS:
        for ch in CHANNELS:
            ce = single_curve(S, fam, ch, "exp", hl_grid); cr = single_curve(S, fam, ch, "rect", win_grid)
            ie = int(np.argmax(ce[0])); ir = int(np.argmax(cr[0]))
            boots = hl_grid[np.argmax(ce[1:], axis=1)] if S.M > 0 else np.array([np.nan])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            curves[f"{fam}.{ch}"] = {
                "hl_hat": float(hl_grid[ie]), "hl_ci": [float(lo), float(hi)], "r2_exp_best": float(ce[0, ie]),
                "win_best": float(win_grid[ir]), "r2_rect_best": float(cr[0, ir]),
                "r2_exp_curve": ce[0].tolist(), "r2_rect_curve": cr[0].tolist()}
            log(f"  {fam:5s}.{ch:6s} hl^={hl_grid[ie]:7.2f} CI[{lo:6.2f},{hi:7.2f}]  R2 exp={ce[0, ie]:.4f} "
                f"| лучшее окно={win_grid[ir]:6.2f} R2 rect={cr[0, ir]:.4f}")
    rep["curves"] = curves; rep["hl_grid"] = hl_grid.tolist(); rep["win_grid"] = win_grid.tolist()

    # наборы признаков, которые потом реально обучаем
    best1 = [mk(f, c, "exp", curves[f"{f}.{c}"]["hl_hat"]) for f in FAMS for c in CHANNELS]
    sets = {"manual_1_7_30": [s for s in manual_set() if s.kind in ("rect", "exp")],
            "rect_8scales": [s for s in rect8_set() if s.kind in ("rect", "exp")],
            "exp_8scales": [mk(f, c, "exp", h) for f in FAMS for c in CHANNELS for h in HL_GRID],
            "exp_1_best": best1}
    selected, hist = greedy_select(S, kmax=kmax, min_gain=min_gain, log=log)
    sets["theory_greedy"] = selected
    Wall = np.eye(S.D)                                  # ядро Винера: все лаг-бины как признаки
    res_sets, Wm = {}, rows(sets["manual_1_7_30"], S.edges)
    base_boot = np.array([r2_in(Wm, S.G[m], S.g[m], S.syy[m]) for m in range(S.M + 1)])
    for name, sp in sets.items():
        W = rows(sp, S.edges)
        boot = np.array([r2_in(W, S.G[m], S.g[m], S.syy[m]) for m in range(S.M + 1)])
        d = boot[1:] - base_boot[1:] if S.M > 0 else np.array([np.nan])
        res_sets[name] = {"n_kernel_features": len(W), "r2_in": float(boot[0]), "r2_oos": r2_oos(W, S),
                          "delta_vs_manual_in": float(boot[0] - base_boot[0]),
                          "delta_ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    # Верхняя граница: все лаг-бины как признаки. Регуляризация минимальна (1e-10), иначе из-за сильной коллинеарности бинов граница занижается (так было в первом запуске на Yambda).
    res_sets["wiener_all_bins"] = {"n_kernel_features": S.D, "r2_in": r2_in(Wall, S.G[0], S.g[0], S.syy[0], 1e-10),
                                   "r2_oos": r2_oos(Wall, S, 1e-10)}
    top = max(r["r2_in"] for k, r in res_sets.items() if k != "wiener_all_bins")
    if res_sets["wiener_all_bins"]["r2_in"] < top - 1e-6:
        log("[theory][warn] верхняя граница ниже одного из наборов: численная неустойчивость, "
            "уменьшите --bins или проверьте данные")
    # прямые парные сравнения (общие бутстрэп-веса пользователей -> корректные CI разности)
    def _boot(name):
        W = rows(sets[name], S.edges)
        return np.array([r2_in(W, S.G[m], S.g[m], S.syy[m]) for m in range(S.M + 1)])
    paired = {}
    for a_, b_ in [("exp_8scales", "rect_8scales"), ("exp_8scales", "manual_1_7_30"),
                   ("theory_greedy", "manual_1_7_30"), ("theory_greedy", "exp_8scales")]:
        ba, bb = _boot(a_), _boot(b_)
        d = (ba - bb)[1:] if S.M > 0 else np.array([np.nan])
        paired[f"{a_} - {b_}"] = {"delta_in": float(ba[0] - bb[0]),
                                  "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    rep["paired"] = paired
    rep["sets"] = res_sets
    rep["selected"] = [{"fam": s.fam, "ch": s.ch, "kind": s.kind, "p": s.p} for s in selected]
    rep["selection_history"] = hist
    log("\n[theory] Предсказание до обучения (линейный R2 внутри юзера; oos = на val):")
    log(f"  {'набор':18s} {'#':>4s} {'R2_in':>8s} {'R2_oos':>8s} {'Δ vs manual [95% CI]':>28s}")
    for n, r in res_sets.items():
        dd = f"{r['delta_vs_manual_in']:+.4f} [{r['delta_ci'][0]:+.4f},{r['delta_ci'][1]:+.4f}]" if "delta_ci" in r else "—"
        log(f"  {n:18s} {r['n_kernel_features']:4d} {r['r2_in']:8.4f} {r['r2_oos']:8.4f} {dd:>28s}")
    log("\n[theory] парные разности (R2_in, 95% CI по бутстрэпу пользователей):")
    for k, v in paired.items():
        sig = "значимо" if (v["ci"][0] > 0 or v["ci"][1] < 0) else "не отличимо от 0"
        log(f"  {k:40s} {v['delta_in']:+.5f} [{v['ci'][0]:+.5f},{v['ci'][1]:+.5f}]  {sig}")
    json.dump(rep, open(f"{out_dir}/theory.json", "w"), indent=1)
    try:
        plot_theory(rep, out_dir, truth_hl)
    except Exception as e:  
        log(f"[theory] график пропущен: {e}")
    return (rep, S) if return_stats else rep


def load_selected(path: str) -> list[Spec]:
    """Признаки, отобранные теорией, для обучения бустинга (features.matrix(specs, ...))."""
    rep = json.load(open(path))
    return [mk(d["fam"], d["ch"], d["kind"], d["p"]) for d in rep["selected"]]


def plot_theory(rep, out_dir, truth_hl=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"listen": "tab:blue", "good": "tab:orange", "like": "tab:green"}
    hl, wg = np.array(rep["hl_grid"]), np.array(rep["win_grid"])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, fam in zip(axes.ravel(), FAMS):
        for ch in CHANNELS:
            c = rep["curves"][f"{fam}.{ch}"]
            ax.plot(hl, c["r2_exp_curve"], color=colors[ch], label=f"{ch}: экспонента (полупериод)")
            ax.plot(wg, c["r2_rect_curve"], color=colors[ch], ls="--", label=f"{ch}: окно")
            ax.axvline(c["hl_hat"], color=colors[ch], ls=":", alpha=.6)
        if truth_hl:
            ax.axvline(truth_hl, color="k", lw=1.5)
        ax.set_xscale("log"); ax.set_title(fam); ax.set_xlabel("дни"); ax.set_ylabel("R² (внутри юзера)")
    axes[0, 0].legend(fontsize=7)
    plt.suptitle("Объясняемая дисперсия одного признака-ядра; пунктир-точка - argmax; чёрная - истина")
    plt.tight_layout(); plt.savefig(f"{out_dir}/theory.png", dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "yambda", "generic"])
    ap.add_argument("--size", default="50m"); ap.add_argument("--path", default="")
    ap.add_argument("--users", type=int, default=3000); ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results_theory")
    ap.add_argument("--bins", type=int, default=20); ap.add_argument("--boot", type=int, default=16)
    ap.add_argument("--kmax", type=int, default=24)
    ap.add_argument("--n-train", type=int, default=6); ap.add_argument("--step", type=float, default=3.0)
    ap.add_argument("--horizon", type=float, default=1.0); ap.add_argument("--hist-days", type=float, default=30.0)
    ap.add_argument("--max-hist", type=float, default=120.0)
    ap.add_argument("--n-hist", type=int, default=150); ap.add_argument("--n-pop", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.dataset == "yambda":
        ev = load_yambda(a.size, a.users, a.seed)
    elif a.dataset == "generic":
        ev = load_generic(a.path)
    else:
        ev = make_synthetic(n_users=a.users, seed=a.seed)
    print(f"events={len(ev.t):,} users={ev.n_users:,} items={ev.n_items:,} span={ev.t.max():.1f} дней")
    snaps, tr, va, te = build_splits(ev, a.horizon, a.hist_days, a.n_train, a.step, a.n_hist, a.n_pop)  # te — список
    store = FeatureStore(ev, snaps, a.device, a.max_hist)
    t = analytic_tables(10.0)
    print("[аналитика] потеря корреляции при ошибке полупериода (истина 10 дней):",
          {k: round(v, 3) for k, v in t["exp_mismatch"].items()})
    print("[аналитика] окно w против экспоненты (истина 10 дней):",
          {k: round(v, 3) for k, v in t["rect_window"].items()})
    run_theory(store, tr, va, a.out, a.bins, a.boot, a.seed, a.kmax,
               truth_hl=10.0 if a.dataset == "synthetic" else None)


if __name__ == "__main__":
    main()