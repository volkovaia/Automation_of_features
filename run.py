"""Многосидовый эксперимент: теория (предсказание) -> бустинг (проверка) на нескольких выборках пользователей и нескольких тестовых днях. Анализ в analyze.py.

Смоук-тест:  python run.py --dataset synthetic --device cpu --users 250 --seeds 0,1 --n-test 3 \
                 --theory-boot 4 --theory-bins 16 --n-rand 2 --stages heur,theory,sets
Основной:    python run.py --dataset yambda --size 50m --users 3000 --device cuda \
                 --seeds 0,1,2,3,4 --n-test 3 --out results_multi

Этапы (--stages): heur, theory, sets, tuned, extra, extra2, scales, extra3, learned
  sets  - все сравниваемые наборы признаков (ручные, 8 масштабов, выбранные теорией, вложенные k=1..K,
          случайные K ядер, контроли по хвосту признаков);
  tuned - "сильный ручной baseline": окна подобраны покоординатным спуском по val;
  extra2 - контроли теории (--append): rscale_k_*/rscale_t_* (случайный масштаб при тех же
          семействах/каналах), topk_kernels/topk_sel (top-k по одиночному R2), предсказания для krand*;
  scales - проверка двух шкал (1 экспонента vs смесь 2) + страты активности (без бустинга; --append не нужен);
  extra3 - rect8_exp8_union (окна ∪ экспоненты) и lag_hist/lag_hist_tail (непараметрическое ядро; --append);
  extra - доп. контроли (нужен tuned в results.json; запускать с --append):
          rect_8scales_expratios, exp_8scales_rectratios, manual_tuned_expratios, krand6_* (без хвоста).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import numpy as np
import pandas as pd

import models
from data import CHANNELS, build_splits, full_metrics, load_generic, load_yambda, make_synthetic
import features
from features import (FAMS, FeatureStore, _ratios, exp_set, extras, lag_hist_set, learned_set, manual_set,
                      mk, rect8_set)
from models import run_set
from theory import (compute_stats, predict_set, run_two_scale, random_specs, resample_scales, run_theory,
                    topk_single)


def kern(specs):
    return [s for s in specs if s.kind in ("rect", "exp")]


def run_seed(a, seed: int):
    out = f"{a.out}/seed{seed}"; os.makedirs(out, exist_ok=True)
    stages = a.stages.split(","); log = print
    res: dict = {"args": vars(a), "seed": seed, "pred": {}}
    peruser: dict = {}
    if a.append and os.path.exists(f"{out}/results.json") and os.path.exists(f"{out}/peruser.pkl"):
        res = json.load(open(f"{out}/results.json")); peruser = pd.read_pickle(f"{out}/peruser.pkl")
        res.setdefault("pred", {}); res["args_extra"] = vars(a)
        log(f"[append] загружено {len([k for k in res if k in peruser])} готовых наборов для seed={seed}")

    if a.dataset == "yambda":
        ev = load_yambda(a.size, a.users, seed)
    elif a.dataset == "generic":
        ev = load_generic(a.path)
    else:
        ev = make_synthetic(n_users=a.users, seed=seed)
    log(f"\n#### seed={seed}: events={len(ev.t):,} users={ev.n_users:,} items={ev.n_items:,} span={ev.t.max():.1f}д")
    snaps, tr, va, te = build_splits(ev, a.horizon, a.hist_days, a.n_train, a.step, a.n_hist, a.n_pop, a.n_test)
    store = FeatureStore(ev, snaps, a.device, a.max_hist)
    S = None

    def save():
        json.dump(res, open(f"{out}/results.json", "w"), indent=1, default=str)
        pd.to_pickle(peruser, f"{out}/peruser.pkl")

    def fit(name, specs, **kw):
        t0 = time.time()
        r = run_set(store, specs, tr, va, te, **kw)
        peruser[name] = r.pop("_per_user")
        r["seconds"] = round(time.time() - t0, 1); res[name] = r
        if S is not None and kern(specs):
            res["pred"][name] = predict_set(S, kern(specs))
        t_ = r["test"]
        log(f"==> {name:18s} feats={r['n_features']:4d} val_auc={r['val']['auc']:.4f} test auc={t_['auc']:.4f} "
            f"auc_user={t_['auc_user']:.4f} ndcg@10={t_['ndcg@10']:.4f}"
            + (f" | pred R2_oos={res['pred'][name]['r2_oos']:.4f}" if name in res["pred"] else ""))
        save()

    # эвристики без обучения
    if "heur" in stages:
        for name, sp in [("heur_item_pop7", mk("item", "good", "rect", 7)),
                         ("heur_pair_exp7", mk("pair", "good", "exp", 7))]:
            store.ensure([sp]); days, tabs = [], []
            for ti in te:
                m, tab = full_metrics(snaps[ti], store.cols[ti][sp], prob=False, return_table=True)
                days.append(m); tabs.append(tab)
            res[name] = {"test": {k: float(np.nanmean([d.get(k, np.nan) for d in days])) for k in days[0]},
                         "n_features": 1}
            peruser[name] = tabs
            log(f"==> {name:18s} auc_user={res[name]['test']['auc_user']:.4f} ndcg@10={res[name]['test']['ndcg@10']:.4f}")
        save()

    # Теория до обучения
    sel, theory = [], None
    if "theory" in stages or "sets" in stages:
        t0 = time.time()
        theory, S = run_theory(store, tr, va, f"{out}/theory", a.theory_bins, a.theory_boot, seed,
                               a.theory_k, truth_hl=10.0 if a.dataset == "synthetic" else None, log=log,
                               min_gain=0.0, return_stats=True)
        sel = [mk(d["fam"], d["ch"], d["kind"], d["p"]) for d in theory["selected"]]
        res["theory"] = {"sets": theory["sets"], "paired": theory["paired"], "selected": theory["selected"],
                         "selection_history": theory["selection_history"], "seconds": round(time.time() - t0, 1)}
        save()

    # сравниваемые наборы
    if "sets" in stages:
        tail = _ratios("exp") + extras()                    # общий хвост: конверсии + last/age
        man = manual_set()
        sets = {"manual_1_7_30": man,                        # классика (конверсии по окнам)
                "manual_expratios": kern(man) + tail,        # те же окна, но хвост как у остальных
                "manual_kernels": kern(man),                 # 54 ядра, без хвоста
                "rect_8scales": rect8_set(), "exp_8scales": exp_set(),
                "theory_sel": sel + tail,                    # K ядер от теории + общий хвост
                "theory_kernels": sel}                       # те же K ядер, без хвоста
        for k in range(1, len(sel)):                         # вложенные наборы
            sets[f"theory_k{k}"] = sel[:k] + tail
        for r in range(a.n_rand):                            # случайные K ядер из того же пула
            sets[f"rand{len(sel)}_{r}"] = random_specs(np.random.default_rng(seed * 1000 + r), len(sel)) + tail
        for name, specs in sets.items():
            fit(name, specs)

    # сильный ручной baseline: подбор окон
    if "tuned" in stages:
        t0 = time.time(); grid = (2, 3, 5, 7, 10, 14, 21, 45, 60)
        wins = {f: (1, 7, 30) for f in FAMS}
        best = run_set(store, manual_set(wins), tr, va, None, quick=True)["val"]["auc"]
        for fam in FAMS:
            for w in grid:
                if w == wins[fam][1]:
                    continue
                trial = dict(wins); trial[fam] = (1, w, 30)
                s_ = run_set(store, manual_set(trial), tr, va, None, quick=True)["val"]["auc"]
                if s_ > best + 1e-5:
                    best, wins = s_, trial
            log(f"[tuned] {fam}: mid-window={wins[fam][1]} val_auc(quick)={best:.5f}")
        fit("manual_tuned", manual_set(wins))
        res["manual_tuned"]["windows"] = {k: list(v) for k, v in wins.items()}
        res["manual_tuned"]["tuning_seconds"] = round(time.time() - t0, 1); save()

    #  доп. контроли: форма ядра * форма конверсий, хвост у подобранных окон, случайные ядра без хвоста
    if "extra" in stages:
        tail_e = _ratios("exp") + extras()
        ex = {"rect_8scales_expratios": kern(rect8_set()) + tail_e,        # окна + эксп. конверсии
              "exp_8scales_rectratios": kern(exp_set()) + _ratios("rect") + extras()}   # экспоненты + оконные конверсии
        wins = res.get("manual_tuned", {}).get("windows")
        if wins:
            ex["manual_tuned_expratios"] = kern(manual_set({k: tuple(v) for k, v in wins.items()})) + tail_e
        else:
            log("[extra] нет manual_tuned.windows (сначала этап tuned) - manual_tuned_expratios пропущен")
        for r in range(a.n_rand):                                            # K случайных ядер БЕЗ хвоста
            ex[f"krand{a.theory_k}_{r}"] = random_specs(np.random.default_rng(seed * 1000 + 500 + r), a.theory_k)
        for name, specs in ex.items():
            fit(name, specs)

    # контроли теории: случайный масштаб, простой top-k, предсказания для наборов без хвоста
    if "extra2" in stages:
        assert "theory" in res and res["theory"].get("selected"), "сначала этап sets/theory (--append)"
        S = compute_stats(store, tr, va, a.theory_bins, 0, seed, check=False, log=log)   # без бутстрэпа
        sel = [mk(d["fam"], d["ch"], d["kind"], d["p"]) for d in res["theory"]["selected"]]
        tail = _ratios("exp") + extras()
        res["pred"]["manual_kernels"] = predict_set(S, kern(manual_set()))
        res["pred"]["theory_kernels"] = predict_set(S, sel)
        for r_ in range(a.n_rand):          # предсказания для уже обученных krand* (те же случайные ядра)
            nm = f"krand{a.theory_k}_{r_}"
            if nm in res:
                res["pred"][nm] = predict_set(S, random_specs(np.random.default_rng(seed * 1000 + 500 + r_), a.theory_k))
        ex2 = {}
        for r_ in range(a.n_rand):          # те же (семейство, канал, тип), случайный масштаб
            rs = resample_scales(np.random.default_rng(seed * 1000 + 700 + r_), sel)
            ex2[f"rscale_k_{r_}"] = rs; ex2[f"rscale_t_{r_}"] = rs + tail
        tk = topk_single(S, a.theory_k)     # top-k по одиночному R2, без жадности
        ex2["topk_kernels"] = tk; ex2["topk_sel"] = tk + tail
        res["topk_selected"] = [s.name for s in tk]; save()
        for name, specs in ex2.items():
            fit(name, specs)

    # проверка "двух шкал" + оценки по стратам активности (без обучения бустинга)
    if "scales" in stages:
        S2 = compute_stats(store, tr, va, a.theory_bins, a.theory_boot, seed, check=False, log=log,
                           n_strata=a.n_strata)
        res["two_scale"] = run_two_scale(S2, f"{out}/scales", log=log); save()
        del S2

    # вложенность "окна ∪ экспоненты" и непараметрическое ядро (гистограммы лагов)
    if "extra3" in stages:
        features.LAG_BINS = a.lag_bins
        tail_e = _ratios("exp") + extras()
        ex3 = {"rect8_exp8_union": kern(rect8_set()) + kern(exp_set()) + tail_e,     # окна ∪ экспоненты
               "lag_hist": lag_hist_set(),                                           # без хвоста
               "lag_hist_tail": lag_hist_set() + tail_e}
        for name, specs in ex3.items():
            fit(name, specs); store.clear_cols()      # экономим RAM: наборы большие

    # бучаемые ядра (опционально; нейросеть оценивается на первом тестовом дне)
    if "learned" in stages:
        from learn_kernel import predict_snapshot, train_kernel_net
        net = train_kernel_net(store, tr, va, FAMS, K=a.kernel_k, kernel_users=a.kernel_users,
                               epochs=a.kernel_epochs, seed=seed, log=log)
        res["neural_only"] = {"test": full_metrics(snaps[te[0]], predict_snapshot(net, store, te[0])),
                              "note": "только первый тестовый день"}
        hl = net.half_lives(); res["learned_half_lives"] = {f"{k[0]}.{k[1]}": v for k, v in hl.items()}
        fit("learned_exp", learned_set(hl))

    save()
    del store; gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "yambda", "generic"])
    ap.add_argument("--size", default="50m"); ap.add_argument("--path", default="")
    ap.add_argument("--users", type=int, default=3000)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="results_multi")
    ap.add_argument("--seeds", default="0,1,2,3,4"); ap.add_argument("--n-test", type=int, default=3)
    ap.add_argument("--stages", default="heur,theory,sets,tuned",
                    help="heur,theory,sets,tuned,extra,extra2,scales,extra3,learned")
    ap.add_argument("--append", action="store_true", help="дописать к уже посчитанным seed*/results.json")
    ap.add_argument("--n-train", type=int, default=6); ap.add_argument("--step", type=float, default=3.0)
    ap.add_argument("--horizon", type=float, default=1.0); ap.add_argument("--hist-days", type=float, default=30.0)
    ap.add_argument("--max-hist", type=float, default=120.0)
    ap.add_argument("--n-hist", type=int, default=150); ap.add_argument("--n-pop", type=int, default=50)
    ap.add_argument("--theory-bins", type=int, default=20); ap.add_argument("--theory-boot", type=int, default=16)
    ap.add_argument("--theory-k", type=int, default=6, help="сколько ядер отбирает теория (и случайный контроль)")
    ap.add_argument("--n-rand", type=int, default=5, help="сколько случайных наборов из K ядер на сид")
    ap.add_argument("--lag-bins", type=int, default=20, help="лаг-бины для lag_hist (extra3)")
    ap.add_argument("--n-strata", type=int, default=3, help="страты активности для этапа scales")
    ap.add_argument("--kernel-users", type=int, default=1500); ap.add_argument("--kernel-k", type=int, default=8)
    ap.add_argument("--kernel-epochs", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    models.set_device(a.device)
    for seed in [int(x) for x in a.seeds.split(",")]:
        run_seed(a, seed)
    print(f"\nготово: {a.out}/seed*/results.json  ->  python analyze.py --dir {a.out}")


if __name__ == "__main__":
    main()