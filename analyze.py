"""Анализ многосидового эксперимента: парные разности с доверительными интервалами и проверка "теория предсказывает метрики".

python analyze.py --dir results_multi

Критерий (задан заранее, до просмотра результатов 5 сидов * 3 дня):
  Для пары A−B и метрики m берётся разность по каждому сиду (среднее по пользователям и тестовым дням), затем t-интервал 95% по сидам (df = n_seeds−1, пользователи внутри сида - бутстрэп).
    "подтверждено"  - интервал целиком > 0 (A лучше B);
    "опровергнуто"  - интервал целиком < 0;
    "не определено" - интервал накрывает 0.
Основная метрика: auc_user (AUC внутри пользователя); вторичные: ndcg@10, recall@10; общий AUC и logloss только на уровне сидов.
Главное утверждение: theory_sel (K ядер от теории + общий хвост) лучше manual_1_7_30 (и manual_tuned) по auc_user и ndcg@10, и это не объясняется случайным выбором K ядер
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

PAIRS = [  
    ("theory_sel", "manual_1_7_30", "Главное: K ядер теории против классики"),
    ("theory_sel", "manual_expratios", "то же, но одинаковые конверсии/хвост"),
    ("theory_kernels", "manual_kernels", "чистый отбор: K ядер против 54, без хвоста"),
    ("theory_sel", "manual_tuned", "против сильного baseline (подобранные окна)"),
    ("theory_sel", "rand*", "против случайных K ядер (среднее по draw'ам)"),
    ("rand*", "manual_1_7_30", "случайные K ядер против классики"),
    ("exp_8scales", "manual_1_7_30", "8 экспонент против классики"),
    ("exp_8scales", "rect_8scales", "форма ядра: экспонента против окна"),
    ("exp_8scales", "theory_sel", "много признаков против K ядер"),
    # контроль хвоста (конверсии/last/age), добавлено после первого анализа
    ("manual_expratios", "manual_1_7_30", "эффект хвоста: экспон. конверсии вместо оконных"),
    ("manual_expratios", "manual_tuned", "хвост против подбора окон"),
    ("exp_8scales", "manual_expratios", "8 масштабов при одинаковом хвосте"),
    ("manual_tuned", "manual_1_7_30", "эффект подбора окон"),
    # план "форма ядра * форма конверсий" на 8 масштабах
    ("rect_8scales_expratios", "rect_8scales", "конверсии: exp вместо окон (ядра - окна)"),
    ("exp_8scales", "exp_8scales_rectratios", "конверсии: exp вместо окон (ядра - exp)"),
    ("exp_8scales", "rect_8scales_expratios", "чисто ядро: exp vs окно (конверсии exp у обоих)"),
    ("exp_8scales_rectratios", "rect_8scales", "чисто ядро: exp vs окно (конверсии окна у обоих)"),
    # подобранные окна + гладкие конверсии; контроль случайных ядер без хвоста
    ("manual_tuned_expratios", "manual_tuned", "подобранные окна: exp-конверсии вместо оконных"),
    ("manual_tuned_expratios", "manual_expratios", "подбор окон при одинаковом хвосте"),
    ("theory_kernels", "krand*", "K ядер теории против K случайных (без хвоста)"),
    ("krand*", "manual_kernels", "K случайных ядер против 54 ручных (без хвоста)"),
    # контроли теории: важен ли МАСШТАБ и чем жадный отбор лучше простого top-k
    ("theory_kernels", "rscale_k*", "масштаб: те же семейства/каналы, масштаб случаен (без хвоста)"),
    ("theory_sel", "rscale_t*", "масштаб: то же с хвостом"),
    ("rscale_k*", "krand*", "семейство/канал: выбранные теорией при случайном масштабе против случайных ядер"),
    ("theory_kernels", "topk_kernels", "жадный отбор против top-k по одиночному R² (без хвоста)"),
    ("theory_sel", "topk_sel", "жадный отбор против top-k (с хвостом)"),
    ("topk_kernels", "krand*", "top-k против случайных ядер (без хвоста)"),
    # порядок добавления: окна ∪ экспоненты (общий экспоненциальный хвост у всех трёх наборов)
    ("rect8_exp8_union", "rect_8scales_expratios", "экспоненты поверх окон (хвост общий)"),
    ("rect8_exp8_union", "exp_8scales", "ОКНА поверх экспонент (хвост общий)"),
    # непараметрическое ядро (гистограммы лагов) против параметрических
    ("lag_hist_tail", "exp_8scales", "непараметрическое ядро против 8 экспонент (хвост общий)"),
    ("lag_hist_tail", "rect_8scales_expratios", "непараметрическое ядро против 8 окон (хвост общий)"),
]
METRICS = ["auc_user", "ndcg@10", "recall@10"]
SEED_METRICS = ["auc", "auc_user", "ndcg@10", "recall@10", "logloss", "auc_novel", "ndcg@10_novel"]


def load(dir_):
    seeds = {}
    for d in sorted(glob.glob(f"{dir_}/seed*")):
        s = int(d.split("seed")[-1])
        if os.path.exists(f"{d}/results.json") and os.path.exists(f"{d}/peruser.pkl"):
            seeds[s] = (json.load(open(f"{d}/results.json")), pd.read_pickle(f"{d}/peruser.pkl"))
    return seeds


def tables(peruser, name):
    """Список таблиц по дням; 'rand*'- усреднение по случайным наборам."""
    if name.endswith("*"):
        names = [n for n in peruser if n.startswith(name[:-1])]
        if not names:
            return None
        return [sum(peruser[n][d] for n in names) / len(names) for d in range(len(peruser[names[0]]))]
    return peruser.get(name)


def diff_rows(peruser, a, b, metric):
    ta, tb = tables(peruser, a), tables(peruser, b)
    if ta is None or tb is None:
        return None
    parts = []
    for da, db in zip(ta, tb):
        d = (da[metric] - db[metric]).dropna()
        parts.append(d)
    s = pd.concat(parts)  # индекс = пользователь (повторяется по дням)
    return s


def boot_ci(s: pd.Series, n_boot=2000, seed=0):
    g = s.groupby(level=0).agg(["sum", "count"])
    sums, cnt = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    m = sums[idx].sum(1) / cnt[idx].sum(1)
    return float(sums.sum() / cnt.sum()), [float(x) for x in np.percentile(m, [2.5, 97.5])]


def t_ci(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return float(x.mean()), [float("nan")] * 2
    se = x.std(ddof=1) / np.sqrt(n); h = stats.t.ppf(0.975, n - 1) * se
    return float(x.mean()), [float(x.mean() - h), float(x.mean() + h)]


def verdict(ci):
    if np.isnan(ci[0]): return "n/a"
    return "подтверждено" if ci[0] > 0 else "опровергнуто" if ci[1] < 0 else "не определено"


def paired_report(seeds):
    out = []
    for a, b, note in PAIRS:
        for m in METRICS:
            per_seed = []
            for s, (res, pu) in seeds.items():
                d = diff_rows(pu, a, b, m)
                if d is None:
                    break
                mean, ci = boot_ci(d, seed=s)
                per_seed.append((s, mean, ci))
            else:
                means = [x[1] for x in per_seed]
                mu, ci = t_ci(means)
                out.append({"A": a, "B": b, "note": note, "metric": m, "mean": mu, "ci_t": ci,
                            "wins": f"{sum(x > 0 for x in means)}/{len(means)}", "verdict": verdict(ci),
                            "per_seed": [{"seed": s, "diff": mn, "ci_boot": c} for s, mn, c in per_seed]})
    return out


def seed_level_pooled(seeds):
    """Метрики на уровне сидов (общий AUC, logloss, novel): среднее и t-интервал по сидам."""
    names = sorted(set.intersection(*[set(k for k, v in r.items() if isinstance(v, dict) and "test" in v)
                                      for r, _ in seeds.values()]))
    rows = []
    for n in names:
        row = {"set": n, "n_features": next(iter(seeds.values()))[0][n].get("n_features")}
        for m in SEED_METRICS:
            vals = [r[n]["test"].get(m, np.nan) for r, _ in seeds.values()]
            if not np.all(np.isnan(vals)):
                mu, ci = t_ci([v for v in vals if not np.isnan(v)])
                row[m] = mu; row[m + "_pm"] = (ci[1] - ci[0]) / 2
        rows.append(row)
    return pd.DataFrame(rows)


def pooled_auc_pairs(seeds):
    out = []
    for a, b, _ in PAIRS:
        if a.endswith("*") or b.endswith("*"):
            continue
        for m in ("auc", "logloss"):
            d = [r[a]["test"][m] - r[b]["test"][m] for r, _ in seeds.values()
                 if "test" in r.get(a, {}) and "test" in r.get(b, {})]
            if len(d) == len(seeds):
                mu, ci = t_ci(d)
                out.append({"A": a, "B": b, "metric": m, "mean": mu, "ci_t": ci, "wins": f"{sum(x > 0 for x in d)}/{len(d)}"})
    return out


TAIL_RE = r"^(theory_k\d+|theory_sel|rand\d+_\d+|exp_8scales)$"
NOTAIL_RE = r"^(theory_kernels|krand\d+_\d+|rscale_k_\d+|topk_kernels)$"


def _group(n):
    for g, p in (("rand", r"^rand\d"), ("krand", r"^krand"), ("rscale_k", r"^rscale_k"), ("topk", r"^topk"),
                 ("theory_k", r"^theory_k\d|^theory_sel"), ("theory_kernels", r"^theory_kernels")):
        if re.match(p, n):
            return g
    return n


def pred_vs_actual(seeds, base="manual_expratios", metric="auc_user", pred="r2_oos", pattern=TAIL_RE):
    """Предсказанный прирост R2 против реального прироста метрики внутри семейства наборов с одинаковым хвостом (pattern=TAIL_RE, base=manual_expratios) 
    или без хвоста (pattern=NOTAIL_RE, base=manual_kernels)."""
    pts = []
    for s, (res, pu) in seeds.items():
        if base not in res["pred"] or base not in pu:
            continue
        for n, p in res["pred"].items():
            if re.match(pattern, n) is None or n == base or n not in pu:
                continue
            d = diff_rows(pu, n, base, metric)
            pts.append({"seed": s, "set": n, "group": _group(n), "dx": p[pred] - res["pred"][base][pred],
                        "dy": float(d.mean())})
    return pd.DataFrame(pts)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="results_multi")
    a = ap.parse_args()
    seeds = load(a.dir)
    assert seeds, f"нет seed*/results.json и peruser.pkl в {a.dir}"
    print(f"сидов: {len(seeds)} -> {sorted(seeds)}; тестовых дней на сид: "
          f"{len(next(iter(seeds.values()))[0]['args'] and next(iter(seeds.values()))[1]['manual_1_7_30'])}")
    pr = paired_report(seeds); pl = pooled_auc_pairs(seeds); lv = seed_level_pooled(seeds)

    md = ["# Результаты (парные разности A − B; >0 значит A лучше)\n"]
    md.append(f"Сидов: {len(seeds)}. Интервал — t-95% по сидам; в `per_seed` бутстрэп по пользователям.\n")
    for m in METRICS:
        md.append(f"\n## {m}\n\n| A | B | Δ (среднее) | 95% CI | побед по сидам | вердикт |\n|---|---|---|---|---|---|")
        for r in [x for x in pr if x["metric"] == m]:
            md.append(f"| {r['A']} | {r['B']} | {r['mean']:+.4f} | [{r['ci_t'][0]:+.4f}, {r['ci_t'][1]:+.4f}] | {r['wins']} | {r['verdict']} |")
    md.append("\n## Общий AUC и logloss (уровень сидов)\n\n| A | B | метрика | Δ | 95% CI | побед |\n|---|---|---|---|---|---|")
    for r in pl:
        md.append(f"| {r['A']} | {r['B']} | {r['metric']} | {r['mean']:+.4f} | [{r['ci_t'][0]:+.4f}, {r['ci_t'][1]:+.4f}] | {r['wins']} |")
    md.append("\n## Все наборы (среднее по сидам ± полуширина t-интервала)\n")
    cols = ["set", "n_features"] + [c for m in SEED_METRICS for c in (m,) if c in lv]
    md.append(lv[cols].round(4).to_markdown(index=False) if hasattr(lv, "to_markdown") else lv[cols].round(4).to_string(index=False))
    try:
        for metric in ("auc_user", "ndcg@10"):
            pv = pred_vs_actual(seeds, metric=metric)
            if len(pv) > 3:
                rho = stats.spearmanr(pv["dx"], pv["dy"]).statistic
                per = [stats.spearmanr(g["dx"], g["dy"]).statistic for _, g in pv.groupby("seed") if len(g) > 3]
                md.append(f"\n**Предсказание против факта ({metric}, n={len(pv)} наборов×сидов):** "
                          f"Spearman по всем = {rho:+.2f}; по сидам = {np.nanmean(per):+.2f} "
                          f"[{np.nanmin(per):+.2f}, {np.nanmax(per):+.2f}]\n")
        for metric in ("auc_user", "ndcg@10"):
            pv = pred_vs_actual(seeds, base="manual_kernels", metric=metric, pattern=NOTAIL_RE)
            if len(pv) > 3:
                rho = stats.spearmanr(pv["dx"], pv["dy"]).statistic
                per = [stats.spearmanr(g["dx"], g["dy"]).statistic for _, g in pv.groupby("seed") if len(g) > 3]
                inner = pv[pv["group"] == "krand"]
                rk = stats.spearmanr(inner["dx"], inner["dy"]).statistic if len(inner) > 3 else float("nan")
                md.append(f"\n**Без хвоста: предсказание против факта ({metric}, n={len(pv)}):** Spearman по всем = "
                          f"{rho:+.2f}; по сидам = {np.nanmean(per):+.2f} [{np.nanmin(per):+.2f}, {np.nanmax(per):+.2f}]; "
                          f"только среди krand* = {rk:+.2f}\n")
    except Exception as e:  # noqa: BLE001
        md.append(f"\n(предсказание против факта: пропущено: {e})")
    open(f"{a.dir}/analysis.md", "w").write("\n".join(md))
    json.dump({"paired": pr, "pooled": pl}, open(f"{a.dir}/analysis.json", "w"), indent=1, default=str)
    print("\n".join(md)); print(f"\nсохранено: {a.dir}/analysis.md, analysis.json")
    try:
        plots(a.dir, seeds, pr)
    except Exception as e:  
        print(f"[plots] пропущено: {e}")


def plots(dir_, seeds, pr):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(METRICS), figsize=(6 * len(METRICS), 5), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), METRICS):
        rs = [r for r in pr if r["metric"] == m]
        for yy, r in enumerate(rs):
            for ps in r["per_seed"]:
                ax.plot(ps["diff"], yy, "o", color="lightgray", ms=5, zorder=1)
            ax.errorbar(r["mean"], yy, xerr=[[r["mean"] - r["ci_t"][0]], [r["ci_t"][1] - r["mean"]]],
                        fmt="s", color="tab:red" if r["verdict"] == "опровергнуто" else "tab:green" if r["verdict"] == "подтверждено" else "tab:blue",
                        capsize=4, zorder=2)
        ax.axvline(0, color="k", lw=1); ax.set_title(f"Δ {m}  (серые — сиды)")
        ax.set_yticks(range(len(rs))); ax.set_yticklabels([f"{r['A']} − {r['B']}" for r in rs], fontsize=8)
        ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{dir_}/forest.png", dpi=130); plt.close()

    colors = {"theory_k": "tab:orange", "rand": "tab:gray", "exp_8scales": "tab:blue", "theory_sel": "tab:red"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metric in zip(axes, ["auc_user", "ndcg@10"]):
        pv = pred_vs_actual(seeds, metric=metric)
        for g, d in pv.groupby("group"):
            ax.scatter(d["dx"], d["dy"], s=22, color=colors.get(g, "tab:purple"), label=g, alpha=.8)
        ax.axhline(0, color="k", lw=.8); ax.axvline(0, color="k", lw=.8)
        ax.set_xlabel("предсказанный прирост R² (val) против manual_expratios")
        ax.set_ylabel(f"реальный прирост {metric}"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{dir_}/pred_vs_actual_all.png", dpi=130); plt.close()

    colors2 = {"theory_kernels": "tab:red", "krand": "tab:gray", "rscale_k": "tab:purple", "topk": "tab:green"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metric in zip(axes, ["auc_user", "ndcg@10"]):
        pv = pred_vs_actual(seeds, base="manual_kernels", metric=metric, pattern=NOTAIL_RE)
        for g, d in pv.groupby("group"):
            ax.scatter(d["dx"], d["dy"], s=24, color=colors2.get(g, "tab:blue"), label=g, alpha=.8)
        ax.axhline(0, color="k", lw=.8); ax.axvline(0, color="k", lw=.8)
        ax.set_xlabel("предсказанный прирост R² (val) против manual_kernels"); ax.set_ylabel(f"реальный прирост {metric}")
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{dir_}/pred_vs_actual_notail.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()