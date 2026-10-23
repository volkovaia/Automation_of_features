"""Насколько наборы признаков не дотягивают до ядра Винера (линейный потолок для сумм по ядрам).

python wiener_gap.py --dir results_multi

Берёт theory.sets из seed*/results.json (считается этапом theory/sets). Доля потолка = R2_in(набор) / R2_in(Винер) на TRAIN; интервал — t-95% по сидам. Потолок линейный и в выборке: вне выборки он не гарантирован (вал может быть ниже), хвост (last/age/конверсии) в него не входит.
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
from scipy import stats

SETS = ["manual_1_7_30", "exp_1_best", "theory_greedy", "rect_8scales", "exp_8scales"]


def t_ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), [float("nan")] * 2
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), [float(x.mean() - h), float(x.mean() + h)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="results_multi"); a = ap.parse_args()
    runs = [json.load(open(p)) for p in sorted(glob.glob(f"{a.dir}/seed*/results.json"))]
    runs = [r for r in runs if "theory" in r and "wiener_all_bins" in r["theory"]["sets"]]
    assert runs, "нет theory.sets.wiener_all_bins (нужен этап theory/sets)"
    out, md = {}, ["# Зазор до ядра Винера (линейный потолок, R² на train)\n",
                   f"Сидов: {len(runs)}. Потолок: все лаг-бины как признаки (R²_in, минимальная регуляризация).\n",
                   "| набор | число ядер | R² | доля потолка | 95% CI доли | зазор (R²) |", "|---|---|---|---|---|---|"]
    for n in SETS:
        if not all(n in r["theory"]["sets"] for r in runs):
            continue
        w = np.array([r["theory"]["sets"]["wiener_all_bins"]["r2_in"] for r in runs])
        v = np.array([r["theory"]["sets"][n]["r2_in"] for r in runs])
        k = int(np.mean([r["theory"]["sets"][n]["n_kernel_features"] for r in runs]))
        fr_mu, fr_ci = t_ci(v / w); gap_mu, _ = t_ci(w - v)
        out[n] = {"r2_in": float(v.mean()), "frac": fr_mu, "frac_ci": fr_ci, "gap": gap_mu, "per_seed_frac": (v / w).tolist()}
        md.append(f"| {n} | {k} | {v.mean():.4f} | {fr_mu:.3f} | [{fr_ci[0]:.3f}, {fr_ci[1]:.3f}] | {gap_mu:.4f} |")
    w = np.array([r["theory"]["sets"]["wiener_all_bins"]["r2_in"] for r in runs])
    md.append(f"\nПотолок (Винер): R² = {w.mean():.4f} (по сидам от {w.min():.4f} до {w.max():.4f}).")
    bad = [i for i, r in enumerate(runs) if any(r["theory"]["sets"][n]["r2_in"] > r["theory"]["sets"]["wiener_all_bins"]["r2_in"] + 1e-6
                                                   for n in SETS if n in r["theory"]["sets"])]
    if bad:
        md.append(f"\n**Внимание:** в {len(bad)} запусках какой-то набор выше потолка - численная неустойчивость.")
    open(f"{a.dir}/wiener_gap.md", "w").write("\n".join(md)); json.dump(out, open(f"{a.dir}/wiener_gap.json", "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()