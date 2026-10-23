"""Сводка проверки двух временных шкал по сидам.

python scales_report.py --dir results_multi        (нужен этап scales в seed*/results.json)

Для каждого (семейство, канал):
  * gain_oos - прирост R² на val от смеси двух экспонент над лучшей одной; контроль gain_oos_random2 - то же при второй шкале наугад. Разность d = gain_oos − gain_oos_random2 и её t-95% по сидам: d>0   значимо -> вторая шкала
    несёт сигнал сверх простого добавления признака.
  * h_hat по стратам активности: если оптимум одной экспоненты заметно сдвигается между стратами - это неоднородность пользователей, а не обязательно две шкалы внутри пользователя.
Критерий две шкалы поддержаны: d>0 с интервалом целиком выше 0 И оптимум не сдвигается между стратами.
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
from scipy import stats


def t_ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), [float("nan")] * 2
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), [float(x.mean() - h), float(x.mean() + h)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="results_multi"); a = ap.parse_args()
    runs = [json.load(open(p)) for p in sorted(glob.glob(f"{a.dir}/seed*/results.json"))]
    runs = [r for r in runs if "two_scale" in r]
    assert runs, "нет two_scale (нужно запустить --stages scales)"
    keys = list(runs[0]["two_scale"])
    md = [f"# Две временные шкалы: одна экспонента против смеси двух (сидов: {len(runs)})\n",
          "| (семейство.канал) | h одной (медиана) | пара h (медиана) | gain вне выборки | наугад | d = gain − наугад [95% CI] | побед | закрыто зазора до Винера | h по стратам (мед.) | сдвиг по стратам | вердикт |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for k in keys:
        rs = [r["two_scale"][k] for r in runs]
        g = np.array([x["gain_oos"] for x in rs]); c = np.array([x["gain_oos_random2"] for x in rs]); d = g - c
        mu, ci = t_ci(d)
        h1 = np.median([x["h_single"] for x in rs])
        ha = np.median([x["h_pair"][0] for x in rs]); hb = np.median([x["h_pair"][1] for x in rs])
        gc = [x["gap_closed"] for x in rs if x["gap_closed"] is not None]
        st = [x["strata"] for x in rs if x["strata"]]
        if st:
            hs = np.nanmedian([[s["h_hat"] for s in sx] for sx in st], axis=0)
            mono = sum(all(sx[i]["h_hat"] < sx[i + 1]["h_hat"] for i in range(len(sx) - 1)) for sx in st)
            ratio = float(hs.max() / hs.min())
            shift = f"{ratio:.1f}× (монотонно в {mono}/{len(st)} сидах)"
            hs_txt = " / ".join(f"{h:.2f}" for h in hs); shifted = ratio > 2.0
        else:
            hs_txt, shift, shifted = "—", "—", False
        sup = (ci[0] > 0)
        verdict = ("две шкалы поддержаны" if sup and not shifted else
                   "вторая шкала помогает, но оптимум сдвигается по стратам (неоднородность?)" if sup else "не поддержано")
        summary[k] = {"d_mean": mu, "d_ci": ci, "verdict": verdict}
        md.append(f"| {k} | {h1:.2f} | {ha:.2f} / {hb:.1f} | {g.mean():+.5f} | {c.mean():+.5f} | {mu:+.5f} [{ci[0]:+.5f}, {ci[1]:+.5f}] | "
                  f"{int((d > 0).sum())}/{len(d)} | {np.mean(gc) if gc else float('nan'):.2f} | {hs_txt} | {shift} | {verdict} |")
    open(f"{a.dir}/scales_report.md", "w").write("\n".join(md)); json.dump(summary, open(f"{a.dir}/scales_report.json", "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()