"""Генерация признаков как набор "спецификаций" (DSL) + GPU-вычисление.

Признак = Spec(fam, ch, kind, p, den):
  fam  - семейство ключа агрегации:
         pair (user,item) | uart (user,artist) | ualb (user,album) | item | art | user
  ch   - канал события: listen | good | like
  kind - bin:   число событий в лаг-бине номер p (непараметрическое ядро; границы - lag_edges)
         rect:  sum_k 1[T - t_k <= p]              (прямоугольное окно p дней)
         exp :  sum_k 2^{-(T - t_k)/p}              (экспоненц. ядро, полупериод p дней)
         last:  время с последнего события (дни)
         age :  время с самого первого события (дни)
         ratio_rect / ratio_exp: сглаженная конверсия ch/den в окне/ядре p
Любая Spec по построению использует только события с t < T  -> утечки нет.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch

from data import CHANNELS, Events, Snapshot

LN2 = math.log(2.0)
LAG_BINS = 20          # число лаг-бинов для признаков kind="bin" (run.py: --lag-bins)
FAMS = ["pair", "uart", "ualb", "item", "art", "user"]
KINDS = ["rect", "exp", "last", "age", "ratio_rect", "ratio_exp"]
CH_IDX = {c: k for k, c in enumerate(CHANNELS)}


class Spec(NamedTuple):
    fam: str
    ch: str
    kind: str
    p: float = 0.0
    den: str = ""

    @property
    def name(self) -> str:
        d = f"/{self.den}" if self.den else ""
        return f"{self.fam}.{self.ch}{d}.{self.kind}{self.p:g}"


def mk(fam, ch, kind, p=0.0, den="") -> Spec:
    return Spec(fam, ch, kind, round(float(p), 4), den)


def validate(d: dict) -> Spec | None:
    """Проверка спецификации от LLM/рандома. None - если некорректна."""
    try:
        fam, ch, kind = d["fam"], d["ch"], d["kind"]
        p = float(d.get("p", 0.0)); den = d.get("den", "") or ""
        if fam not in FAMS or ch not in CHANNELS or kind not in KINDS:
            return None
        if kind in ("rect", "exp", "ratio_rect", "ratio_exp") and not (0.01 <= p <= 365):
            return None
        if kind.startswith("ratio") and (den not in CHANNELS or den == ch):
            return None
        if not kind.startswith("ratio"):
            den = ""
        if kind in ("last", "age"):
            p = 0.0
        return mk(fam, ch, kind, p, den)
    except Exception:  # noqa: BLE001
        return None


# наборы признаков
HL_GRID = (0.05, 0.25, 1.0, 3.0, 7.0, 14.0, 30.0, 60.0)       # полупериоды, дни
WIN_GRID = (0.25, 1.0, 3.0, 7.0, 14.0, 30.0, 60.0, 90.0)      # окна, дни


def extras() -> list[Spec]:
    return ([mk("pair", c, "last") for c in CHANNELS]
            + [mk("uart", "listen", "last"), mk("uart", "good", "last"),
               mk("item", "listen", "age"), mk("art", "listen", "age")])


def manual_set(windows: dict | None = None) -> list[Spec]:
    """Классика: окна 1/7/30 для всех семейств, плюс конверсии. windows[fam]=(short,mid,long)."""
    out = []
    for fam in FAMS:
        w = (windows or {}).get(fam, (1, 7, 30))
        for ch in CHANNELS:
            out += [mk(fam, ch, "rect", x) for x in w]
    for fam in ("pair", "uart", "item"):
        w = (windows or {}).get(fam, (1, 7, 30))
        out += [mk(fam, "good", "ratio_rect", w[1], "listen"), mk(fam, "good", "ratio_rect", w[2], "listen"),
                mk(fam, "like", "ratio_rect", w[2], "listen")]
    return out + extras()


def rect8_set() -> list[Spec]:
    out = [mk(f, c, "rect", w) for f in FAMS for c in CHANNELS for w in WIN_GRID]
    return out + _ratios("rect") + extras()


def exp_set(hls=HL_GRID) -> list[Spec]:
    out = [mk(f, c, "exp", h) for f in FAMS for c in CHANNELS for h in hls]
    return out + _ratios("exp") + extras()


def _ratios(base: str) -> list[Spec]:
    out = []
    for fam in ("pair", "uart", "item"):
        for p in (7, 30):
            out += [mk(fam, "good", f"ratio_{base}", p, "listen")]
        out += [mk(fam, "like", f"ratio_{base}", 30, "listen")]
    return out


def learned_set(rates: dict) -> list[Spec]:
    """rates[(fam,ch)] = список полупериодов (дни), найденных градиентом."""
    out = [mk(f, c, "exp", h) for (f, c), hs in rates.items() for h in hs]
    return out + _ratios("exp") + extras()


def lag_edges(B: int, max_hist: float, lo: float = 0.02) -> np.ndarray:
    """B+1 границ лаг-бинов: [0, lo), далее геометрическая сетка до max_hist (как в theory.make_edges)."""
    return np.concatenate([[0.0], np.geomspace(lo, max_hist, B)])


def lag_hist_set() -> list[Spec]:
    """Непараметрическое ядро: счётчики по лаг-бинам для всех семейств и каналов (бустинг сам учит ядро)."""
    return [mk(f, c, "bin", b) for f in FAMS for c in CHANNELS for b in range(LAG_BINS)]


# GPU-агрегация
class Matched:
    """События, попавшие в ключи кандидатов данного снапшота (для одного семейства)."""
    __slots__ = ("q", "dt", "val", "cand_q", "nq")

    def __init__(self, q, dt, val, cand_q, nq):
        self.q, self.dt, self.val, self.cand_q, self.nq = q, dt, val, cand_q, nq


class FeatureStore:
    def __init__(self, ev: Events, snaps: list[Snapshot], device: str = "cuda",
                 max_hist: float = 120.0, chunk: int = 2_000_000):
        self.ev, self.snaps, self.dev = ev, snaps, torch.device(device)
        self.max_hist, self.chunk = max_hist, chunk
        self.t_np = ev.t
        d = self.dev
        self.et = torch.from_numpy(ev.t).to(d)
        self.ev_val = torch.from_numpy(ev.val).to(d)
        u, i = ev.u.astype(np.int64), ev.i.astype(np.int64)
        keys = {"pair": u * ev.n_items + i, "uart": u * ev.n_art + ev.art[i],
                "ualb": u * ev.n_alb + ev.alb[i], "item": i, "art": ev.art[i].astype(np.int64),
                "user": u}
        self.ek = {k: torch.from_numpy(np.ascontiguousarray(v)).to(d) for k, v in keys.items()}
        self.cols: list[dict] = [dict() for _ in snaps]

    # ключи кандидатов
    def cand_keys(self, fam: str, cu: np.ndarray, ci: np.ndarray) -> np.ndarray:
        ev = self.ev; u, i = cu.astype(np.int64), ci.astype(np.int64)
        return {"pair": lambda: u * ev.n_items + i, "uart": lambda: u * ev.n_art + ev.art[i],
                "ualb": lambda: u * ev.n_alb + ev.alb[i], "item": lambda: i,
                "art": lambda: ev.art[i].astype(np.int64), "user": lambda: u}[fam]()

    def build_matched(self, si: int, fam: str, cand_idx: np.ndarray | None = None) -> Matched:
        s = self.snaps[si]
        cu, ci = (s.cu, s.ci) if cand_idx is None else (s.cu[cand_idx], s.ci[cand_idx])
        ck = torch.from_numpy(np.ascontiguousarray(self.cand_keys(fam, cu, ci))).to(self.dev)
        uq, cand_q = torch.unique(ck, return_inverse=True)
        lo = int(np.searchsorted(self.t_np, s.T - self.max_hist)); hi = int(np.searchsorted(self.t_np, s.T))
        ek = self.ek[fam][lo:hi]
        pos = torch.searchsorted(uq, ek).clamp_(max=len(uq) - 1)
        hit = uq[pos] == ek
        q = pos[hit]
        dt = (s.T - self.et[lo:hi][hit]).float()
        val = self.ev_val[lo:hi][hit]
        return Matched(q, dt, val, cand_q, len(uq))

    def kernel_sums(self, m: Matched, kind: str, params) -> torch.Tensor:
        """[nq, C, P] суммы по ядру для списка параметров."""
        C, P = len(CHANNELS), len(params)
        p = torch.tensor(list(params), device=self.dev, dtype=torch.float32)
        out = torch.zeros(m.nq, C * P, device=self.dev)
        for a in range(0, m.q.shape[0], self.chunk):
            dt = m.dt[a:a + self.chunk, None]
            w = torch.exp(-dt * (LN2 / p)[None]) if kind == "exp" else (dt <= p[None]).float()
            v = m.val[a:a + self.chunk].float()
            out.index_add_(0, m.q[a:a + self.chunk], (v[:, :, None] * w[:, None, :]).reshape(-1, C * P))
        return out.view(m.nq, C, P)

    def lag_hist(self, m: Matched) -> torch.Tensor:
        """[nq, C, B] - число событий каждого канала в лаг-бинах."""
        C, B = len(CHANNELS), LAG_BINS
        edges = torch.tensor(lag_edges(B, self.max_hist)[1:-1], dtype=torch.float32, device=self.dev)
        bins = torch.bucketize(m.dt, edges)
        flat = torch.zeros(m.nq * C * B, device=self.dev)
        for c in range(C):
            mk_ = m.val[:, c] > 0
            idx = m.q[mk_] * (C * B) + c * B + bins[mk_]
            flat.index_add_(0, idx, torch.ones(idx.shape[0], device=self.dev))
        return flat.view(m.nq, C, B)

    def clear_cols(self):
        """Освободить кэш признаков (RAM) между крупными наборами."""
        self.cols = [dict() for _ in self.snaps]

    def last_age(self, m: Matched):
        C = len(CHANNELS)
        last = torch.full((m.nq, C), float("inf"), device=self.dev)
        for c in range(C):
            mk_ = m.val[:, c] > 0
            if mk_.any():
                last[:, c].scatter_reduce_(0, m.q[mk_], m.dt[mk_], "amin")
        age = torch.full((m.nq,), -1.0, device=self.dev)
        mk_ = m.val[:, 0] > 0
        if mk_.any():
            age.scatter_reduce_(0, m.q[mk_], m.dt[mk_], "amax")
        last[torch.isinf(last)] = float("nan"); age[age < 0] = float("nan")
        return last, age

    # публичное API
    def ensure(self, specs: list[Spec]):
        specs = list(dict.fromkeys(specs))
        base = []
        for s in specs:                       # ratio -> зависимости
            if s.kind.startswith("ratio"):
                b = s.kind.split("_")[1]
                base += [Spec(s.fam, s.ch, b, s.p), Spec(s.fam, s.den, b, s.p)]
            else:
                base.append(s)
        base = list(dict.fromkeys(base))
        for si in range(len(self.snaps)):
            need = [s for s in base if s not in self.cols[si]]
            for fam in FAMS:
                fs = [s for s in need if s.fam == fam]
                if not fs:
                    continue
                m = self.build_matched(si, fam)
                for kind in ("rect", "exp"):
                    ks = [s for s in fs if s.kind == kind]
                    if ks:
                        ps = sorted({s.p for s in ks})
                        S = self.kernel_sums(m, kind, ps)[m.cand_q].cpu().numpy()
                        for s in ks:
                            self.cols[si][s] = S[:, CH_IDX[s.ch], ps.index(s.p)].astype(np.float32)
                bs = [s for s in fs if s.kind == "bin"]
                if bs:
                    Hc = self.lag_hist(m)[m.cand_q].cpu().numpy()
                    for s in bs:
                        self.cols[si][s] = Hc[:, CH_IDX[s.ch], int(s.p)].astype(np.float32)
                    del Hc
                if any(s.kind in ("last", "age") for s in fs):
                    last, age = self.last_age(m)
                    last = last[m.cand_q].cpu().numpy(); age = age[m.cand_q].cpu().numpy()
                    for s in fs:
                        if s.kind == "last":
                            self.cols[si][s] = last[:, CH_IDX[s.ch]].astype(np.float32)
                        elif s.kind == "age":
                            self.cols[si][s] = age.astype(np.float32)
                del m
            for s in specs:
                if s.kind.startswith("ratio") and s not in self.cols[si]:
                    b = s.kind.split("_")[1]
                    num = self.cols[si][Spec(s.fam, s.ch, b, s.p)].astype(np.float64)
                    den = self.cols[si][Spec(s.fam, s.den, b, s.p)].astype(np.float64)
                    prior = num.sum() / max(den.sum(), 1.0); a = 3.0
                    self.cols[si][s] = ((num + a * prior) / (den + a)).astype(np.float32)

    def matrix(self, specs: list[Spec], snap_ids: list[int]) -> np.ndarray:
        self.ensure(specs)
        return np.concatenate([np.stack([self.cols[si][s] for s in specs], 1) for si in snap_ids], 0)

    def labels(self, snap_ids): return np.concatenate([self.snaps[i].y for i in snap_ids])
    def users(self, snap_ids): return np.concatenate([self.snaps[i].cu for i in snap_ids])