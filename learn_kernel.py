"""Обучаемые ядра затухания: полупериоды экспонент подбираются градиентом. Для каждого (семейство, канал) есть K экспонент с обучаемыми скоростями lambda_k:
    s_k(u,i,T) = sum_{t_j < T} a_j * exp(-lambda_k (T - t_j))
Признаки log1p(s) идут в небольшой MLP -> logit; всё дифференцируемо по lambda_k. После обучения полупериоды ln2/lambda_k выгружаются и используются как признаки
для бустинга (двухстадийная схема: нейросеть тюнит ядра, бустинг решает.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from data import CHANNELS
from features import LN2, FeatureStore

HL_MIN, HL_MAX = 0.01, 300.0


class KernelNet(nn.Module):
    def __init__(self, fams: list[str], K: int = 8, hidden: int = 128,
                 hl_init=(0.05, 90.0), seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.fams, self.K, self.C = fams, K, len(CHANNELS)
        init = np.geomspace(*hl_init, K)
        lr0 = torch.log(torch.tensor(LN2 / init, dtype=torch.float32))
        lr0 = lr0.repeat(len(fams), self.C, 1) + 0.15 * torch.randn(len(fams), self.C, K, generator=g)
        self.log_rate = nn.Parameter(lr0)
        d = len(fams) * self.C * K
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    def rates(self):
        lo, hi = np.log(LN2 / HL_MAX), np.log(LN2 / HL_MIN)
        return self.log_rate.clamp(lo, hi).exp()          # [F,C,K]

    def features(self, matched: dict, chunk: int = 1_000_000) -> torch.Tensor:
        parts, R = [], self.rates()
        for f, fam in enumerate(self.fams):
            m = matched[fam]
            S = torch.zeros(m.nq, self.C * self.K, device=R.device)
            for a in range(0, m.q.shape[0], chunk):
                dt = m.dt[a:a + chunk, None, None]
                w = torch.exp(-dt * R[f][None])                       # [c,C,K]
                v = m.val[a:a + chunk].float()[:, :, None]
                S.index_add_(0, m.q[a:a + chunk], (v * w).reshape(-1, self.C * self.K))
            parts.append(torch.log1p(S)[m.cand_q])
        return torch.cat(parts, 1)

    def forward(self, matched: dict, chunk: int = 1_000_000):
        return self.head(self.features(matched, chunk)).squeeze(-1)

    @torch.no_grad()
    def half_lives(self) -> dict:
        hl = (LN2 / self.rates()).cpu().numpy()
        return {(fam, ch): sorted(np.round(hl[f, c], 3).tolist())
                for f, fam in enumerate(self.fams) for c, ch in enumerate(CHANNELS)}

    @torch.no_grad()
    def importance(self) -> np.ndarray:
        """Средний |вес| первого слоя на (fam,ch,k)"""
        w = self.head[1].weight.abs().mean(0).view(len(self.fams), self.C, self.K)
        return w.cpu().numpy()


def _subset_users(snap, frac_users: int, seed: int):
    users = np.unique(snap.cu)
    rng = np.random.default_rng(seed)
    pick = rng.choice(users, size=min(frac_users, len(users)), replace=False)
    return np.flatnonzero(np.isin(snap.cu, pick))


def train_kernel_net(store: FeatureStore, train_ids: list[int], val_id: int, fams: list[str],
                     K: int = 8, kernel_users: int = 1500, epochs: int = 12, lr_rate: float = 5e-2,
                     lr_head: float = 2e-3, seed: int = 0, log=print) -> KernelNet:
    dev = store.dev
    torch.manual_seed(seed)
    net = KernelNet(fams, K=K, seed=seed).to(dev)
    opt = torch.optim.Adam([{"params": [net.log_rate], "lr": lr_rate},
                            {"params": net.head.parameters(), "lr": lr_head, "weight_decay": 1e-5}])
    lossf = nn.BCEWithLogitsLoss()

    def prep(si):
        idx = _subset_users(store.snaps[si], kernel_users, seed)
        m = {fam: store.build_matched(si, fam, idx) for fam in fams}
        y = torch.from_numpy(store.snaps[si].y[idx]).to(dev)
        return m, y, idx

    log(f"[kernel] подготовка данных ({len(train_ids)} train-снапшотов)...")
    train = [prep(si) for si in train_ids]
    val = prep(val_id)
    best, best_state = -1.0, None
    for ep in range(epochs):
        net.train(); t0 = time.time(); tot = 0.0
        for k in np.random.default_rng(ep).permutation(len(train)):
            m, y, _ = train[k]
            opt.zero_grad(set_to_none=True)
            loss = lossf(net(m), y)
            loss.backward(); opt.step(); tot += float(loss)
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(val[0])).cpu().numpy()
        auc = roc_auc_score(val[1].cpu().numpy(), pv)
        log(f"[kernel] epoch {ep:02d} loss={tot/len(train):.4f} val_auc={auc:.4f} ({time.time()-t0:.1f}s)")
        if auc > best:
            best, best_state = auc, {k_: v.clone() for k_, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net


@torch.no_grad()
def predict_snapshot(net: KernelNet, store: FeatureStore, si: int) -> np.ndarray:
    net.eval()
    m = {fam: store.build_matched(si, fam) for fam in net.fams}
    return torch.sigmoid(net(m)).cpu().numpy()