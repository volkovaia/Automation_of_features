"""Данные: загрузка (Yambda / generic parquet / синтетика), кандидаты, метрики.

Все времена хранятся в ДНЯХ (float64) от начала датасета.
Каналы событий (колонки val):  0 = listen (любое прослушивание),
                               1 = good   (прослушано >= 50%),
                               2 = like.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

CHANNELS = ["listen", "good", "like"]


@dataclass
class Events:
    u: np.ndarray      # int32 [n]   user (dense 0..n_users-1)
    i: np.ndarray      # int32 [n]   item (dense 0..n_items-1)
    t: np.ndarray      # float64 [n] дни, отсортировано по возрастанию
    val: np.ndarray    # uint8 [n,3] флаги каналов
    n_users: int
    n_items: int
    art: np.ndarray    # int32 [n_items] артист айтема (dense)
    alb: np.ndarray    # int32 [n_items] альбом айтема (dense)
    n_art: int
    n_alb: int

    def sort(self) -> "Events":
        o = np.argsort(self.t, kind="stable")
        return Events(self.u[o], self.i[o], self.t[o], self.val[o], self.n_users,
                      self.n_items, self.art, self.alb, self.n_art, self.n_alb)


def _dense(a: np.ndarray):
    uniq, inv = np.unique(a, return_inverse=True)
    return inv.astype(np.int32), uniq


def load_yambda(size: str = "50m", n_users: int = 3000, seed: int = 0) -> Events:
    """flat/{size}/multi_event.parquet. timestamp в Yambda = 5-секундные ячейки."""
    from huggingface_hub import hf_hub_download

    f = hf_hub_download("yandex/yambda", f"flat/{size}/multi_event.parquet", repo_type="dataset")
    uid, item, ts, is_listen, is_like, played = _read_yambda_events(f, n_users, seed)
    val = np.zeros((len(uid), 3), dtype=np.uint8)
    val[:, 0] = is_listen
    val[:, 1] = is_listen & (played >= 50)
    val[:, 2] = is_like
    t = ts.astype(np.float64) * 5.0 / 86400.0       # timestamp = 5-секундные ячейки
    t -= t.min()
    u, _ = _dense(uid)
    i, items_orig = _dense(item)
    n_items = len(items_orig)
    art, n_art = _mapping("artist_item_mapping.parquet", items_orig)
    alb, n_alb = _mapping("album_item_mapping.parquet", items_orig)
    return Events(u, i, t, val, int(u.max()) + 1, n_items, art, alb, n_art, n_alb).sort()


def _read_yambda_events(path: str, n_users: int, seed: int):
    """Чтение через pyarrow: event_type в файле - dictionary(uint32), pandas такое не конвертирует."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    uids = pq.read_table(path, columns=["uid"])["uid"].unique().to_numpy()
    rng = np.random.default_rng(seed)
    sel = rng.choice(uids, size=min(n_users, len(uids)), replace=False)
    tbl = pq.read_table(path, columns=["uid", "item_id", "timestamp", "event_type", "played_ratio_pct"],
                        filters=[("uid", "in", [int(x) for x in sel])])
    et = tbl["event_type"]
    if pa.types.is_dictionary(et.type):
        et = pa.chunked_array([c.dictionary_decode() for c in et.chunks], type=et.type.value_type)
    is_listen = pc.equal(et, "listen").to_numpy(zero_copy_only=False)
    is_like = pc.equal(et, "like").to_numpy(zero_copy_only=False)
    keep = is_listen | is_like
    played = pc.fill_null(tbl["played_ratio_pct"], 0).to_numpy()
    uid, item, ts = (tbl[c].to_numpy() for c in ("uid", "item_id", "timestamp"))
    return uid[keep], item[keep], ts[keep], is_listen[keep], is_like[keep], played[keep]


def _mapping(fname: str, items_orig: np.ndarray):
    """item -> artist/album. Если файла нет, группа = сам айтем (фича вырождается)."""
    try:
        from huggingface_hub import hf_hub_download
        m = pd.read_parquet(hf_hub_download("yandex/yambda", fname, repo_type="dataset"))
        other = [c for c in m.columns if c != "item_id"][0]
        s = pd.Series(m[other].to_numpy(), index=m["item_id"].to_numpy())
        s = s[~s.index.duplicated()]
        g = s.reindex(items_orig).to_numpy()
        miss = np.isnan(g.astype(float))
        g = np.where(miss, -1 - np.arange(len(g)), g)  # уникальные группы для неизвестных
        dense, uniq = _dense(g)
        return dense, len(uniq)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] {fname}: {e}; использую item как группу")
        return np.arange(len(items_orig), dtype=np.int32), len(items_orig)


def load_generic(path: str, ts_unit_days: float = 1 / 86400.0) -> Events:
    """parquet/csv с колонками user,item,timestamp,label (0 view, 1 click, 2 buy); опционально c1 (категория айтема), c2 (вторая категория)."""
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    t = df["timestamp"].to_numpy().astype(np.float64) * ts_unit_days
    t -= t.min()
    lab = df["label"].to_numpy()
    val = np.stack([lab >= 0, lab >= 1, lab >= 2], 1).astype(np.uint8)
    u, _ = _dense(df["user"].to_numpy())
    i, items_orig = _dense(df["item"].to_numpy())
    n_items = len(items_orig)

    def grp(col):
        if col not in df:
            return np.arange(n_items, dtype=np.int32), n_items
        s = pd.Series(df[col].to_numpy(), index=df["item"].to_numpy())
        s = s[~s.index.duplicated()].reindex(items_orig)
        d, uq = _dense(s.fillna(-1).to_numpy())
        return d, len(uq)

    art, n_art = grp("c1")
    alb, n_alb = grp("c2")
    return Events(u, i, t, val, int(u.max()) + 1, n_items, art, alb, n_art, n_alb).sort()


# synthetic
def make_synthetic(n_users=800, n_art=150, items_per_art=8, days=100, per_day=25,
                   true_half_life=10.0, seed=0) -> Events:
    """Пользователь "заражается" артистом: интерес s_ua(t) затухает с известным полупериодом true_half_life и подпитывается прослушиваниями. Ground truth для
    проверки: обученное ядро по семейству uart должно сходиться к true_half_life."""
    rng = np.random.default_rng(seed)
    base = rng.dirichlet(np.full(n_art, 0.15), size=n_users) * 0.4 + 0.6 / n_art
    item_pop = rng.zipf(1.6, size=n_art * items_per_art).clip(max=50).astype(float)
    item_pop = item_pop.reshape(n_art, items_per_art)
    item_pop /= item_pop.sum(1, keepdims=True)
    decay = 0.5 ** (1.0 / true_half_life)
    s = np.zeros((n_users, n_art))
    U, I, T, V = [], [], [], []
    for d in range(days):
        p = base + 0.5 * s / (1.0 + s.sum(1, keepdims=True) * 0.02)
        p /= p.sum(1, keepdims=True)
        cdf = np.cumsum(p, 1)
        r = rng.random((n_users, per_day))
        a = (cdf[:, None, :] < r[:, :, None]).sum(-1).clip(max=n_art - 1)
        k = np.empty_like(a)
        for j in range(n_art):                    
            m = a == j
            if m.any():
                k[m] = rng.choice(items_per_art, size=m.sum(), p=item_pop[j])
        item = a * items_per_art + k
        uu = np.repeat(np.arange(n_users), per_day)
        aa, ii = a.ravel(), item.ravel()
        tt = d + rng.random(uu.shape[0])
        pg = np.clip(0.25 + 0.15 * s[uu, aa] / (1 + s[uu, aa]) + 0.1 * (base[uu, aa] * n_art > 2), 0, 0.95)
        good = rng.random(uu.shape[0]) < pg
        like = good & (rng.random(uu.shape[0]) < 0.06)
        v = np.stack([np.ones_like(good), good, like], 1).astype(np.uint8)
        U.append(uu); I.append(ii); T.append(tt); V.append(v)
        s *= decay
        np.add.at(s, (uu, aa), 0.5 * good)
    u = np.concatenate(U).astype(np.int32); i = np.concatenate(I).astype(np.int32)
    art = (np.arange(n_art * items_per_art) // items_per_art).astype(np.int32)
    return Events(u, i, np.concatenate(T), np.concatenate(V), n_users, n_art * items_per_art,
                  art, art.copy(), n_art, n_art).sort()


#  snapshots
@dataclass
class Snapshot:
    T: float
    cu: np.ndarray      # int32 [N] user кандидата (отсортировано по user)
    ci: np.ndarray      # int32 [N] item кандидата
    y: np.ndarray       # float32 [N]
    novel: np.ndarray   # bool [N] пара user-item не встречалась в истории окна


def build_snapshot(ev: Events, T: float, horizon: float = 1.0, hist_days: float = 30.0,
                   n_hist: int = 150, n_pop: int = 50, label_ch=(1, 2)) -> Snapshot:
    """Кандидаты (без подглядывания в таргет): недавняя история юзера + глобально популярные за 7 дней. Метка: был good-listen/like пары в [T, T+horizon)."""
    lo = np.searchsorted(ev.t, T - hist_days); hi = np.searchsorted(ev.t, T)
    u, i, t = ev.u[lo:hi], ev.i[lo:hi], ev.t[lo:hi]
    df = pd.DataFrame({"u": u, "i": i, "w": np.exp(-(T - t) * np.log(2) / 7.0)})
    g = df.groupby(["u", "i"], sort=False)["w"].sum().reset_index()
    hist_key = g["u"].to_numpy().astype(np.int64) * ev.n_items + g["i"].to_numpy()
    g = g.sort_values(["u", "w"], ascending=[True, False]).groupby("u").head(n_hist)

    p_lo = np.searchsorted(ev.t, T - 7.0)
    pm = ev.val[p_lo:hi, 1] > 0
    pop = np.bincount(ev.i[p_lo:hi][pm], minlength=ev.n_items)
    top = np.argsort(-pop)[:n_pop]
    users = np.unique(g["u"].to_numpy())
    cu = np.concatenate([g["u"].to_numpy(), np.repeat(users, len(top))])
    ci = np.concatenate([g["i"].to_numpy(), np.tile(top, len(users))])
    key = np.unique(cu.astype(np.int64) * ev.n_items + ci)
    cu = (key // ev.n_items).astype(np.int32); ci = (key % ev.n_items).astype(np.int32)

    a = np.searchsorted(ev.t, T); b = np.searchsorted(ev.t, T + horizon)
    pm = np.zeros(b - a, bool)
    for c in label_ch:
        pm |= ev.val[a:b, c] > 0
    pos_key = np.unique(ev.u[a:b][pm].astype(np.int64) * ev.n_items + ev.i[a:b][pm])
    y = np.isin(key, pos_key).astype(np.float32)
    novel = ~np.isin(key, hist_key)
    # покрытие: доля позитивов юзеров-кандидатов, попавших в кандидаты
    pu = pos_key // ev.n_items
    cov = np.isin(key, pos_key[np.isin(pu, users)]).sum() / max(1, np.isin(pu, users).sum())
    print(f"  snapshot T={T:7.2f}: N={len(key):,} users={len(users):,} pos_rate={y.mean():.4f} "
          f"coverage={cov:.3f} novel={novel.mean():.2f}")
    return Snapshot(T, cu, ci, y, novel)


# metrics
def ranking_metrics(u: np.ndarray, y: np.ndarray, score: np.ndarray, ks=(10,)) -> dict:
    """NDCG@k и Recall@k по юзерам, у которых есть хотя бы 1 позитив среди кандидатов."""
    df = pd.DataFrame({"u": u, "y": y, "s": score})
    df = df.sort_values(["u", "s"], ascending=[True, False], kind="stable")
    df["r"] = df.groupby("u").cumcount().to_numpy()
    npos = df.groupby("u")["y"].transform("sum")
    df = df[npos > 0]
    if df.empty:
        return {f"ndcg@{k}": float("nan") for k in ks}
    disc = 1.0 / np.log2(df["r"].to_numpy() + 2.0)
    n_u = df.groupby("u")["y"].sum()
    out = {}
    for k in ks:
        inside = (df["r"].to_numpy() < k) & (df["y"].to_numpy() > 0)
        dcg = pd.Series(disc * inside, index=df.index).groupby(df["u"]).sum()
        ideal = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(k) + 2.0))])
        idcg = ideal[np.minimum(n_u.to_numpy(), k).astype(int)]
        hits = pd.Series(inside.astype(float), index=df.index).groupby(df["u"]).sum()
        out[f"ndcg@{k}"] = float((dcg.to_numpy() / idcg).mean())
        out[f"recall@{k}"] = float((hits.to_numpy() / n_u.to_numpy()).mean())
    return out


def per_user_table(u, y, score, k: int = 10) -> pd.DataFrame:
    """Метрики по каждому пользователю с хотя бы одним позитивом среди кандидатов: auc_user (NaN, если нет негативов), ndcg@k, recall@k. Нужна для бутстрэпа по юзерам."""
    df = pd.DataFrame({"u": u, "y": np.asarray(y, dtype=np.float64), "s": score})
    df["npos"] = df.groupby("u")["y"].transform("sum")
    df["n"] = df.groupby("u")["y"].transform("size")
    df = df[df["npos"] > 0].copy()
    df["r_asc"] = df.groupby("u")["s"].rank(method="average")
    npos = df.groupby("u")["npos"].first(); n = df.groupby("u")["n"].first(); nneg = n - npos
    pos_rank_sum = (df["r_asc"] * df["y"]).groupby(df["u"]).sum()
    auc = (pos_rank_sum - npos * (npos + 1) / 2) / (npos * nneg)
    auc[nneg == 0] = np.nan
    df = df.sort_values(["u", "s"], ascending=[True, False], kind="stable")
    r = df.groupby("u").cumcount().to_numpy()
    inside = (r < k) & (df["y"].to_numpy() > 0)
    dcg = pd.Series(inside / np.log2(r + 2.0), index=df.index).groupby(df["u"]).sum()
    hits = pd.Series(inside.astype(float), index=df.index).groupby(df["u"]).sum()
    ideal = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(k) + 2.0))])
    idcg = pd.Series(ideal[np.minimum(npos.to_numpy(), k).astype(int)], index=npos.index)
    out = pd.DataFrame({"auc_user": auc, f"ndcg@{k}": dcg / idcg, f"recall@{k}": hits / npos})
    out.index.name = "u"
    return out


def full_metrics(snap: Snapshot, score: np.ndarray, prob: bool = True, return_table: bool = False):
    y = snap.y
    tab = per_user_table(snap.cu, y, score)
    m = {"auc": float(roc_auc_score(y, score)),            # общий AUC (по всем парам подряд)
         "auc_user": float(tab["auc_user"].mean()),        # средний AUC внутри пользователя
         "ndcg@10": float(tab["ndcg@10"].mean()), "recall@10": float(tab["recall@10"].mean())}
    if prob:
        m["logloss"] = float(log_loss(y, np.clip(score, 1e-6, 1 - 1e-6)))
    nv = snap.novel
    if nv.sum() > 100 and 0 < y[nv].sum() < nv.sum():
        m["auc_novel"] = float(roc_auc_score(y[nv], score[nv]))
        mm = ranking_metrics(snap.cu[nv], y[nv], score[nv], ks=(10,))
        m["ndcg@10_novel"] = mm["ndcg@10"]
    return (m, tab) if return_table else m


def build_splits(ev: Events, horizon=1.0, hist_days=30.0, n_train=6, step=3.0, n_hist=150, n_pop=50,
                 n_test=1):
    """Временное разбиение: train-снапшоты -> val -> n_test подряд идущих тестовых дней. Окна меток не пересекаются; признаки каждого снапшота - только по событиям до его T.
    Возвращает snaps, train_ids, val_id, test_ids (список)."""
    span = float(ev.t.max())
    T_tests = [span - (n_test - j) * horizon for j in range(n_test)]
    T_val = T_tests[0] - horizon
    T_train = [T_val - horizon - step * k for k in range(n_train)][::-1]
    assert T_train[0] - hist_days > 0, "мало данных: уменьшите n_train/step/hist_days/n_test"
    snaps = [build_snapshot(ev, T, horizon, hist_days, n_hist, n_pop) for T in T_train + [T_val] + T_tests]
    return snaps, list(range(n_train)), n_train, [n_train + 1 + j for j in range(n_test)]