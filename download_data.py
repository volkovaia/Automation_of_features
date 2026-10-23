"""Скачивание и проверка Yambda (Yandex Music) с Hugging Face.

    python download_data.py                 # 50m: ~47.8 млн событий
    python download_data.py --size 500m     # больше (нужно много RAM/диска)
    python download_data.py --check-only    # только проверить уже скачанное

Файлы кладутся в кэш huggingface_hub (~/.cache/huggingface), оттуда их потом читает data.load_yambda() из run.py 
"""
from __future__ import annotations

import argparse
import sys

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "yandex/yambda"


def get(filename: str) -> str:
    print(f"-> {filename}")
    return hf_hub_download(REPO, filename, repo_type="dataset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="50m", choices=["50m", "500m", "5b"])
    ap.add_argument("--check-only", action="store_true",
                    help="не качать заново, только прочитать из кэша и проверить схему")
    ap.add_argument("--no-mappings", action="store_true",
                    help="не качать item->artist/album (признаки uart/ualb выродятся)")
    a = ap.parse_args()

    files = {"events": f"flat/{a.size}/multi_event.parquet"}
    if not a.no_mappings:
        files["artist"] = "artist_item_mapping.parquet"
        files["album"] = "album_item_mapping.parquet"

    paths = {}
    for k, f in files.items():
        try:
            paths[k] = hf_hub_download(REPO, f, repo_type="dataset",
                                       local_files_only=a.check_only)
            print(f"-> {f}: ok")
        except Exception as e:  # noqa: BLE001
            print(f"[error] {f}: {e}")
            if k == "events":
                sys.exit(1)

    # проверка схемы и базовая статистика (читаем метаданные, не весь файл)
    pf = pq.ParquetFile(paths["events"])
    print("\nevents: строк =", f"{pf.metadata.num_rows:,}", "| row groups =", pf.metadata.num_row_groups)
    print("колонки:", pf.schema_arrow.names)
    need = {"uid", "item_id", "timestamp", "event_type", "played_ratio_pct"}
    miss = need - set(pf.schema_arrow.names)
    if miss:
        print(f"[error] нет колонок {miss}: схема изменилась, поправьте data.load_yambda")
        sys.exit(1)

    head = pf.read_row_group(0, columns=["uid", "item_id", "timestamp", "event_type", "played_ratio_pct"]).to_pandas()
    print("\nпервые строки:\n", head.head())
    print("\nevent_type:", head["event_type"].astype(str).value_counts().to_dict())
    print("timestamp хранится в 5-секундных ячейках: 1 день =", 86400 // 5, "единиц")

    for k in ("artist", "album"):
        if k in paths:
            m = pq.read_table(paths[k]).to_pandas()
            print(f"\n{k}_item_mapping: {len(m):,} строк, колонки {list(m.columns)}")
            if "item_id" not in m.columns:
                print("[warn] нет колонки item_id — поправьте data._mapping")

    print("\nГотово")


if __name__ == "__main__":
    main()