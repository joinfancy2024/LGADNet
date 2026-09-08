#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查询 Bailer-Jones 距离估计

从 VizieR TAP 服务查询 Gaia DR3 Bailer-Jones 距离估计，
包括几何距离和光度几何距离。

输入：vizier_dedup_double.csv（双重去重后的匹配结果）
输出：bailer_jones_distance_for_candidates.csv
"""

from __future__ import annotations

import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


INPUT = Path(
    "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/"
    "spatial_filtering_unique/vizier_dedup_double.csv"
)
OUTPUT = Path(
    "/home/DM13/workspace/sky/data/new_dataset3/lgadnet/"
    "spatial_filtering_unique/bailer_jones_distance_for_candidates.csv"
)

TAP_URL = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
BATCH_SIZE = 1000
SLEEP_SEC = 1.0


def query_batch(source_ids: list[int]) -> pd.DataFrame:
    """
    查询一批 Source ID 的 Bailer-Jones 距离。

    Parameters
    ----------
    source_ids : list[int]
        Gaia DR3 Source ID 列表

    Returns
    -------
    pd.DataFrame
        查询结果，包含距离估计和置信区间
    """
    source_list = ",".join(str(int(x)) for x in source_ids)

    query = f"""
    SELECT
        Source,
        rgeo AS r_med_geo,
        "b_rgeo" AS r_lo_geo,
        "B_rgeo" AS r_hi_geo,
        rpgeo AS r_med_photogeo,
        "b_rpgeo" AS r_lo_photogeo,
        "B_rpgeo" AS r_hi_photogeo,
        Flag AS bj_flag
    FROM "I/352/gedr3dis"
    WHERE Source IN ({source_list})
    """

    response = requests.post(
        TAP_URL,
        data={
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "csv",
            "QUERY": query,
        },
        timeout=120,
    )
    response.raise_for_status()

    text = response.text.strip()
    if not text or text.startswith("<?xml"):
        return pd.DataFrame()

    return pd.read_csv(StringIO(text))


def main() -> int:
    print("=" * 80)
    print("查询 Bailer-Jones 距离估计")
    print("=" * 80)

    # 读取输入文件
    df = pd.read_csv(INPUT, usecols=["Source"])
    sources = (
        pd.to_numeric(df["Source"], errors="coerce")
        .dropna()
        .astype("int64")
        .drop_duplicates()
        .tolist()
    )

    print(f"\n输入文件: {INPUT}")
    print(f"唯一 Source 数量: {len(sources):,}")

    # 分批查询
    all_parts = []
    total = len(sources)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\n批次大小: {BATCH_SIZE}")
    print(f"总批次数: {total_batches}")
    print(f"预计时间: ~{total_batches * SLEEP_SEC / 60:.1f} 分钟（不含网络时间）")
    print("-" * 80)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = sources[start:end]
        batch_num = start // BATCH_SIZE + 1

        print(f"批次 {batch_num}/{total_batches}: 查询 {start:,} - {end:,} / {total:,}")

        try:
            part = query_batch(batch)
            if len(part) > 0:
                all_parts.append(part)
            print(f"  ✓ 匹配数: {len(part):,}")
        except Exception as exc:
            print(f"  ✗ 错误: {exc}")

        time.sleep(SLEEP_SEC)

    # 合并结果
    print("\n" + "-" * 80)
    print("合并结果")

    if all_parts:
        out = pd.concat(all_parts, ignore_index=True)
        out = out.drop_duplicates("Source", keep="first")
    else:
        out = pd.DataFrame(
            columns=[
                "Source",
                "r_med_geo",
                "r_lo_geo",
                "r_hi_geo",
                "r_med_photogeo",
                "r_lo_photogeo",
                "r_hi_photogeo",
                "bj_flag",
            ]
        )

    # 保存输出
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)

    print("=" * 80)
    print("查询完成")
    print("=" * 80)
    print(f"输出文件: {OUTPUT}")
    print(f"匹配行数: {len(out):,}")
    print(f"匹配率: {len(out) / len(sources) * 100:.2f}%")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
