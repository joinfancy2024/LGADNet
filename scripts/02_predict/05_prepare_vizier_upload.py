#!/usr/bin/env python3
"""准备VizieR交叉匹配上传文件"""
import pandas as pd
from pathlib import Path

# 配置
INPUT_FILE = Path("/home/DM13/workspace/sky/data/new_dataset3/lgadnet/spatial_filtering_unique/cemp_unique_b_greater_30.csv")
OUTPUT_DIR = Path("/home/DM13/workspace/sky/data/new_dataset3/lgadnet/spatial_filtering_unique")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # 读取数据
    print(f"读取: {INPUT_FILE.name}")
    df = pd.read_csv(INPUT_FILE)
    print(f"候选数量: {len(df):,}")

    # 确保ID列
    if 'candidate_id' not in df.columns:
        df['candidate_id'] = range(len(df))

    # 保存上传文件
    output_file = OUTPUT_DIR / "vizier_upload_with_id.csv"
    df[['candidate_id', 'ra', 'dec']].to_csv(output_file, index=False)
    print(f"保存: {output_file.name}")
    print("完成！")


if __name__ == "__main__":
    main()
