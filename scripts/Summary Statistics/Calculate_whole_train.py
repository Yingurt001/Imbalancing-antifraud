import os, sys
import numpy as np
import pandas as pd
import csv
import random

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),  ".."))
sys.path.insert(0, ROOT)

SEED = 42
FRACTION = 1.0   # 可选：抽样加速

# =====================================
# Step 1: Load and group data by loan ID
# =====================================
def load_and_process_data(input_file, fraction=1.0, seed=42):
    df = pd.read_csv(input_file)
    unique_loans = df['LOAN SEQUENCE NUMBER'].unique()

    if 0 < fraction < 1.0:
        np.random.seed(seed)
        sampled_loans = np.random.choice(unique_loans, size=int(len(unique_loans) * fraction), replace=False)
        df = df[df['LOAN SEQUENCE NUMBER'].isin(sampled_loans)]

    sorted_data = df.sort_values('LOAN SEQUENCE NUMBER')
    grouped_data = sorted_data.groupby('LOAN SEQUENCE NUMBER')
    return grouped_data, df


# =====================================
# Step 2: Build base train data (只提取统计信息)
# =====================================
def collect_statistics(grouped_data):
    all_group_sizes = []
    num_defaults = 0
    total_rows_all_groups = 0

    for _, group in grouped_data:
        group = group.sort_values('REMAINING MONTHS TO LEGAL MATURITY', ascending=False)
        group_size = len(group)
        total_rows_all_groups += group_size
        all_group_sizes.append(group_size)

        # 最后一列是标签
        if group.iloc[:, -1].sum() > 0:
            num_defaults += 1

    return total_rows_all_groups, len(all_group_sizes), sum(all_group_sizes), num_defaults


# =====================================
# Step 3: Main loop (只统计原始数据，不做采样)
# =====================================
if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)

    input_files = [
        "data/processed_data/historical_data_time_2022Q1.csv",
        "data/processed_data/historical_data_time_2022Q2.csv",
        "data/processed_data/historical_data_time_2022Q3.csv",
        "data/processed_data/historical_data_time_2022Q4.csv"
    ]

    stats_csv_path = "2022_Statistics_summary_original.csv"
    write_header = not os.path.exists(stats_csv_path)

    # 年份从文件名里提取（假设同一年）
    first_file = os.path.basename(input_files[0])
    year_str = next((part[:4] for part in first_file.split('_') if part[:4].isdigit()), "Unknown")

    total_sample_size = 0
    total_num_loans = 0
    total_loan_length = 0
    total_num_defaults = 0

    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"❌ File not found: {input_file}")
            continue

        grouped_data, _ = load_and_process_data(input_file, fraction=FRACTION, seed=SEED)
        sample_size, num_loans, loan_length, num_defaults = collect_statistics(grouped_data)

        total_sample_size += sample_size
        total_num_loans += num_loans
        total_loan_length += loan_length
        total_num_defaults += num_defaults

    if total_num_loans > 0:
        avg_loan_length = total_loan_length / total_num_loans
        default_rate = total_num_defaults / total_num_loans
    else:
        avg_loan_length = 0
        default_rate = 0

    # 写入 CSV
    with open(stats_csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow([
                "year",
                "original_sample_size",
                "original_num_loans",
                "avg_loan_length",
                "num_defaults",
                "default_rate"
            ])
            write_header = False

        writer.writerow([
            year_str,
            total_sample_size,
            total_num_loans,
            round(avg_loan_length, 2),
            total_num_defaults,
            round(default_rate, 4)
        ])

    print(f"✅ Stats saved to {stats_csv_path}")
