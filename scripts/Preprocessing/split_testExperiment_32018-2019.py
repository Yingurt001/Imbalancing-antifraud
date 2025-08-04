import os
import pandas as pd
import numpy as np
import random
import csv

# Step 1: 读取数据并排序
def load_and_process_data(input_file):
    df = pd.read_csv(input_file)
    sorted_data = df.sort_values('LOAN SEQUENCE NUMBER')
    grouped_data = sorted_data.groupby('LOAN SEQUENCE NUMBER')
    return grouped_data, df

# Step 2: 构建测试集
def build_test_data(grouped_data, master_data_size, split_ratio, stats_accumulator):
    test_data_X = []
    test_data_Y = []
    predict_month = 3

    num_defaults = 0
    total_rows_all_groups = 0
    all_group_sizes = []

    for _, group in grouped_data:
        group = group.sort_values('REMAINING MONTHS TO LEGAL MATURITY', ascending=False)
        group_size = len(group)
        total_rows_all_groups += group_size
        all_group_sizes.append(group_size)

        if group.iloc[:, -1].sum() > 0:
            num_defaults += 1

        if group_size >= master_data_size:
            X = group.iloc[0:master_data_size - predict_month - 1, 1:-1]
            Y = group.iloc[master_data_size - predict_month - 1:master_data_size - 1, -1]

            if 'ZERO BALANCE CODE' in X.columns:
                X = X.drop(['ZERO BALANCE CODE'], axis=1)
            if 'REMAINING MONTHS TO LEGAL MATURITY' in X.columns:
                X = X.drop(['REMAINING MONTHS TO LEGAL MATURITY'], axis=1)

            label = 1 if Y.sum() > 0 else 0
            test_data_X.append(X.values)
            test_data_Y.append(label)

    stats_accumulator["original_sample_size"] += total_rows_all_groups
    stats_accumulator["original_num_loans"] += len(all_group_sizes)
    stats_accumulator["original_total_loan_length"] += sum(all_group_sizes)
    stats_accumulator["original_num_defaults"] += num_defaults

    default = [(x, 1) for x, y in zip(test_data_X, test_data_Y) if y == 1]
    undefault = [(x, 0) for x, y in zip(test_data_X, test_data_Y) if y == 0]
    sampled_undefault = undefault[:split_ratio * len(default)]
    combined = default + sampled_undefault
    random.shuffle(combined)

    stats_accumulator["final_sample_size"] += sum([len(x[0]) for x in combined])
    stats_accumulator["final_num_loans"] += len(combined)
    stats_accumulator["final_total_loan_length"] += sum([len(x[0]) for x in combined])
    stats_accumulator["final_num_defaults"] += sum([y for _, y in combined])

    if not combined:
        return np.array([]), np.array([])

    X, Y = zip(*combined)
    return np.array(X, dtype='float32'), np.array(Y, dtype='float32')

# Step 3: 保存数据
def save_data(X, Y, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'testX.npy'), X)
    np.save(os.path.join(output_dir, 'testY.npy'), Y)
    print(f"✅ Saved to {output_dir} | Samples: {len(Y)}, Defaults: {int(Y.sum())}, Non-defaults: {len(Y) - int(Y.sum())}")

# 主流程
if __name__ == "__main__":
    input_files = [
        "data/processed_data/historical_data_time_2019Q1.csv",
        "data/processed_data/historical_data_time_2019Q2.csv",
        "data/processed_data/historical_data_time_2019Q3.csv",
        "data/processed_data/historical_data_time_2019Q4.csv"
    ]

    master_data_sizes = [18, 21, 24, 27]
    split_ratios = [1,2,5,10,25]

    stats_csv_path = "test_data_summary.csv"
    write_header = not os.path.exists(stats_csv_path)

    year_str = os.path.basename(input_files[0]).split('_')[-1][:4]

    with open(stats_csv_path, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow([
                "year", "master_data_size", "split_ratio",
                "original_sample_size", "original_num_loans", "avg_loan_length", "num_defaults", "default_rate",
                "final_sample_size", "final_num_loans", "final_avg_loan_length", "final_num_defaults", "final_default_rate"
            ])
            write_header = False
            csvfile.flush()  # ✅ 写完表头就立即 flush

        for master_data_size in master_data_sizes:
            for split_ratio in split_ratios:
                print(f"\n🔧 master_data_size = {master_data_size}, split_ratio = 1:{split_ratio}")

                all_X, all_Y = [], []
                stats = {
                    "original_sample_size": 0,
                    "original_num_loans": 0,
                    "original_total_loan_length": 0,
                    "original_num_defaults": 0,
                    "final_sample_size": 0,
                    "final_num_loans": 0,
                    "final_total_loan_length": 0,
                    "final_num_defaults": 0
                }

                for input_file in input_files:
                    if not os.path.exists(input_file):
                        print(f"❌ File not found: {input_file}")
                        continue

                    grouped_data, _ = load_and_process_data(input_file)
                    X, Y = build_test_data(grouped_data, master_data_size, split_ratio, stats)

                    if X.size == 0:
                        continue

                    all_X.append(X)
                    all_Y.append(Y)

                if all_X and all_Y:
                    final_X = np.concatenate(all_X, axis=0)
                    final_Y = np.concatenate(all_Y, axis=0)

                    output_dir = f"data/test_merged_{year_str}_master{master_data_size}_ratio1to{split_ratio}"
                    save_data(final_X, final_Y, output_dir)

                    avg_len = stats['original_total_loan_length'] / stats['original_num_loans'] if stats['original_num_loans'] else 0
                    final_avg_len = stats['final_total_loan_length'] / stats['final_num_loans'] if stats['final_num_loans'] else 0
                    default_rate = stats['original_num_defaults'] / stats['original_num_loans'] if stats['original_num_loans'] else 0
                    final_default_rate = stats['final_num_defaults'] / stats['final_num_loans'] if stats['final_num_loans'] else 0

                    writer.writerow([
                        year_str,
                        master_data_size,
                        split_ratio,
                        stats['original_sample_size'],
                        stats['original_num_loans'],
                        round(avg_len, 2),
                        stats['original_num_defaults'],
                        round(default_rate, 4),
                        stats['final_sample_size'],
                        stats['final_num_loans'],
                        round(final_avg_len, 2),
                        stats['final_num_defaults'],
                        round(final_default_rate, 4)
                    ])
                    csvfile.flush()  # ✅ 每写一行都立即写入磁盘

                    print(f"📈 Stats recorded: {stats['final_num_loans']} loans | {stats['final_num_defaults']} defaults")
                else:
                    print("⚠️ No valid samples generated. Skipping.")
