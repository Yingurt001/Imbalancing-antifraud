import os
import pandas as pd
import numpy as np
import random
import csv

# Step 1: 读取数据并排序
def load_and_process_data(input_file):
    encoded_MonthlyData = pd.read_csv(input_file)
    sorted_data = encoded_MonthlyData.sort_values('LOAN SEQUENCE NUMBER')
    grouped_data = sorted_data.groupby('LOAN SEQUENCE NUMBER')
    return grouped_data, encoded_MonthlyData

# Step 2: 初始化变量并构建测试集和标签
def build_train_data(grouped_data, master_data_size, stats_accumulator):
    train_data_X = []
    train_data_Y = []
    predict_month = 3
    SPLIT = 1  # 控制未违约样本的比例

    all_group_sizes = []
    total_rows_all_groups = 0
    num_defaults = 0

    for name, group in grouped_data:
        group = group.sort_values('REMAINING MONTHS TO LEGAL MATURITY', ascending=False)
        group_size = len(group)
        all_group_sizes.append(group_size)
        total_rows_all_groups += group_size

        # 检查 default（只看 label 列是否出现 1）
        if group.iloc[:, -1].sum() > 0:
            num_defaults += 1

        if group_size >= master_data_size:
            train_X = group.iloc[0:master_data_size - predict_month - 1, 1:-1]
            train_Y = group.iloc[master_data_size - predict_month - 1:master_data_size - 1, -1]

            if 'ZERO BALANCE CODE' in train_X.columns and 'REMAINING MONTHS TO LEGAL MATURITY' in train_X.columns:
                train_X = train_X.drop(['ZERO BALANCE CODE', 'REMAINING MONTHS TO LEGAL MATURITY'], axis=1)

            binary_label = 1 if train_Y.sum() > 0 else 0
            train_data_X.append(train_X.values)
            train_data_Y.append(binary_label)

    # ✅ 原始统计（无筛选）
    stats_accumulator["original_sample_size"] += total_rows_all_groups
    stats_accumulator["original_num_loans"] += len(all_group_sizes)
    stats_accumulator["original_total_loan_length"] += sum(all_group_sizes)
    stats_accumulator["original_num_defaults"] += num_defaults

    # ✅ 欠采样
    default = []
    undefault = []
    for i in range(len(train_data_Y)):
        if train_data_Y[i] == 1:
            default.append((train_data_X[i], 1))
        else:
            undefault.append((train_data_X[i], 0))

    combined_list = default + undefault[:SPLIT * len(default)]
    random.shuffle(combined_list)

    train_X = []
    train_Y = []
    for X, Y in combined_list:
        train_X.append(X)
        train_Y.append(Y)

    stats_accumulator["final_sample_size"] += sum([len(x) for x in train_X])
    stats_accumulator["final_num_loans"] += len(train_X)
    stats_accumulator["final_total_loan_length"] += sum([len(x) for x in train_X])
    stats_accumulator["final_num_defaults"] += sum(train_Y)

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    return train_X, train_Y

# Step 3: 保存
def save_data(train_X, train_Y, output_dir='data/npy_train'):
    train_X = train_X.astype('float32')
    train_Y = train_Y.astype('float32')
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'trainX.npy'), train_X)
    np.save(os.path.join(output_dir, 'trainY.npy'), train_Y)
    print(f"✅ Saved to {output_dir}")


if __name__ == "__main__":
    input_files = [
        "data/processed_data/historical_data_time_2018Q1.csv",
        "data/processed_data/historical_data_time_2018Q2.csv",
        "data/processed_data/historical_data_time_2018Q3.csv",
        "data/processed_data/historical_data_time_2018Q4.csv"
    ]

    master_data_sizes = [12]

    # 统计文件路径
    stats_csv_path = "statistics_summary.csv"
    write_header = not os.path.exists(stats_csv_path)

    # 提取年份（从第一个文件名中提取 '2018Q1' -> '2018'）
    first_file = os.path.basename(input_files[0])
    year_str = next((part[:4] for part in first_file.split('_') if part[:4].isdigit()), "Unknown")

    for master_data_size in master_data_sizes:
        print(f"\n🔧 Processing master_data_size = {master_data_size} ...")

        all_train_X = []
        all_train_Y = []

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

            grouped_data, full_df = load_and_process_data(input_file)
            train_X, train_Y = build_train_data(grouped_data, master_data_size, stats)

            all_train_X.append(train_X)
            all_train_Y.append(train_Y)

            print(f"  ✅ File {os.path.basename(input_file)} -> Samples: {len(train_Y)}")

        if all_train_X and all_train_Y:
            final_train_X = np.concatenate(all_train_X, axis=0)
            final_train_Y = np.concatenate(all_train_Y, axis=0)

            output_dir = f'data/npy_merged_master_data_size_{master_data_size}'
            save_data(final_train_X, final_train_Y, output_dir=output_dir)

            avg_len = stats['original_total_loan_length'] / stats['original_num_loans'] if stats['original_num_loans'] else 0
            final_avg_len = stats['final_total_loan_length'] / stats['final_num_loans'] if stats['final_num_loans'] else 0
            default_rate = stats['original_num_defaults'] / stats['original_num_loans'] if stats['original_num_loans'] else 0
            final_default_rate = stats['final_num_defaults'] / stats['final_num_loans'] if stats['final_num_loans'] else 0

            # ✅ 追加写入 CSV 文件
            with open(stats_csv_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                if write_header:
                    writer.writerow([
                        "year",  # 新增年份列
                        "master_data_size",
                        "original_sample_size",
                        "original_num_loans",
                        "avg_loan_length",
                        "num_defaults",
                        "default_rate",
                        "final_sample_size",
                        "final_num_loans",
                        "final_avg_loan_length",
                        "final_num_defaults",
                        "final_default_rate"
                    ])
                    write_header = False

                writer.writerow([
                    year_str,
                    master_data_size,
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

            print(f"📈 Stats saved to {stats_csv_path}")
        else:
            print(f"⚠️ Skipped saving for master_data_size = {master_data_size}, no data.")
