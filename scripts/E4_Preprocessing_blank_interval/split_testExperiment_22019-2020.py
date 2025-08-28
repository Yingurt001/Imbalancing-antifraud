import os
import pandas as pd
import numpy as np
import random

# 固定参数
master_data_size_ori = 21
predict_month = 3
blank_intervals = list(range(1, 13))

# Step 1: 读取并处理原始数据
def load_and_process_data(input_file, fraction=0.1, seed=42):
    """
    读取 CSV 文件，按 LOAN SEQUENCE NUMBER 分组。
    支持基于贷款编号的分组抽样。
    """
    df = pd.read_csv(input_file)
    unique_loans = df['LOAN SEQUENCE NUMBER'].unique()
    np.random.seed(seed)

    if 0 < fraction < 1.0:
        sampled_loans = np.random.choice(unique_loans, size=int(len(unique_loans) * fraction), replace=False)
        df = df[df['LOAN SEQUENCE NUMBER'].isin(sampled_loans)]

    sorted_data = df.sort_values('LOAN SEQUENCE NUMBER')
    grouped_data = sorted_data.groupby('LOAN SEQUENCE NUMBER')
    return grouped_data, df


# Step 2: 构建测试集
def build_test_data(grouped_data, master_data_size_ori, predict_month, blank_interval, split_ratio, stats_accumulator):
    """
    构造测试样本：固定原始窗口 master_data_size_ori，变化 blank_interval 得到 master_data_size_final。
    """
    master_data_size_final = master_data_size_ori - blank_interval
    test_data_X = []
    test_data_Y = []

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

        if group_size >= master_data_size_ori + predict_month:
            # 特征：只取 master_data_size_final 长度
            X = group.iloc[0:master_data_size_final, 1:-1]
            # 标签：仍然基于原始窗口之后的 predict_month
            Y = group.iloc[master_data_size_ori:master_data_size_ori + predict_month, -1]

            for col in ['ZERO BALANCE CODE', 'REMAINING MONTHS TO LEGAL MATURITY']:
                if col in X.columns:
                    X = X.drop(columns=col)

            label = 1 if Y.sum() > 0 else 0
            test_data_X.append(X.values)
            test_data_Y.append(label)

    # 原始样本统计
    stats_accumulator["original_sample_size"] += total_rows_all_groups
    stats_accumulator["original_num_loans"] += len(all_group_sizes)
    stats_accumulator["original_total_loan_length"] += sum(all_group_sizes)
    stats_accumulator["original_num_defaults"] += num_defaults

    default = [(x, 1) for x, y in zip(test_data_X, test_data_Y) if y == 1]
    undefault = [(x, 0) for x, y in zip(test_data_X, test_data_Y) if y == 0]

    if len(default) == 0:
        return np.array([]), np.array([])

    if split_ratio == "original":
        combined = default + undefault
    else:
        if len(undefault) < split_ratio * len(default):
            combined = default + undefault
        else:
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
    print(f"✅ Saved to {output_dir} | Samples: {len(Y)}, Defaults: {int(Y.sum())}, Non-defaults: {len(Y)-int(Y.sum())}")


# 主流程
if __name__ == "__main__":
    input_files = [
        "data/processed_data/historical_data_time_2019Q1.csv",
        "data/processed_data/historical_data_time_2019Q2.csv",
        "data/processed_data/historical_data_time_2019Q3.csv",
        "data/processed_data/historical_data_time_2019Q4.csv"
    ]

    split_ratios = [1, "original"]
    stats_csv_path = "E2_test_data_summary_2019.csv"

    if os.path.exists(stats_csv_path):
        os.remove(stats_csv_path)
    first_write = True

    year_str = os.path.basename(input_files[0]).split('_')[-1][:4]

    for blank_interval in blank_intervals:
        master_data_size_final = master_data_size_ori - blank_interval

        for split_ratio in split_ratios:
            print(f"\n🔧 blank_interval = {blank_interval}, master_data_size_final = {master_data_size_final}, split_ratio = {split_ratio}")

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
                X, Y = build_test_data(grouped_data, master_data_size_ori, predict_month, blank_interval, split_ratio, stats)

                if X.size == 0:
                    continue

                all_X.append(X)
                all_Y.append(Y)

            if all_X and all_Y:
                final_X = np.concatenate(all_X, axis=0)
                final_Y = np.concatenate(all_Y, axis=0)

                if split_ratio == "original":
                    output_dir = f"data/test_merged_{year_str}_blank{blank_interval}_ratio_original"
                    ratio_str = "original"
                else:
                    output_dir = f"data/test_merged_{year_str}_blank{blank_interval}_ratio1to{split_ratio}"
                    ratio_str = f"1:{split_ratio}"

                save_data(final_X, final_Y, output_dir)

                avg_len = stats['original_total_loan_length']/stats['original_num_loans'] if stats['original_num_loans'] else 0
                final_avg_len = stats['final_total_loan_length']/stats['final_num_loans'] if stats['final_num_loans'] else 0
                default_rate = stats['original_num_defaults']/stats['original_num_loans'] if stats['original_num_loans'] else 0
                final_default_rate = stats['final_num_defaults']/stats['final_num_loans'] if stats['final_num_loans'] else 0

                result_row = {
                    "year": year_str,
                    "blank_interval": blank_interval,
                    "master_data_size_final": master_data_size_final,
                    "split_ratio": ratio_str,
                    "original_sample_size": stats['original_sample_size'],
                    "original_num_loans": stats['original_num_loans'],
                    "avg_loan_length": round(avg_len, 2),
                    "num_defaults": stats['original_num_defaults'],
                    "default_rate": round(default_rate, 4),
                    "final_sample_size": stats['final_sample_size'],
                    "final_num_loans": stats['final_num_loans'],
                    "final_avg_loan_length": round(final_avg_len, 2),
                    "final_num_defaults": stats['final_num_defaults'],
                    "final_default_rate": round(final_default_rate, 4)
                }

                df_row = pd.DataFrame([result_row])
                df_row.to_csv(stats_csv_path, mode='a', header=first_write, index=False)
                first_write = False

                print(f"📈 Stats written to {os.path.abspath(stats_csv_path)} | loans={stats['final_num_loans']}, defaults={stats['final_num_defaults']}")
            else:
                print("⚠️ No valid samples generated. Skipping.")
