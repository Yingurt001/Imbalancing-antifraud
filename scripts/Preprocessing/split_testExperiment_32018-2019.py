import os
import pandas as pd
import numpy as np
import random
import csv

# Step 1: 读取并处理原始数据
def load_and_process_data(input_file, fraction=0.1, seed=42):
    """
    读取 CSV 文件，按 LOAN SEQUENCE NUMBER 分组，用于按贷款追踪每个样本的历史序列。
    支持基于贷款编号的分组抽样（而非逐行抽样）。
    """
    df = pd.read_csv(input_file)

    # 先获取所有的贷款编号（唯一的 LOAN SEQUENCE NUMBER）
    unique_loans = df['LOAN SEQUENCE NUMBER'].unique()
    np.random.seed(seed)

    if 0 < fraction < 1.0:
        sampled_loans = np.random.choice(unique_loans, size=int(len(unique_loans) * fraction), replace=False)
        df = df[df['LOAN SEQUENCE NUMBER'].isin(sampled_loans)]

    sorted_data = df.sort_values('LOAN SEQUENCE NUMBER')  # 保证分组前排序
    grouped_data = sorted_data.groupby('LOAN SEQUENCE NUMBER')  # 按贷款编号分组
    return grouped_data, df


# Step 2: 构建测试集
def build_test_data(grouped_data, master_data_size, split_ratio, stats_accumulator):
    """
    构造测试样本，每个贷款截取固定长度的历史（master_data_size），判断是否违约。

    参数：
    - grouped_data: 按贷款分组后的数据
    - master_data_size: 每个样本截取的时间步长度
    - split_ratio: 欠采样比例（非违约 : 违约）
    - stats_accumulator: 用于累积统计量的字典

    返回：
    - X, Y: numpy 格式的测试特征和标签
    """
    test_data_X = []
    test_data_Y = []
    predict_month = 3  # 留 3 个月做预测窗口

    num_defaults = 0
    total_rows_all_groups = 0
    all_group_sizes = []

    for _, group in grouped_data:
        # 按剩余月份倒序排列，确保时间一致性
        group = group.sort_values('REMAINING MONTHS TO LEGAL MATURITY', ascending=False)
        group_size = len(group)
        total_rows_all_groups += group_size
        all_group_sizes.append(group_size)

        if group.iloc[:, -1].sum() > 0:  # 如果某贷款存在违约
            num_defaults += 1
        
        if group_size >= master_data_size + predict_month:
            # 截取前一段作为特征
            X = group.iloc[0:master_data_size, 1:-1]
            # 接下来的 predict_month 作为标签判断未来违约
            Y = group.iloc[master_data_size:master_data_size + predict_month, -1]

            # 删除可能导致泄漏或高度相关的列
            cols_to_remove = ['ZERO BALANCE CODE', 'REMAINING MONTHS TO LEGAL MATURITY']
            for col in cols_to_remove:
                if col in X.columns:
                    X = X.drop(columns=col)

            label = 1 if Y.sum() > 0 else 0
            test_data_X.append(X.values)
            test_data_Y.append(label)


    # 原始样本统计信息记录
    stats_accumulator["original_sample_size"] += total_rows_all_groups
    stats_accumulator["original_num_loans"] += len(all_group_sizes)
    stats_accumulator["original_total_loan_length"] += sum(all_group_sizes)
    stats_accumulator["original_num_defaults"] += num_defaults

    # 执行欠采样
    default = [(x, 1) for x, y in zip(test_data_X, test_data_Y) if y == 1]
    undefault = [(x, 0) for x, y in zip(test_data_X, test_data_Y) if y == 0]

    if len(default) == 0:
        return np.array([]), np.array([])

    if len(undefault) < split_ratio * len(default):
        # 非违约样本不足，直接保留全部样本
        combined = default + undefault
        print(f"⚠️ Not enough non-default samples to match 1:{split_ratio}, using full dataset instead.")
    else:
        sampled_undefault = undefault[:split_ratio * len(default)]
        combined = default + sampled_undefault

    random.shuffle(combined)  # 打乱顺序


    # 记录最终样本集统计信息
    stats_accumulator["final_sample_size"] += sum([len(x[0]) for x in combined])
    stats_accumulator["final_num_loans"] += len(combined)
    stats_accumulator["final_total_loan_length"] += sum([len(x[0]) for x in combined])
    stats_accumulator["final_num_defaults"] += sum([y for _, y in combined])

    if not combined:
        return np.array([]), np.array([])

    X, Y = zip(*combined)
    return np.array(X, dtype='float32'), np.array(Y, dtype='float32')

# Step 3: 保存数据到本地
def save_data(X, Y, output_dir):
    """
    将生成的测试集保存为 .npy 格式

    参数：
    - X, Y: 测试集的特征和标签
    - output_dir: 保存的文件夹路径
    """
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'testX.npy'), X)
    np.save(os.path.join(output_dir, 'testY.npy'), Y)
    print(f"✅ Saved to {output_dir} | Samples: {len(Y)}, Defaults: {int(Y.sum())}, Non-defaults: {len(Y) - int(Y.sum())}")

# 主流程控制（批量生成不同参数组合的数据）
if __name__ == "__main__":
    # 原始数据来源
    input_files = [
        "data/processed_data/historical_data_time_2019Q1.csv",
        "data/processed_data/historical_data_time_2019Q2.csv",
        "data/processed_data/historical_data_time_2019Q3.csv",
        "data/processed_data/historical_data_time_2019Q4.csv"
    ]

    # 设置不同的主数据长度和采样比例
    master_data_sizes = [12,18]
    split_ratios = [1, 2, 5, 10, 25]

    # 汇总统计信息保存路径
    stats_csv_path = "test_data_summary.csv"
    write_header = not os.path.exists(stats_csv_path)

    # 读取年份（假设全部是同一年）
    year_str = os.path.basename(input_files[0]).split('_')[-1][:4]

    # 打开 CSV 写入文件
    with open(stats_csv_path, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow([
                "year", "master_data_size", "split_ratio",
                "original_sample_size", "original_num_loans", "avg_loan_length", "num_defaults", "default_rate",
                "final_sample_size", "final_num_loans", "final_avg_loan_length", "final_num_defaults", "final_default_rate"
            ])
            csvfile.flush()  # 写入表头后立即保存

        # 枚举不同参数组合
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

                # 遍历每个季度数据
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

                # 合并所有季度的数据
                if all_X and all_Y:
                    final_X = np.concatenate(all_X, axis=0)
                    final_Y = np.concatenate(all_Y, axis=0)

                    # 按参数命名保存路径
                    output_dir = f"data/test_merged_{year_str}_master{master_data_size}_ratio1to{split_ratio}"
                    save_data(final_X, final_Y, output_dir)

                    # 计算统计量
                    avg_len = stats['original_total_loan_length'] / stats['original_num_loans'] if stats['original_num_loans'] else 0
                    final_avg_len = stats['final_total_loan_length'] / stats['final_num_loans'] if stats['final_num_loans'] else 0
                    default_rate = stats['original_num_defaults'] / stats['original_num_loans'] if stats['original_num_loans'] else 0
                    final_default_rate = stats['final_num_defaults'] / stats['final_num_loans'] if stats['final_num_loans'] else 0

                    # 写入统计信息到 CSV
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
                    csvfile.flush()
                    print(f"📈 Stats recorded: {stats['final_num_loans']} loans | {stats['final_num_defaults']} defaults")
                else:
                    print("⚠️ No valid samples generated. Skipping.")
