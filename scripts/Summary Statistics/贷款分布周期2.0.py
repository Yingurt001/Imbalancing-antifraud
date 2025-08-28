import os, sys
import numpy as np
import pandas as pd
import csv
import random

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
        sampled_loans = np.random.choice(
            unique_loans, 
            size=int(len(unique_loans) * fraction), 
            replace=False
        )
        df = df[df['LOAN SEQUENCE NUMBER'].isin(sampled_loans)]

    sorted_data = df.sort_values(['LOAN SEQUENCE NUMBER'])
    grouped_data = sorted_data.groupby('LOAN SEQUENCE NUMBER')
    return grouped_data, df


# =====================================
# Step 2: Collect default loan sequence numbers
# =====================================
def collect_default_loans(grouped_data):
    default_loan_seq = []

    for loan_seq, group in grouped_data:
        # 判断是否违约：使用预处理生成的 CURRENT LOAN DELINQUENCY STATUS_1 列
        if "CURRENT LOAN DELINQUENCY STATUS_1" not in group.columns:
            raise ValueError("预处理后的 CSV 缺少列: CURRENT LOAN DELINQUENCY STATUS_1")

        if group["CURRENT LOAN DELINQUENCY STATUS_1"].sum() > 0:
            default_loan_seq.append(loan_seq)

    return default_loan_seq


# =====================================
# Step 3: Main loop (处理指定年份的数据)
# =====================================
if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)

    # all_years = ["2018", "2019", "2020", "2021", "2022"]  # 可以修改为需要处理的年份
    all_years = ["2022"]
    # 创建输出目录
    output_dir = "default_loans"
    os.makedirs(output_dir, exist_ok=True)

    for year in all_years:
        input_files = [
            f"data/processed_data/historical_data_time_{year}Q1.csv",
            f"data/processed_data/historical_data_time_{year}Q2.csv",
            f"data/processed_data/historical_data_time_{year}Q3.csv",
            f"data/processed_data/historical_data_time_{year}Q4.csv",
        ]

        year_default_loans = set()  # 使用集合避免重复

        for input_file in input_files:
            if not os.path.exists(input_file):
                print(f"❌ File not found: {input_file}")
                continue

            try:
                grouped_data, _ = load_and_process_data(
                    input_file, fraction=FRACTION, seed=SEED
                )
                default_loans = collect_default_loans(grouped_data)
                year_default_loans.update(default_loans)  # 添加到集合中
                print(f"✅ Processed {input_file}, found {len(default_loans)} default loans")
            except Exception as e:
                print(f"❌ Error processing {input_file}: {e}")

        # 保存违约贷款序列号到CSV文件
        if year_default_loans:
            output_file = os.path.join(output_dir, f"default_loans_{year}.csv")
            with open(output_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["LOAN SEQUENCE NUMBER"])
                for loan_seq in sorted(year_default_loans):
                    writer.writerow([loan_seq])
            
            print(f"✅ {year} finished, {len(year_default_loans)} default loans saved to {output_file}")
        else:
            print(f"⚠️  {year} finished, no default loans found")

    print("📊 Default loan sequence numbers extraction completed.")