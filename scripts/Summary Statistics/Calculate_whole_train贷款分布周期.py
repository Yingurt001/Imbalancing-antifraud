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
# Step 2: Collect statistics (违约客户的原始贷款周期，以月为单位)
# =====================================
def collect_statistics(grouped_data):
    default_loan_terms = []

    for _, group in grouped_data:
        # 判断是否违约：使用预处理生成的 CURRENT LOAN DELINQUENCY STATUS_1 列
        if "CURRENT LOAN DELINQUENCY STATUS_1" not in group.columns:
            raise ValueError("预处理后的 CSV 缺少列: CURRENT LOAN DELINQUENCY STATUS_1")

        if group["CURRENT LOAN DELINQUENCY STATUS_1"].sum() > 0:
            # 初始贷款周期：取最早一期（即该贷款的第一行）的剩余月数
            orig_term = group["REMAINING MONTHS TO LEGAL MATURITY"].iloc[0]
            default_loan_terms.append(orig_term)

    return default_loan_terms


# =====================================
# Step 3: Main loop (2019–2022 每年统计)
# =====================================
if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)

    # all_years = ["2019", "2020", "2021", "2022"]
    # all_years = [ "2020"]
    all_years = [ "2018"]

    stats_csv_path = "2018_LoanTerm_Distribution_Defaults.csv"

    with open(stats_csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        # 写表头
        writer.writerow(["year", "LoanTerm_Months", "Count"])

        for year in all_years:
            input_files = [
                f"data/processed_data/historical_data_time_{year}Q1.csv",
                f"data/processed_data/historical_data_time_{year}Q2.csv",
                f"data/processed_data/historical_data_time_{year}Q3.csv",
                f"data/processed_data/historical_data_time_{year}Q4.csv",
            ]

            year_default_terms = []

            for input_file in input_files:
                if not os.path.exists(input_file):
                    print(f"❌ File not found: {input_file}")
                    continue

                grouped_data, _ = load_and_process_data(
                    input_file, fraction=FRACTION, seed=SEED
                )
                default_terms = collect_statistics(grouped_data)
                year_default_terms.extend(default_terms)

            # 统计分布（以月为单位）
            if year_default_terms:
                term_counts = pd.Series(year_default_terms).value_counts().sort_index()
                for term, count in term_counts.items():
                    writer.writerow([year, term, count])

            print(f"✅ {year} finished, {len(year_default_terms)} default loans processed.")

    print(f"📊 Default loan term distribution saved to {stats_csv_path}")
