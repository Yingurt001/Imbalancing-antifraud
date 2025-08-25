import os
import pandas as pd

def integrate_quarterly_data(file_paths, output_path):
    # 定义列名
    HistoryHeader = "LOAN SEQUENCE NUMBER,MONTHLY REPORTING PERIOD,CURRENT ACTUAL UPB," \
                    "CURRENT LOAN DELINQUENCY STATUS,LOAN AGE,REMAINING MONTHS TO LEGAL MATURITY," \
                    "DEFECT SETTLEMENT DATE,MODIFICATION FLAG,ZERO BALANCE CODE," \
                    "ZERO BALANCE EFFECTIVE DATE,CURRENT INTEREST RATE,CURRENT DEFERRED UPB," \
                    "DUE DATE OF LAST PAID INSTALLMENT (DDLPI),MI RECOVERIES,NET SALE PROCEEDS," \
                    "NON MI RECOVERIES,EXPENSES,LEGAL COSTS,MAINTENANCE AND PRESERVATION COSTS," \
                    "TAXES AND INSURANCE,MISCELLANEOUS EXPENSES,ACTUAL LOSS CALCULATION," \
                    "MODIFICATION COST,STEP MODIFICATION FLAG,DEFERRED PAYMENT PLAN,ESTIMATED LOAN TO VALUE (ELTV)," \
                    "ZERO BALANCE REMOVAL UPB,DELINQUENT ACCRUED INTEREST,DELINQUENCY DUE TO DISASTER," \
                    "BORROWER ASSISTANCE STATUS CODE,CURRENT MONTH MODIFICATION COST," \
                    "INTEREST BEARING UPB"

    column_names = HistoryHeader.split(",")

    all_data = []

    for file_path in file_paths:
        df = pd.read_csv(file_path, sep='|', header=None, names=column_names, index_col=False)
        df.drop(df.index[0], inplace=True)  # 删除原始可能存在的多余首行
        all_data.append(df)

    annual_data = pd.concat(all_data, ignore_index=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存为无表头、| 分隔的 txt 格式，兼容原始数据格式
    annual_data.to_csv(output_path, sep='|', index=False, header=False)

    print(f"Data integration complete. Output saved to: {output_path}")

if __name__ == "__main__":
    base_path = "/Users/zhangying/Documents/Imbalance data&Financial Fraud/Dataset/historical_data"
    files = [
        os.path.join(base_path, "historical_data_time_2019Q1.txt"),
        os.path.join(base_path, "historical_data_time_2019Q2.txt"),
        os.path.join(base_path, "historical_data_time_2019Q3.txt"),
        os.path.join(base_path, "historical_data_time_2019Q4.txt"),
    ]

    output_file = os.path.join(base_path, "historical_data_time_2019.txt")
    integrate_quarterly_data(files, output_file)
