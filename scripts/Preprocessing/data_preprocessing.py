#!/opt/anaconda3/envs/pytorch2.2/bin/python

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(input_path, output_dir):
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

    try:
        df = pd.read_csv(input_path, sep='|', header=None, names=HistoryHeader.split(","), index_col=False, low_memory=False)
        df.drop(df.index[0], inplace=True)
        df['CURRENT LOAN DELINQUENCY STATUS'] = df['CURRENT LOAN DELINQUENCY STATUS'].replace('RA', 100)
        df['CURRENT LOAN DELINQUENCY STATUS'] = pd.to_numeric(df['CURRENT LOAN DELINQUENCY STATUS'], errors='coerce')

        saved = df[['LOAN SEQUENCE NUMBER','CURRENT LOAN DELINQUENCY STATUS','REMAINING MONTHS TO LEGAL MATURITY',
                    'CURRENT INTEREST RATE','INTEREST BEARING UPB','BORROWER ASSISTANCE STATUS CODE',
                    'CURRENT DEFERRED UPB','ESTIMATED LOAN TO VALUE (ELTV)','CURRENT ACTUAL UPB']]

        saved['BORROWER ASSISTANCE STATUS CODE'].fillna('Unknown', inplace=True)
        encoded = pd.get_dummies(saved['BORROWER ASSISTANCE STATUS CODE'], prefix='ASSISTANCE STATUS').astype(int)
        saved = pd.concat([saved, encoded], axis=1)
        saved.drop(['BORROWER ASSISTANCE STATUS CODE'], axis=1, inplace=True)

        saved['INTEREST BEARING UPB-Delta'] = saved.groupby('LOAN SEQUENCE NUMBER')['INTEREST BEARING UPB'].diff().fillna(0).abs()
        saved.drop(['INTEREST BEARING UPB'], axis=1, inplace=True)

        saved['CURRENT LOAN DELINQUENCY STATUS'] = saved['CURRENT LOAN DELINQUENCY STATUS'].replace([0,1,2], 0)
        saved['CURRENT LOAN DELINQUENCY STATUS'] = saved['CURRENT LOAN DELINQUENCY STATUS'].apply(lambda x: 1 if x != 0 else 0)

        saved['ESTIMATED LOAN TO VALUE (ELTV)'] = saved['ESTIMATED LOAN TO VALUE (ELTV)'].fillna(0)
        saved = saved[saved['ESTIMATED LOAN TO VALUE (ELTV)'] != 999]

        final = pd.get_dummies(saved, columns=['CURRENT LOAN DELINQUENCY STATUS'], dtype=int)

        scaler = MinMaxScaler()
        cols_to_scale = [col for col in final.columns if col not in ('LOAN SEQUENCE NUMBER', 'REMAINING MONTHS TO LEGAL MATURITY')]
        final[cols_to_scale] = scaler.fit_transform(final[cols_to_scale])

        os.makedirs(output_dir, exist_ok=True)
        file_name = os.path.basename(input_path).replace(".txt", ".csv")
        output_path = os.path.join(output_dir, file_name)

        final.to_csv(output_path, index=False)
        print(f"✅ Processed: {file_name}")
    except Exception as e:
        print(f"❌ Error processing {input_path}: {e}")

def batch_preprocess_custom_files(file_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file_path in file_list:
        preprocess_data(file_path, output_dir)

if __name__ == "__main__":
    # 指定要处理的多个文件路径：
    file_list = [
        "/Users/zhangying/Documents/Imbalance data&Financial Fraud/Dataset/historical_data/historical_data_time_2022Q1.txt",
        "/Users/zhangying/Documents/Imbalance data&Financial Fraud/Dataset/historical_data/historical_data_time_2022Q2.txt",
        "/Users/zhangying/Documents/Imbalance data&Financial Fraud/Dataset/historical_data/historical_data_time_2022Q3.txt",
        "/Users/zhangying/Documents/Imbalance data&Financial Fraud/Dataset/historical_data/historical_data_time_2022Q4.txt"
    ]

    output_dir = "/Users/zhangying/Documents/Imbalance data&Financial Fraud/data/processed_data"
    
    batch_preprocess_custom_files(file_list, output_dir)
