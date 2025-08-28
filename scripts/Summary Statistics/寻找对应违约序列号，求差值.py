import pandas as pd
import os
import numpy as np
from datetime import datetime

def parse_date(date_val):
    """将日期值转换为年月整数"""
    if pd.isna(date_val) or date_val == 0 or str(date_val) == 'nan' or str(date_val) == '':
        return None
    
    try:
        # 转换为整数并确保是6位数
        date_int = int(float(date_val))
        date_str = f"{date_int:06d}"
        
        # 提取年月
        year = int(date_str[:4])
        month = int(date_str[4:6])
        return year, month
    except:
        return None

def calculate_loan_term(first_payment, maturity):
    """计算贷款期限（以年为单位）"""
    if first_payment is None or maturity is None:
        return None
    
    first_year, first_month = first_payment
    maturity_year, maturity_month = maturity
    
    # 计算总月份差
    total_months = (maturity_year - first_year) * 12 + (maturity_month - first_month)
    
    # 转换为年（四舍五入到最接近的整数）
    return round(total_months / 12)

def process_default_loans(default_loans_file, data_files, output_file):
    """处理违约贷款数据"""
    # 读取违约贷款序列号
    default_loans_df = pd.read_csv(default_loans_file)
    default_loan_sequences = set(default_loans_df['LOAN SEQUENCE NUMBER'].astype(str).str.strip())
    
    print(f"找到 {len(default_loan_sequences)} 个违约贷款序列号")
    print(f"前10个违约贷款序列号: {list(default_loan_sequences)[:10]}")
    
    # 收集所有违约贷款的数据
    all_default_loans_data = []
    
    for data_file in data_files:
        print(f"\n处理文件: {data_file}")
        
        # 检查文件是否存在
        if not os.path.exists(data_file):
            print(f"❌ 文件不存在: {data_file}")
            continue
        
        try:
            # 读取数据文件，不预设列名
            df = pd.read_csv(data_file, sep='|', header=None, low_memory=False)
            print(f"成功读取，行数: {len(df)}，列数: {len(df.columns)}")
            
            # 明确指定列位置（基于原始数据格式）
            loan_seq_col = 19  # 贷款序列号在第19列（从0开始计数）
            first_payment_col = 1  # 首次还款日期在第1列
            maturity_col = 3  # 到期日期在第3列
            
            print(f"贷款序列号列: {loan_seq_col}")
            print(f"首次还款日期列: {first_payment_col}")
            print(f"到期日期列: {maturity_col}")
            
            # 筛选违约贷款
            df[loan_seq_col] = df[loan_seq_col].astype(str).str.strip()
            default_loans_df = df[df[loan_seq_col].isin(default_loan_sequences)]
            
            print(f"匹配到的违约贷款行数: {len(default_loans_df)}")
            
            # 提取需要的列
            if not default_loans_df.empty:
                for _, row in default_loans_df.iterrows():
                    first_payment = parse_date(row[first_payment_col])
                    maturity = parse_date(row[maturity_col])
                    loan_term = calculate_loan_term(first_payment, maturity)
                    
                    all_default_loans_data.append({
                        'LOAN SEQUENCE NUMBER': row[loan_seq_col],
                        'FIRST PAYMENT DATE': row[first_payment_col],
                        'MATURITY DATE': row[maturity_col],
                        'LOAN_TERM_YEARS': loan_term,
                        'SOURCE_FILE': os.path.basename(data_file)
                    })
                    
                    # 打印前几条记录以验证
                    if len(all_default_loans_data) <= 5:
                        print(f"  示例记录: {row[loan_seq_col]}, "
                              f"首次还款: {row[first_payment_col]}, "
                              f"到期: {row[maturity_col]}, "
                              f"期限: {loan_term}年")
        except Exception as e:
            print(f"❌ 处理文件时出错 {data_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # 创建结果DataFrame
    result_df = pd.DataFrame(all_default_loans_data)
    
    if not result_df.empty:
        # 去除重复的贷款序列号（保留每个贷款的第一条记录）
        result_df = result_df.drop_duplicates(subset=['LOAN SEQUENCE NUMBER'], keep='first')
        
        # 保存结果
        result_df.to_csv(output_file, index=False)
        print(f"\n结果已保存到: {output_file}")
        print(f"共处理 {len(result_df)} 个违约贷款")
        
        # 显示一些统计信息
        print("\n贷款期限统计:")
        print(result_df['LOAN_TERM_YEARS'].describe())
        
        # 显示前几条记录
        print("\n前10条记录:")
        print(result_df[['LOAN SEQUENCE NUMBER', 'FIRST PAYMENT DATE', 'MATURITY DATE', 'LOAN_TERM_YEARS']].head(10))
    else:
        print("\n❌ 没有找到任何匹配的违约贷款记录")
    
    return result_df

if __name__ == "__main__":
    # 设置文件路径
    default_loans_file = "default_loans/default_loans_2022.csv"  # 修改为您的实际路径
    data_files = [
        "/Users/zhangying/Documents/Imbalance data&Financial Fraud/data/processed_data/historical_data_2022Q1.txt",
        "/Users/zhangying/Documents/Imbalance data&Financial Fraud/data/processed_data/historical_data_2022Q2.txt",
        "/Users/zhangying/Documents/Imbalance data&Financial Fraud/data/processed_data/historical_data_2022Q3.txt",
        "/Users/zhangying/Documents/Imbalance data&Financial Fraud/data/processed_data/historical_data_2022Q4.txt"
    ]
    output_file = "2022_default_loans_with_terms.csv"
    
    # 处理数据
    result_df = process_default_loans(default_loans_file, data_files, output_file)