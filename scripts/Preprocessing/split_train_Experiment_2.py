#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor

# Step 1: 读取数据并排序
def load_and_process_data(input_file):
    # 读取预处理后的数据
    encoded_MonthlyData = pd.read_csv(input_file)
    
    # 按照 'LOAN SEQUENCE NUMBER' 进行排序
    sorted_data = encoded_MonthlyData.sort_values('LOAN SEQUENCE NUMBER')
    
    # 根据 'LOAN SEQUENCE NUMBER' 进行分组
    grouped_data = sorted_data.groupby('LOAN SEQUENCE NUMBER')
    
    return grouped_data

# Step 2: 初始化变量并构建测试集和标签
def build_train_data(grouped_data, interval):
    train_data_X = []
    train_data_Y = []
    MASTER_DATA_SIZE = 24
    predict_month = 3
    SPLIT = 3  # 控制未违约样本的比例

    # 遍历每个贷款的数据组
    for name, group in grouped_data:
        # 根据 'REMAINING MONTHS TO LEGAL MATURITY' 进行降序排序
        group = group.sort_values('REMAINING MONTHS TO LEGAL MATURITY', ascending=False)
        
        # 只取长度>=40的贷款组
        group_size = len(group)
        if group_size >= MASTER_DATA_SIZE:
            # 提取特征(X)和标签(Y)
            train_X = group.iloc[0:MASTER_DATA_SIZE-interval-predict_month-1, 1:-1]  # 选择相关特征列
            train_Y = group.iloc[MASTER_DATA_SIZE-predict_month-1:MASTER_DATA_SIZE-1, -1]  # 选择标签列
            
            # 删除不需要的列
            if 'ZERO BALANCE CODE' in train_X.columns and 'REMAINING MONTHS TO LEGAL MATURITY' in train_X.columns:
                train_X = train_X.drop(['ZERO BALANCE CODE', 'REMAINING MONTHS TO LEGAL MATURITY'], axis=1)
                
            # 检查train_Y中的值并转换为二元标签
            binary_label = 1 if train_Y.sum() > 0 else 0
            
            # 将数据添加到测试集
            train_data_X.append(train_X.values)
            train_data_Y.append(binary_label)
    
    # 平衡数据集
    default = []
    undefault = []
    for i in range(len(train_data_Y)):
        if train_data_Y[i] == 1:
            default.append((train_data_X[i], 1))
        else:
            undefault.append((train_data_X[i], 0))
    
    # 控制未违约样本的比例
    combined_list = default + undefault[:SPLIT * len(default)]
    random.shuffle(combined_list)

    train_X = []
    train_Y = []
    for X, Y in combined_list:
        train_X.append(X)
        train_Y.append(Y)
    
    # 转换为 NumPy 数组
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    
    return train_X, train_Y

# Step 3: 数据保存
def save_data(train_X, train_Y, output_dir='data/npy_train'):
    # 转换为 float32 类型
    train_X = train_X.astype('float32')
    train_Y = train_Y.astype('float32')
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存处理后的数据
    np.save(os.path.join(output_dir, 'trainX.npy'), train_X)
    np.save(os.path.join(output_dir, 'trainY.npy'), train_Y)
    print(f"train data and labels have been saved to '{output_dir}/trainX.npy' and '{output_dir}/trainY.npy'.")

# 多线程处理函数
def process_interval(interval):
    # 加载并处理数据
    grouped_data = load_and_process_data(input_file)
    
    # 构建测试数据
    train_X, train_Y = build_train_data(grouped_data, interval)
    
    # 保存数据到不同的目录
    output_dir = f'data/npy_test_interval_{interval}'
    save_data(train_X, train_Y, output_dir)

if __name__ == "__main__":
    # 设置输入文件路径
    input_file = "data/processed_data/historical_data_time_2019Q1.csv"
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"Error: The file '{input_file}' does not exist. Please check the path and try again.")
    else:
        # 定义不同的 interval 值
        intervals = [4,5, 7,8, 9]  # 例如，使用三个不同的 interval 进行测试

        # 使用 ThreadPoolExecutor 来并行处理不同的 interval
        with ThreadPoolExecutor(max_workers=len(intervals)) as executor:
            executor.map(process_interval, intervals)
