import os
import numpy as np
import tensorflow as tf
from keras.models import Model,load_model
from keras.layers import LSTM, Dense, Dropout, Input, Masking, Flatten, Attention
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 定义带注意力的 LSTM 模型
def attention(input_shape, epochs=10, batch_size=64):
    inputs = Input(shape=input_shape)
    masked_input = Masking(mask_value=0.0)(inputs)
    lstm_out = LSTM(128, return_sequences=True)(masked_input)
    lstm_out2 = LSTM(64, return_sequences=True)(lstm_out)
    
    # 添加注意力层
    attention_out = Attention()([lstm_out2, lstm_out2])
    
    # Flatten层
    flattened_out = Flatten()(attention_out)
    
    # 密集层
    dense_out = Dense(25, activation='sigmoid')(flattened_out)
    output = Dense(1, activation='sigmoid')(dense_out)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# 评估模型性能的函数
def evaluate_model(master_data_size, test_X, test_Y, run):
    try:
        # 加载每次运行中验证集上表现最好的模型
        model_path = f'实验三models/attention_best_model_master_data_size_{master_data_size}_run_{run}.h5'
        model = load_model(model_path)
    except FileNotFoundError:
        print(f"Model for master_data_size {master_data_size}, run {run} not found at {model_path}. Skipping this run.")
        return None
    except ValueError as e:
        print(f"Error loading model: {e}")
        return None

    # 在测试集上评估模型性能
    test_loss, test_acc = model.evaluate(test_X, test_Y, verbose=0)
    print(f'master_data_size {master_data_size}, Run {run} - Test Accuracy: {test_acc:.3f}')

    # 预测概率和分类标签
    yhat_probs = model.predict(test_X, verbose=0)
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = (yhat_probs > 0.5).astype(int)

    # 计算并打印性能指标
    accuracy = accuracy_score(test_Y, yhat_classes)
    precision = precision_score(test_Y, yhat_classes)
    recall = recall_score(test_Y, yhat_classes)
    f1 = f1_score(test_Y, yhat_classes)
    conf_matrix = confusion_matrix(test_Y, yhat_classes)

    print(f'master_data_size {master_data_size}, Run {run} - Accuracy: {accuracy:.3f}')
    print(f'master_data_size {master_data_size}, Run {run} - Precision: {precision:.3f}')
    print(f'master_data_size {master_data_size}, Run {run} - Recall: {recall:.3f}')
    print(f'master_data_size {master_data_size}, Run {run} - F1 Score: {f1:.3f}')
    return {
        'master_data_size': master_data_size,
        'run': run,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# 保存结果到 CSV
def save_results_to_csv(results, filepath):
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")

# 主函数
if __name__ == "__main__":
    results = []
    csv_file_path = 'E3_attention_model_results.csv'  # 保存结果的CSV文件路径
      # 每个 master_data_size 运行 20 次
    master_data_sizes = [15, 18, 21, 24, 27]  # 预定义的 master_data_size 值

    for master_data_size in master_data_sizes:
        # Step 1: 加载保存好的训练数据
        train_x_path = f'data/npy_test_master_data_size_{master_data_size}/trainX.npy'
        train_y_path = f'data/npy_test_master_data_size_{master_data_size}/trainY.npy'
        test_x_path = f'data/npy_test_master_data_size_{master_data_size}/testX.npy'
        test_y_path = f'data/npy_test_master_data_size_{master_data_size}/testY.npy'

        try:
            train_GRU_X = np.load(train_x_path)
            train_GRU_y = np.load(train_y_path)
            test_X = np.load(test_x_path)
            test_Y = np.load(test_y_path)
        except FileNotFoundError as e:
            print(f"Training or test data for master_data_size {master_data_size} not found: {e}")
            continue

        print(f"train_GRU_X shape: {train_GRU_X.shape}, train_GRU_y shape: {train_GRU_y.shape}")

        # Step 2: 划分训练集和验证集
        train_GRU_X, val_GRU_X, train_GRU_y, val_GRU_y = train_test_split(
            train_GRU_X, train_GRU_y, test_size=0.2, random_state=42, stratify=train_GRU_y)

        # 训练并评估模型 20 次
        for run in range(1,  21):  # 每个 master_data_size 运行 20 次
            print(f"Training master_data_size {master_data_size}, run {run}...")
            try:
                # Step 3: 构建和训练 GRU 模型
                input_shape = (train_GRU_X.shape[1], train_GRU_X.shape[2])
                model = attention(input_shape)
                
                # 定义 ModelCheckpoint 回调函数，保存每次 run 中验证集上性能最好的模型
                model_save_path = f'实验三models/attention_best_model_master_data_size_{master_data_size}_run_{run}.h5'
                checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', verbose=1,
                                             save_best_only=True, mode='max')

                # 训练模型
                model.fit(train_GRU_X, train_GRU_y, epochs=25, batch_size=64, validation_data=(val_GRU_X, val_GRU_y),
                          callbacks=[checkpoint])

                # Step 4: 在测试集上评估模型
                result = evaluate_model(master_data_size, test_X, test_Y, run)
                if result:
                    results.append(result)
                    # 每次评估后保存结果
                    save_results_to_csv(results, csv_file_path)
            except Exception as e:
                print(f"Error during training or evaluation at master_data_size {master_data_size}, run {run}: {e}")
                save_results_to_csv(results, csv_file_path)
                continue  # 继续下一个 run 或 master_data_size

    # 最终输出所有结果并保存
    print("\nFinal Results:")
    for result in results:
        print(result)
    
    # 保存最终结果
    save_results_to_csv(results, csv_file_path)
