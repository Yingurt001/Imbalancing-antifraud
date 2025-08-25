#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Model,load_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Masking, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from keras.layers import MultiHeadAttention, LayerNormalization, Add, GlobalAveragePooling1D
from concurrent.futures import ThreadPoolExecutor
from keras.callbacks import ModelCheckpoint
import seaborn as sns
import os
import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import GRU, Dense, Dropout, Input, Masking, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

# 构建 GRU 模型
def masked_hybrid_lstm_transformer_model(input_shape, num_units=256, num_heads=8, ff_dim=1024, rate=0.4):
    inputs = Input(shape=input_shape)
    
    # 添加 Masking 层
    masked_input = Masking(mask_value=0.0)(inputs)
    
    # LSTM 层
    lstm_out = LSTM(num_units, return_sequences=True)(masked_input)
    
    # Transformer 位置编码
    def positional_encoding(position, d_model):
        def get_angles(pos, i, d_model):
            angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
            return pos * angle_rates

        angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)

    # 计算位置编码
    pos_enc = positional_encoding(input_shape[0], num_units)
    pos_enc = tf.tile(pos_enc, [tf.shape(inputs)[0], 1, 1])  # 调整位置编码大小

    # 将位置编码添加到 LSTM 输出中
    combined = Add()([lstm_out, pos_enc])

    # 多头注意力机制
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=num_units)(combined, combined)
    attn_output = Dropout(rate)(attn_output)
    attn_out_norm = LayerNormalization(epsilon=1e-6)(attn_output + combined)

    # 前馈神经网络
    ffn_output = Dense(ff_dim, activation='relu')(attn_out_norm)
    ffn_output = Dense(num_units)(ffn_output)
    ffn_output = Dropout(rate)(ffn_output)
    ffn_output_norm = LayerNormalization(epsilon=1e-6)(ffn_output + attn_out_norm)

    # Global Average Pooling
    final_output = GlobalAveragePooling1D()(ffn_output_norm)
    final_output = Dense(1, activation='sigmoid')(final_output)

    model = Model(inputs=inputs, outputs=final_output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

# 评估模型性能的函数
def evaluate_model(interval, test_X, test_Y, run):
    try:
        # 加载每次运行中验证集上表现最好的模型
        model_path = f'实验二models/Hybrid_best_model_interval_{interval}_run_{run}.h5'
        model = load_model(model_path)
    except FileNotFoundError:
        print(f"Model for interval {interval}, run {run} not found at {model_path}. Skipping this run.")
        return None
    except ValueError as e:
        print(f"Error loading model: {e}")
        return None

    # 在测试集上评估模型性能
    test_loss, test_acc = model.evaluate(test_X, test_Y, verbose=0)
    print(f'Interval {interval}, Run {run} - Test Accuracy: {test_acc:.3f}')

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

    print(f'Interval {interval}, Run {run} - Accuracy: {accuracy:.3f}')
    print(f'Interval {interval}, Run {run} - Precision: {precision:.3f}')
    print(f'Interval {interval}, Run {run} - Recall: {recall:.3f}')
    print(f'Interval {interval}, Run {run} - F1 Score: {f1:.3f}')
    
    return {
        'interval': interval,
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
    csv_file_path = 'E2_Hybrid_model_results.csv'  # 保存结果的CSV文件路径
    intervals = [3, 4, 5, 6, 7, 8, 9]  # 预定义的 interval 值

    for interval in intervals:
        # Step 1: 加载保存好的训练数据
        train_x_path = f'data/npy_test_interval_{interval}/trainX.npy'
        train_y_path = f'data/npy_test_interval_{interval}/trainY.npy'
        test_x_path = f'data/npy_test_interval_{interval}/testX.npy'
        test_y_path = f'data/npy_test_interval_{interval}/testY.npy'

        try:
            train_GRU_X = np.load(train_x_path)
            train_GRU_y = np.load(train_y_path)
            test_X = np.load(test_x_path)
            test_Y = np.load(test_y_path)
        except FileNotFoundError as e:
            print(f"Training or test data for interval {interval} not found: {e}")
            continue

        print(f"train_GRU_X shape: {train_GRU_X.shape}, train_GRU_y shape: {train_GRU_y.shape}")

        # Step 2: 划分训练集和验证集
        train_GRU_X, val_GRU_X, train_GRU_y, val_GRU_y = train_test_split(
            train_GRU_X, train_GRU_y, test_size=0.2, random_state=42, stratify=train_GRU_y)

        # 训练并评估模型 20 次
        for run in range(1, 21):  # 每个 interval 运行 20 次
            print(f"Training interval {interval}, run {run}...")
            try:
                # Step 3: 构建和训练 GRU 模型
                input_shape = (train_GRU_X.shape[1], train_GRU_X.shape[2])
                model = masked_hybrid_lstm_transformer_model(input_shape)
                
                # 定义 ModelCheckpoint 回调函数，保存每次 run 中验证集上性能最好的模型
                model_save_path = f'实验二models/Hybrid_best_model_interval_{interval}_run_{run}.h5'
                checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', verbose=1,
                                             save_best_only=True, mode='max')

                # 训练模型
                model.fit(train_GRU_X, train_GRU_y, epochs=25, batch_size=64, validation_data=(val_GRU_X, val_GRU_y),
                          callbacks=[checkpoint])

                # Step 4: 在测试集上评估模型
                result = evaluate_model(interval, test_X, test_Y, run)
                if result:
                    results.append(result)
                    # 每次评估后保存结果
                    save_results_to_csv(results, csv_file_path)
            except Exception as e:
                print(f"Error during training or evaluation at interval {interval}, run {run}: {e}")
                save_results_to_csv(results, csv_file_path)
                continue  # 继续下一个 run 或 interval

    # 最终输出所有结果并保存
    print("\nFinal Results:")
    for result in results:
        print(result)
    
    # 保存最终结果
    save_results_to_csv(results, csv_file_path)
