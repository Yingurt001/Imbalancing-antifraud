import os
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Masking, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from keras.layers import MultiHeadAttention, LayerNormalization, Add, GlobalAveragePooling1D
from concurrent.futures import ThreadPoolExecutor
import seaborn as sns
import pandas as pd

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
def evaluate_model(interval, model, test_X, test_Y):
    test_loss, test_acc = model.evaluate(test_X, test_Y, verbose=0)
    yhat_probs = model.predict(test_X, verbose=0)
    yhat_classes = (yhat_probs[:, 0] > 0.5).astype(int)

    accuracy = accuracy_score(test_Y, yhat_classes)
    precision = precision_score(test_Y, yhat_classes)
    recall = recall_score(test_Y, yhat_classes)
    f1 = f1_score(test_Y, yhat_classes)

    print(f'Interval {interval} - Test Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}')

    return {
        'interval': interval,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# 主函数
if __name__ == "__main__":
    results = []
    intervals = [500000, 1000000, 1500000, 2000000, 3000000, 5000000]

    for interval in intervals:
        print(f"Processing interval {interval}...")

        # Step 1: 加载训练数据
        try:
            train_GRU_X = np.load(f'data/npy_test_nrows_{interval}/trainX.npy')
            train_GRU_y = np.load(f'data/npy_test_nrows_{interval}/trainY.npy')
            test_X = np.load(f'data/npy_test_nrows_{interval}/testX.npy')
            test_Y = np.load(f'data/npy_test_nrows_{interval}/testY.npy')
        except FileNotFoundError:
            print(f"Training or test data for interval {interval} not found.")
            continue

        print(f"train_GRU_X shape: {train_GRU_X.shape}, train_GRU_y shape: {train_GRU_y.shape}")

        # Step 2: 划分训练集和验证集
        train_GRU_X, val_GRU_X, train_GRU_y, val_GRU_y = train_test_split(
            train_GRU_X, train_GRU_y, test_size=0.2, random_state=42, stratify=train_GRU_y)

        # 保存划分后的数据
        np.save(f'data/npy/valX_interval_{interval}.npy', val_GRU_X)
        np.save(f'data/npy/valY_interval_{interval}.npy', val_GRU_y)
        np.save(f'data/npy/trainX_new_interval_{interval}.npy', train_GRU_X)
        np.save(f'data/npy/trainY_new_interval_{interval}.npy', train_GRU_y)

        # 训练和评估每个interval 10次
        for run in range(10):
            print(f"Training interval {interval}, run {run+1}/10...")
            # Step 4: 构建和训练 GRU 模型
            input_shape = (train_GRU_X.shape[1], train_GRU_X.shape[2])
            model = masked_hybrid_lstm_transformer_model(input_shape)
            # 训练模型
            history = model.fit(train_GRU_X, train_GRU_y, epochs=20, batch_size=64, validation_data=(val_GRU_X, val_GRU_y), verbose=0)
            # 评估模型在训练集和验证集上的性能
            train_loss, train_acc = model.evaluate(train_GRU_X, train_GRU_y, verbose=0)
            val_loss, val_acc = model.evaluate(val_GRU_X, val_GRU_y, verbose=0)
            print(f'Run {run+1}: Train Accuracy: {train_acc:.3f}, Validation Accuracy: {val_acc:.3f}')

            # Step 5: 在测试集上评估模型
            result = evaluate_model(interval, model, test_X, test_Y)
            if result:
                result['run'] = run + 1  # 记录第几次运行
                results.append(result)

    # Step 6: 保存测试结果到CSV文件
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'/Users/zhangying/Documents/Imbalanced_Atten_LSTM/Results/Hybrid_results.csv', index=False)
        print("All results saved to 'Hybrid_results.csv'.")













