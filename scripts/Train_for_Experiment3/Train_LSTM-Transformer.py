import tensorflow as tf
ops = tf
import os
import numpy as np
import tensorflow as tf
import keras
from keras import backend
from keras.src import initializers
from keras.src.layers import Layer, Dropout, LayerNormalization
from keras.models import Model,load_model
from keras.layers import GRU, LSTM, Dense, Dropout, BatchNormalization, Input, Masking, Layer,Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix,roc_auc_score
import pandas as pd
from keras.utils import CustomObjectScope
from tqdm.keras import TqdmCallback
import time

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
def evaluate_model(master_data_size, test_X, test_Y, run, split_ratio):
    print(f"Test_X shape: {test_X.shape}, Test_Y shape: {test_Y.shape}")
    model_path = f'实验三models/真KAN/LSTM_Transformer_best_model_master_data_size_{master_data_size}_ratio1to{split_ratio}_run_{run}.h5'
    print(f"🔍 Trying to load model from: {model_path}")
    
    try:
        model = load_model(model_path)
    except FileNotFoundError:
        print(f"❌ Model not found at {model_path}. Skipping run {run}.")
        return None
    except ValueError as e:
        print(f"❌ Error loading model: {e}")
        return None

    # 测试集评估
    test_loss, test_acc = model.evaluate(test_X, test_Y, verbose=0)
    print(f'✅ master_data_size {master_data_size}, Run {run} - Test Accuracy: {test_acc:.3f}')

    # 预测 & 分类
    yhat_probs = model.predict(test_X, verbose=0)[:, 0]
    yhat_classes = (yhat_probs > 0.5).astype(int)

    # 性能指标
    accuracy = accuracy_score(test_Y, yhat_classes)
    precision = precision_score(test_Y, yhat_classes)
    recall = recall_score(test_Y, yhat_classes)
    f1 = f1_score(test_Y, yhat_classes)
    auc = roc_auc_score(test_Y, yhat_probs)

    conf_matrix = confusion_matrix(test_Y, yhat_classes)
    tn, fp, fn, tp = conf_matrix.ravel()
    # 输出所有指标
    print(f'📊 Eval Results - Acc: {accuracy:.3f} | Prec: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}')
    print(f'🧮 Confusion Matrix:\n{conf_matrix}')
    
    return {
        'master_data_size': master_data_size,
        'split_ratio': split_ratio,
        'run': run,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }


# ✅ 安全地保存结果到 CSV（避免空写入）
def save_results_to_csv(results, filepath):
    if not results:
        print("⚠️ No results to save.")
        return
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)
    print(f"✅ Results saved to {filepath}")

# ✅ 主程序
if __name__ == "__main__":
    results = []
    csv_file_path = 'E3_LSTM_Transformer_model_results.csv'
    os.makedirs(os.path.dirname(csv_file_path) or ".", exist_ok=True)

    master_data_sizes = [15, 18, 21, 24, 27]
    split_ratios = [1, 2, 5, 10, 25]

    for master_data_size in master_data_sizes:
        for split_ratio in split_ratios:
            print(f"\n🔧 Processing master_data_size = {master_data_size}, split_ratio = 1:{split_ratio}")

            train_x_path = f'data/npy_merged_master_data_size_{master_data_size}/trainX.npy'
            train_y_path = f'data/npy_merged_master_data_size_{master_data_size}/trainY.npy'
            test_x_path  = f'data/test_merged_2019_master{master_data_size}_ratio1to{split_ratio}/testX.npy'
            test_y_path  = f'data/test_merged_2019_master{master_data_size}_ratio1to{split_ratio}/testY.npy'

            try:
                train_GRU_X = np.load(train_x_path)
                train_GRU_y = np.load(train_y_path)
                test_X = np.load(test_x_path)
                test_Y = np.load(test_y_path)
            except FileNotFoundError as e:
                print(f"❌ Data not found: {e}")
                continue

            print(f"✅ Loaded train shape: {train_GRU_X.shape}, {train_GRU_y.shape}")

            # 训练/验证集划分
            train_GRU_X, val_GRU_X, train_GRU_y, val_GRU_y = train_test_split(
                train_GRU_X, train_GRU_y, test_size=0.2, random_state=42, stratify=train_GRU_y)

            for run in range(1, 21):
                print(f"🚀 Training Run {run} | master_data_size={master_data_size}, split_ratio=1:{split_ratio}")
                try:
                    input_shape = (train_GRU_X.shape[1], train_GRU_X.shape[2])
                    model = masked_hybrid_lstm_transformer_model(input_shape)

                    model_save_path = f'实验三models/真KAN/LSTM_Transformer_best_model_master_data_size_{master_data_size}_ratio1to{split_ratio}_run_{run}.h5'
                    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

                    checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', verbose=0,
                                                 save_best_only=True, mode='max')

                    start_time = time.time()
                    model.fit(
                        train_GRU_X, train_GRU_y,
                        epochs=25,
                        batch_size=64,
                        validation_data=(val_GRU_X, val_GRU_y),
                        callbacks=[checkpoint, TqdmCallback(verbose=1)]
                    )
                    print(f"⏱️ Time for run {run}: {time.time() - start_time:.2f}s")

                    # ✅ 现在正确传入 split_ratio
                    result = evaluate_model(master_data_size, test_X, test_Y, run, split_ratio)
                    if result:
                        print(f"📌 Result: {result}")
                        results.append(result)
                        save_results_to_csv(results, csv_file_path)

                except Exception as e:
                    print(f"❌ Error in run {run}: {e}")
                    save_results_to_csv(results, csv_file_path)
                    continue

    print("\n✅ All Finished. Summary:")
    for result in results:
        print(result)
    save_results_to_csv(results, csv_file_path)
