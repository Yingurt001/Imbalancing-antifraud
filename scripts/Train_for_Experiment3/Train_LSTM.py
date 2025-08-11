import os
import numpy as np
import tensorflow as tf
from keras.models import Model,load_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Masking, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.sequence import pad_sequences
import seaborn as sns
import pandas as pd
from tqdm.keras import TqdmCallback
# 定义 LSTM 模型
def masked_LSTM_model(input_shape):
    inputs = Input(shape=input_shape)
    masked_input = Masking(mask_value=0.0)(inputs)
    lstm_out = LSTM(128, return_sequences=True)(masked_input)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out2 = LSTM(64, return_sequences=False)(lstm_out)
    flattened_out = Flatten()(lstm_out2)
    dense_out = Dense(25, activation='sigmoid')(flattened_out)
    dropout_out = Dropout(0.3)(dense_out)
    output = Dense(1, activation='sigmoid')(dropout_out)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# 评估模型性能的函数
def evaluate_model_by_method(model_path, test_X, test_Y):
    try:
        model = load_model(model_path, custom_objects={'KANLinear': KANLinear})
    except Exception as e:
        print(f"❌ Failed to load model: {model_path}\n{e}")
        return None
    test_loss, test_acc = model.evaluate(test_X, test_Y, verbose=0)
    yhat_probs = model.predict(test_X, verbose=0)[:, 0]
    yhat_classes = (yhat_probs > 0.5).astype(int)
    accuracy = accuracy_score(test_Y, yhat_classes)
    precision = precision_score(test_Y, yhat_classes)
    recall = recall_score(test_Y, yhat_classes)
    f1 = f1_score(test_Y, yhat_classes)
    auc = roc_auc_score(test_Y, yhat_probs)
    tn, fp, fn, tp = confusion_matrix(test_Y, yhat_classes).ravel()
    return dict(accuracy=accuracy, precision=precision, recall=recall, f1_score=f1, auc=auc, tn=tn, fp=fp, fn=fn, tp=tp)

# ✅ 结果保存

def save_results_to_csv(results, filepath):
    if not results:
        print("⚠️ No results to save.")
        return
    pd.DataFrame(results).to_csv(filepath, index=False)
    print(f"✅ Results saved to {filepath}")



# ✅ 主程序：训练固定，测试多种方法
if __name__ == "__main__":
    results = []
    csv_file_path = 'E3_LSTM_model_results111.csv'
    os.makedirs(os.path.dirname(csv_file_path) or ".", exist_ok=True)

    master_data_sizes = [12,18]
    split_ratios = [1, 2, 5, 10, 25]
    sampling_methods = ['random_undersample','tsmote', 'timegan','base']  # 训练集采样方法
    # sampling_methods = ['base']  # 训练集采样方法
    test_year = 2019  # 固定年份

    for method in sampling_methods:
        for master_data_size in master_data_sizes:
            for split_ratio in split_ratios:
                print(f"\n🔧 Training | method = {method}, master_data_size = {master_data_size}, split_ratio = 1:{split_ratio}")

                train_x_path = f'data/npy_merged_master_data_size_{master_data_size}_{method}/trainX.npy'
                train_y_path = f'data/npy_merged_master_data_size_{master_data_size}_{method}/trainY.npy'

                try:
                    train_X = np.load(train_x_path)
                    train_Y = np.load(train_y_path)
                except Exception as e:
                    print(f"❌ Failed loading training data for method={method}: {e}")
                    continue

                train_X, val_X, train_Y, val_Y = train_test_split(
                    train_X, train_Y, test_size=0.2, random_state=42, stratify=train_Y)

                for run in range(1, 3):
                    try:
                        print(f"🚀 Run {run}")
                        input_shape = (train_X.shape[1], train_X.shape[2])
                        model = masked_LSTM_model(input_shape)

                        model_save_path = f'实验三models/真KAN/LSTM_best_model_{method}_master_data_size_{master_data_size}_ratio1to{split_ratio}_run_{run}.h5'
                        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

                        checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max')

                        model.fit(
                            train_X, train_Y,
                            epochs=25,
                            batch_size=64,
                            validation_data=(val_X, val_Y),
                            callbacks=[checkpoint,TqdmCallback(verbose=1)],
                            verbose=0
                        )

                        test_x_path = f"data/test_merged_{test_year}_master{master_data_size}_ratio1to{split_ratio}/testX.npy"
                        test_y_path = f"data/test_merged_{test_year}_master{master_data_size}_ratio1to{split_ratio}/testY.npy"

                        if not os.path.exists(test_x_path):
                            print(f"⚠️ Missing test data: {test_x_path}")
                            continue

                        test_X = np.load(test_x_path)
                        test_Y = np.load(test_y_path)

                        print(f"🔍 Evaluating {method} | size={master_data_size}, ratio=1:{split_ratio}, run={run}")
                        result = evaluate_model_by_method(model_save_path, test_X, test_Y)

                        if result:
                            result.update(dict(
                                year=test_year,
                                master_data_size=master_data_size,
                                split_ratio=split_ratio,
                                run=run,
                                method=method
                            ))
                            # ✅ 调整列顺序
                            ordered_result = {k: result[k] for k in ['year', 'master_data_size', 'split_ratio', 'run', 'method'] if k in result}
                            for k in result:
                                if k not in ordered_result:
                                    ordered_result[k] = result[k]
                            results.append(ordered_result)
                            save_results_to_csv(results, csv_file_path)

                    except Exception as e:
                        print(f"❌ Error in run {run} | method={method}: {e}")
                        save_results_to_csv(results, csv_file_path)
                        continue
