import os
import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Masking, Flatten
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import pandas as pd
from tqdm.keras import TqdmCallback

# 固定参数
master_data_size_ori = 21
blank_intervals = list(range(1, 13))

# ========================
# LSTM 模型
# ========================
def masked_LSTM_model(input_shape):
    inputs = Input(shape=input_shape)
    masked_input = Masking(mask_value=0.0)(inputs)
    lstm_out = LSTM(128, return_sequences=True)(masked_input)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out2 = LSTM(64, return_sequences=False)(lstm_out)

    dense_out = Dense(25, activation='relu')(lstm_out2)
    dropout_out = Dropout(0.3)(dense_out)
    output = Dense(1, activation='sigmoid')(dropout_out)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# ========================
# 模型评估
# ========================
def evaluate_model_by_method(model_path, test_X, test_Y):
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {model_path}\n{e}")
        return None

    yhat_probs = model.predict(test_X, verbose=0)[:, 0]
    yhat_classes = (yhat_probs > 0.5).astype(int)

    return dict(
        accuracy=accuracy_score(test_Y, yhat_classes),
        precision=precision_score(test_Y, yhat_classes),
        recall=recall_score(test_Y, yhat_classes),
        f1_score=f1_score(test_Y, yhat_classes),
        auc=roc_auc_score(test_Y, yhat_probs),
        tn=confusion_matrix(test_Y, yhat_classes).ravel()[0],
        fp=confusion_matrix(test_Y, yhat_classes).ravel()[1],
        fn=confusion_matrix(test_Y, yhat_classes).ravel()[2],
        tp=confusion_matrix(test_Y, yhat_classes).ravel()[3],
    )

# ========================
# 主程序
# ========================
if __name__ == "__main__":
    results = []
    csv_file_path = 'E4_LSTM_model_results.csv'
    os.makedirs(os.path.dirname(csv_file_path) or ".", exist_ok=True)

    train_year = 2019
    test_year = 2020
    split_ratios = [1, "original"]
    sampling_methods = ['random_undersample', 'tsmote', 'timegan', 'base']

    for method in sampling_methods:
        for blank_interval in blank_intervals:
            master_data_size_final = master_data_size_ori - blank_interval

            for split_ratio in split_ratios:
                print(f"\n🔧 Training | method={method}, blank_interval={blank_interval}, "
                      f"final_size={master_data_size_final}, split_ratio={split_ratio}")

                train_x_path = f"data/npy_merged_{train_year}_blank{blank_interval}_{method}/trainX.npy"
                train_y_path = f"data/npy_merged_{train_year}_blank{blank_interval}_{method}/trainY.npy"

                if not os.path.exists(train_x_path):
                    print(f"❌ Missing train data: {train_x_path}")
                    continue

                train_X = np.load(train_x_path)
                train_Y = np.load(train_y_path)
                train_X, val_X, train_Y, val_Y = train_test_split(
                    train_X, train_Y, test_size=0.2, random_state=42, stratify=train_Y)

                for run in range(1, 3):
                    try:
                        print(f"🚀 Run {run}")
                        input_shape = (train_X.shape[1], train_X.shape[2])
                        model = masked_LSTM_model(input_shape)

                        ratio_str = "ratio_original" if split_ratio == "original" else f"ratio1to{split_ratio}"
                        model_save_path = f"models/LSTM_best_model_{method}_blank{blank_interval}_{ratio_str}_run{run}.keras"
                        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

                        checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy',
                                                     save_best_only=True, mode='max')

                        model.fit(
                            train_X, train_Y,
                            epochs=25,
                            batch_size=64,
                            validation_data=(val_X, val_Y),
                            callbacks=[checkpoint, TqdmCallback(verbose=1)],
                            verbose=0
                        )

                        # 测试集路径
                        if split_ratio == "original":
                            test_x_path = f"data/test_merged_{test_year}_blank{blank_interval}_ratio_original/testX.npy"
                            test_y_path = f"data/test_merged_{test_year}_blank{blank_interval}_ratio_original/testY.npy"
                        else:
                            test_x_path = f"data/test_merged_{test_year}_blank{blank_interval}_ratio1to{split_ratio}/testX.npy"
                            test_y_path = f"data/test_merged_{test_year}_blank{blank_interval}_ratio1to{split_ratio}/testY.npy"

                        if not os.path.exists(test_x_path):
                            print(f"⚠️ Missing test data: {test_x_path}")
                            continue

                        test_X = np.load(test_x_path)
                        test_Y = np.load(test_y_path)

                        print(f"🔍 Evaluating {method} | blank={blank_interval}, split_ratio={split_ratio}, run={run}")
                        result = evaluate_model_by_method(model_save_path, test_X, test_Y)

                        if result:
                            result.update(dict(
                                train_year=train_year,
                                test_year=test_year,
                                blank_interval=blank_interval,
                                master_data_size_final=master_data_size_final,
                                split_ratio=split_ratio,
                                run=run,
                                method=method
                            ))
                            results.append(result)
                            pd.DataFrame(results).to_csv(csv_file_path, index=False)

                    except Exception as e:
                        print(f"❌ Error in run {run} | method={method}: {e}")
                        pd.DataFrame(results).to_csv(csv_file_path, index=False)
                        continue

    print("\n✅ All Finished. Summary saved.")
