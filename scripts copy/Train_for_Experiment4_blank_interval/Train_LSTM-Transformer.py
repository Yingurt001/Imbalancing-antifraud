import tensorflow as tf
ops = tf
import os
import numpy as np
import keras
from keras.models import Model, load_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Masking, Add, LayerNormalization, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import pandas as pd
from tqdm.keras import TqdmCallback

# 固定参数
master_data_size_ori = 21
blank_intervals = list(range(1, 13))


# ========================
# Hybrid LSTM + Transformer 模型
# ========================
def masked_hybrid_lstm_transformer_model(input_shape, num_units=128, num_heads=4, ff_dim=256, rate=0.3):
    inputs = Input(shape=input_shape)

    # Masking
    masked_input = Masking(mask_value=0.0)(inputs)

    # LSTM 层
    lstm_out = LSTM(num_units, return_sequences=True)(masked_input)

    # Positional Encoding
    def positional_encoding(position, d_model):
        def get_angles(pos, i, d_model):
            angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
            return pos * angle_rates

        angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                                np.arange(d_model)[np.newaxis, :],
                                d_model)
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)

    pos_enc = positional_encoding(input_shape[0], num_units)
    pos_enc = tf.tile(pos_enc, [tf.shape(inputs)[0], 1, 1])
    combined = Add()([lstm_out, pos_enc])

    # Multi-Head Attention
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=num_units)(combined, combined)
    attn_output = Dropout(rate)(attn_output)
    attn_out_norm = LayerNormalization(epsilon=1e-6)(attn_output + combined)

    # Feed Forward
    ffn_output = Dense(ff_dim, activation='relu')(attn_out_norm)
    ffn_output = Dense(num_units)(ffn_output)
    ffn_output = Dropout(rate)(ffn_output)
    ffn_output_norm = LayerNormalization(epsilon=1e-6)(ffn_output + attn_out_norm)

    # Pooling + Output
    final_output = GlobalAveragePooling1D()(ffn_output_norm)
    final_output = Dense(1, activation='sigmoid')(final_output)

    model = Model(inputs=inputs, outputs=final_output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
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
    csv_file_path = 'E4_Hybrid_LSTM_Transformer_results.csv'
    os.makedirs(os.path.dirname(csv_file_path) or ".", exist_ok=True)

    train_year = 2019
    test_year = 2020
    split_ratios = [1, "original"]
    sampling_methods = ['random_undersample', 'tsmote', 'timegan', 'base']

    for method in sampling_methods:
        for blank_interval in blank_intervals:
            master_data_size_final = master_data_size_ori - blank_interval

            for split_ratio in split_ratios:
                print(f"\n🔧 Training | method={method}, blank_interval={blank_interval}, final_size={master_data_size_final}, split_ratio={split_ratio}")

                train_x_path = f"data/npy_merged_{train_year}_blank{blank_interval}_{method}/trainX.npy"
                train_y_path = f"data/npy_merged_{train_year}_blank{blank_interval}_{method}/trainY.npy"

                if not os.path.exists(train_x_path):
                    print(f"❌ Missing train data: {train_x_path}")
                    continue

                train_X = np.load(train_x_path)
                train_Y = np.load(train_y_path)
                train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.2,
                                                                  random_state=42, stratify=train_Y)

                for run in range(1, 3):
                    try:
                        print(f"🚀 Run {run}")
                        input_shape = (train_X.shape[1], train_X.shape[2])
                        model = masked_hybrid_lstm_transformer_model(input_shape)

                        ratio_str = "ratio_original" if split_ratio == "original" else f"ratio1to{split_ratio}"
                        model_save_path = f"models/Hybrid_LSTM_Transformer_{method}_blank{blank_interval}_{ratio_str}_run{run}.keras"
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
