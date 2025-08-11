import os
import numpy as np
import tensorflow as tf
from keras.models import Model,load_model
from keras.layers import LSTM, Dense, Dropout, Input, Masking, Flatten, Attention
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,roc_auc_score
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
# ===== 评估函数 =====
def evaluate_model(model_path, test_X, test_Y):
    try:
        model = load_model(model_path)
    except:
        print(f"❌ Failed to load model: {model_path}")
        return None

    y_pred = model.predict(test_X, verbose=0)[:, 0]
    y_class = (y_pred > 0.5).astype(int)
    acc = accuracy_score(test_Y, y_class)
    prec = precision_score(test_Y, y_class)
    rec = recall_score(test_Y, y_class)
    f1 = f1_score(test_Y, y_class)
    auc = roc_auc_score(test_Y, y_pred)
    tn, fp, fn, tp = confusion_matrix(test_Y, y_class).ravel()
    return dict(acc=acc, precision=prec, recall=rec, f1=f1, auc=auc, tn=tn, fp=fp, fn=fn, tp=tp)


# ===== 主流程 =====
if __name__ == "__main__":
    results = []  # 合并所有方法的结果

    master_data_sizes = [12, 18]
    split_ratios = [1, 2, 5, 10, 25]
    sampling_methods = ['original', 'random', 'tSMOTE', 'timeGAN']

    for method in sampling_methods:
        for master_data_size in master_data_sizes:
            for ratio in split_ratios:
                testX_path = f"data/test_{method}_2019_master{master_data_size}_ratio1to{ratio}/testX.npy"
                testY_path = f"data/test_{method}_2019_master{master_data_size}_ratio1to{ratio}/testY.npy"
                if not os.path.exists(testX_path):
                    continue
                test_X = np.load(testX_path)
                test_Y = np.load(testY_path)

                for run in range(1, 3):
                    model_path = f"实验三models/真KAN/{method}/LSTM_Attention_best_model_master_data_size_{master_data_size}_ratio1to{ratio}_run_{run}.h5"
                    print(f"🔍 Evaluating: {method} | size={master_data_size} | ratio=1:{ratio} | run={run}")
                    result = evaluate_model(model_path, test_X, test_Y)
                    if result:
                        result.update(dict(
                            method=method,
                            master_data_size=master_data_size,
                            split_ratio=ratio,
                            run=run
                        ))
                        results.append(result)

    # 📝 所有方法的结果统一保存
    df = pd.DataFrame(results)
    output_csv = "E3_LSTM_Attention_all_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"✅ All results saved to {output_csv}")
