import os
import numpy as np
import tensorflow as tf
from keras.models import Model,load_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Masking, Layer
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.sequence import pad_sequences
import pandas as pd

# Custom KAN Layer
class KANLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, num_functions, hidden_units=64, activation='relu', **kwargs):
        super(KANLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.num_functions = num_functions
        self.hidden_units = hidden_units
        self.activation = activation

    def build(self, input_shape):
        # 定义一维核函数权重
        self.kernels = [self.add_weight(name=f'kernel_{i}',
                                        shape=(input_shape[-1], self.num_functions),
                                        initializer='glorot_uniform',
                                        trainable=True)
                        for i in range(self.output_dim)]
        self.biases = [self.add_weight(name=f'bias_{i}',
                                       shape=(self.num_functions,),
                                       initializer='zeros',
                                       trainable=True)
                       for i in range(self.output_dim)]
        
        # 非线性组合的隐藏层
        self.hidden_layers = [tf.keras.layers.Dense(self.hidden_units, activation=self.activation)
                              for _ in range(2)]  # 使用两层隐藏层作为非线性组合
        self.output_layer = tf.keras.layers.Dense(self.output_dim, activation=None)

    def call(self, inputs):
        # 一维核函数的非线性映射
        one_d_results = []
        for i in range(self.output_dim):
            linear_output = tf.matmul(inputs, self.kernels[i]) + self.biases[i]
            non_linear_output = tf.nn.relu(linear_output)
            one_d_results.append(non_linear_output)
        
        # 将所有一维结果堆叠
        one_d_stack = tf.stack(one_d_results, axis=-1)
        
        # 应用非线性组合的隐藏层
        hidden_output = one_d_stack
        for hidden_layer in self.hidden_layers:
            hidden_output = hidden_layer(hidden_output)
        
        output = self.output_layer(hidden_output)
        # 在 KANLayer 的 call 方法的最后添加以下行，将三维输出压缩为二维
        output = tf.reduce_sum(output, axis=-1)  # 或者使用其他适当的组合方式

        return output

# Build KANS-LSTM model
def KANS_LSTM_model(input_shape, output_dim=1, num_functions=10):
    inputs = Input(shape=input_shape)
    masked_input = Masking(mask_value=0.0)(inputs)

    # LSTM layer
    lstm_out = LSTM(128, return_sequences=True)(masked_input)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out2 = LSTM(64, return_sequences=False)(lstm_out)

    # KAN layer
    kan_out = KANLayer(output_dim=output_dim, num_functions=num_functions)(lstm_out2)

    # Dense layer + Dropout
    dense_out = Dense(64, activation='relu')(kan_out)
    dropout_out = Dropout(0.3)(dense_out)

    # Output layer
    outputs = Dense(1, activation='sigmoid')(dropout_out)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

# 评估模型性能的函数
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
                    model_path = f"实验三models/真KAN/{method}/MLP_LSTM_best_model_master_data_size_{master_data_size}_ratio1to{ratio}_run_{run}.h5"
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
    output_csv = "E3_MLP_LSTM_all_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"✅ All results saved to {output_csv}")
