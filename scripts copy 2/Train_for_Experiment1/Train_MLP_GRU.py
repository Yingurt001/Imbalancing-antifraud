from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
import tensorflow as tf
from keras.models import Model
from keras.layers import GRU, Dense, Dropout, BatchNormalization, Input, Masking
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
# 自定义 KANLayer
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

# 构建模型
def KANS_GRU_model(input_shape, output_dim=1, num_functions=10):
    inputs = Input(shape=input_shape)
    masked_input = Masking(mask_value=0.0)(inputs)
    
    # GRU layers
    gru_out = GRU(128, return_sequences=True)(masked_input)
    gru_out = BatchNormalization()(gru_out)
    gru_out2 = GRU(64, return_sequences=False)(gru_out)
    
    # KAN layer
    kan_out = KANLayer(output_dim=output_dim, num_functions=num_functions)(gru_out2)
    
    # Dense layer + Dropout
    dense_out = Dense(64, activation='relu')(kan_out)
    dropout_out = Dropout(0.3)(dense_out)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid')(dropout_out)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
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
            model = KANS_GRU_model(input_shape)
            # 训练模型
            history = model.fit(train_GRU_X, train_GRU_y, epochs=25, batch_size=64, validation_data=(val_GRU_X, val_GRU_y), verbose=0)
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
        results_df.to_csv(f'/Users/zhangying/Documents/Imbalanced_Atten_LSTM/Results/KAN_GRU_results.csv', index=False)
        print("All results saved to 'GRU_results.csv'.")
