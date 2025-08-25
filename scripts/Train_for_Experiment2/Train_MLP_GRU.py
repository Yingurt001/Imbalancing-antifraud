from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from keras.models import Model,load_model
from keras.layers import GRU, Dense, Dropout, BatchNormalization, Input, Masking
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

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
def evaluate_model(interval, test_X, test_Y, run):
    try:
        # 加载每次运行中验证集上表现最好的模型
        model_path = f'实验二models/KAN_GRU_best_model_interval_{interval}_run_{run}.h5'
        model = load_model(model_path, custom_objects={'KANLayer': KANLayer})
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
    csv_file_path = 'E2_KAN_GRU_model_results.csv'  # 保存结果的CSV文件路径
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
                model = KANS_GRU_model(input_shape)
                
                # 定义 ModelCheckpoint 回调函数，保存每次 run 中验证集上性能最好的模型
                model_save_path = f'实验二models/KAN_GRU_best_model_interval_{interval}_run_{run}.h5'
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
