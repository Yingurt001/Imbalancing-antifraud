
#!/usr/bin/env python3
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
from keras.layers import GRU, LSTM, Dense, Dropout, BatchNormalization, Input, Masking, Layer
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix, roc_auc_score
import pandas as pd
from keras.utils import CustomObjectScope

@keras.utils.register_keras_serializable(package="keras_efficient_kan", name="GridInitializer")
class GridInitializer(initializers.Initializer):
    def __init__(self, grid_range, grid_size, spline_order):
        self.grid_range = grid_range
        self.grid_size = grid_size
        self.spline_order = spline_order

    def __call__(self, shape, dtype=None):
        h = (self.grid_range[1] - self.grid_range[0]) / self.grid_size
        start = -self.spline_order * h + self.grid_range[0]
        stop = (self.grid_size + self.spline_order) * h + self.grid_range[0]
        num = self.grid_size + 2 * self.spline_order + 1
        
        # Create the grid using numpy
        grid = np.linspace(start, stop, num, dtype=np.float32)
        
        # Repeat the grid for each feature
        grid = np.tile(grid, (shape[1], 1))
        
        # Add the batch dimension
        grid = np.expand_dims(grid, 0)
        
        # Convert to the appropriate backend tensor
        return ops.convert_to_tensor(grid, dtype=dtype)

    def get_config(self):
        return {
            "grid_range": self.grid_range,
            "grid_size": self.grid_size,
            "spline_order": self.spline_order
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@keras.utils.register_keras_serializable(package="keras_efficient_kan", name="KANLinear")
class KANLinear(Layer):
    def __init__(
        self,
        units,
        grid_size=3,
        spline_order=3,
        base_activation='relu',
        grid_range=[-1, 1],
        dropout=0.,
        use_bias=True,
        use_layernorm=True,
        **kwargs
    ):
        super(KANLinear, self).__init__(**kwargs)
        self.units = units
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation_name = base_activation
        self.grid_range = grid_range
        self.use_bias = use_bias
        self.use_layernorm = use_layernorm
        self.dropout_rate = dropout
        self.dropout = Dropout(self.dropout_rate)
        if self.use_layernorm:
            self.layer_norm = LayerNormalization(axis=-1)
        else:
            self.layer_norm = None
        self.in_features = None

    def build(self, input_shape):
        self.in_features = input_shape[-1]
        dtype = backend.floatx()
        
        self.grid = self.add_weight(
            name="grid",
            shape=[1, self.in_features, self.grid_size + 2 * self.spline_order + 1],
            initializer=GridInitializer(self.grid_range, self.grid_size, self.spline_order),
            trainable=False,
            dtype=dtype
        )

        self.base_weight = self.add_weight(
            name="base_weight",
            shape=[self.in_features, self.units],
            initializer='glorot_uniform',
            dtype=dtype
        )
        if self.use_bias:
            self.base_bias = self.add_weight(
                name="base_bias",
                shape=[self.units],
                initializer="zeros",
                dtype=dtype
            )
        self.spline_weight = self.add_weight(
            name="spline_weight",
            shape=[self.in_features * (self.grid_size + self.spline_order), self.units],
            initializer='glorot_uniform',
            dtype=dtype
        )
        if self.use_layernorm:
            self.layer_norm.build(input_shape)
        
        self.built = True

    def call(self, x, training=None):
        input_shape = ops.shape(x)
        x = ops.cast(x, self.dtype)
        x_2d = ops.reshape(x, [-1, self.in_features])
        
        if self.use_layernorm:
            x_2d = self.layer_norm(x_2d)
        
        base_activation = getattr(tf.nn, self.base_activation_name)
        base_output = ops.matmul(base_activation(x_2d), self.base_weight)
        if self.use_bias:
            base_output = ops.add(base_output, self.base_bias)
        
        spline_output = ops.matmul(self.b_splines(x_2d), self.spline_weight)
        output_2d = self.dropout(base_output, training=training) + self.dropout(spline_output, training=training)
        
        # Use ops.reshape with a tuple of integers for the new shape
        new_shape = tf.concat([input_shape[:-1], [self.units]], axis=0)
        return ops.reshape(output_2d, new_shape)

    def b_splines(self, x):
        x_expanded = ops.expand_dims(x, -1)
        bases = ops.cast((x_expanded >= self.grid[..., :-1]) & (x_expanded < self.grid[..., 1:]), self.dtype)
        
        for k in range(1, self.spline_order + 1):
            left_denominator = self.grid[..., k:-1] - self.grid[..., :-(k + 1)]
            right_denominator = self.grid[..., k + 1:] - self.grid[..., 1:-k]
            
            left = (x_expanded - self.grid[..., :-(k + 1)]) / left_denominator
            right = (self.grid[..., k + 1:] - x_expanded) / right_denominator
            bases = left * bases[..., :-1] + right * bases[..., 1:]
        return ops.reshape(bases, [ops.shape(x)[0], -1])

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)

    def get_config(self):
        config = super(KANLinear, self).get_config()
        config.update({
            'units': self.units,
            'grid_size': self.grid_size,
            'spline_order': self.spline_order,
            'base_activation': self.base_activation_name,
            'grid_range': self.grid_range,
            'dropout': self.dropout_rate,
            'use_bias': self.use_bias,
            'use_layernorm': self.use_layernorm,
        })
        return config

    def get_build_config(self):
        return {"in_features": self.in_features}

    def build_from_config(self, config):
        self.build((None, config["in_features"]))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# 构建模型
def KANS_GRU_model(input_shape, output_dim=1, num_functions=10):
    inputs = Input(shape=input_shape)
    masked_input = Masking(mask_value=0.0)(inputs)
    
    # GRU layers
    gru_out = GRU(128, return_sequences=True)(masked_input)
    gru_out = BatchNormalization()(gru_out)
    gru_out2 = GRU(64, return_sequences=False)(gru_out)
    
    # KAN layer
    kan_out = KANLinear(units=5, grid_size=3, spline_order=3)(gru_out2)
    
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
        # 使用 CustomObjectScope 来加载自定义的 KANLayer
        model_path = f'实验一models/真KAN/KAN_LSTM_best_model_interval_{interval}_run_{run}.h5'
        with CustomObjectScope({'KANLinear': KANLinear}):
            model = load_model(model_path)
    except FileNotFoundError:
        print(f"Model for interval {interval}, run {run} not found at {model_path}. Skipping this interval.")
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
    auc = roc_auc_score(test_Y, yhat_probs) 

    print(f'Interval {interval}, Run {run} - Accuracy: {accuracy:.3f}')
    print(f'Interval {interval}, Run {run} - Precision: {precision:.3f}')
    print(f'Interval {interval}, Run {run} - Recall: {recall:.3f}')
    print(f'Interval {interval}, Run {run} - F1 Score: {f1:.3f}')
    print(f'Interval {interval}, Run {run} - AUC: {auc:.3f}')
    print(f'Interval {interval}, Run {run} - Confusion Matrix:')
    return {
        'interval': interval,
        'run': run,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }

# 保存结果到 CSV
def save_results_to_csv(results, filepath):
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")

# 主函数
if __name__ == "__main__":
    results = []
    output_csv_path = "E1_KAN_GRU_results.csv"  # 定义结果输出文件
    intervals = [500000, 1000000, 1500000, 2000000, 3000000, 5000000]  # 定义多个 interval

    for interval in intervals:
        print(f"Processing interval {interval}...")

        # Step 1: 加载训练和测试数据
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

        # 训练并评估模型 20 次
        for run in range(1, 21):  # 每个 interval 运行 20 次
            print(f"Training interval {interval}, run {run}...")
            try:
                # Step 3: 构建和训练 GRU 模型
                input_shape = (train_GRU_X.shape[1], train_GRU_X.shape[2])
                model = KANS_GRU_model(input_shape)
                
                # 定义 ModelCheckpoint 回调函数，保存每次 run 中验证集上性能最好的模型
                model_save_path = f'实验一models/真KAN/KAN_LSTM_best_model_interval_{interval}_run_{run}.h5'
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
                    save_results_to_csv(results, output_csv_path)
            except Exception as e:
                print(f"Error during training or evaluation at interval {interval}, run {run}: {e}")
                save_results_to_csv(results, output_csv_path)
                continue  # 继续下一个 run 或 interval

    # 最终输出所有结果并保存
    print("\nFinal Results:")
    for result in results:
        print(result)
    
    # 保存最终结果 
    save_results_to_csv(results, output_csv_path)

