import tensorflow as tf
ops = tf
import os
import numpy as np
import keras
from keras import backend
from keras.src import initializers
from keras.src.layers import Layer, Dropout, LayerNormalization
from keras.models import Model, load_model
from keras.layers import GRU, Dense, Dropout, BatchNormalization, Input, Masking
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import pandas as pd
from tqdm.keras import TqdmCallback
import time


# 自定义 KANLayer
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
        
        grid = np.linspace(start, stop, num, dtype=np.float32)
        grid = np.tile(grid, (shape[1], 1))
        grid = np.expand_dims(grid, 0)
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
    
    gru_out = GRU(128, return_sequences=True)(masked_input)
    gru_out = BatchNormalization()(gru_out)
    gru_out2 = GRU(64, return_sequences=False)(gru_out)
    
    kan_out = KANLinear(units=5, grid_size=3, spline_order=3)(gru_out2)
    
    dense_out = Dense(64, activation='relu')(kan_out)
    dropout_out = Dropout(0.3)(dense_out)
    
    outputs = Dense(1, activation='sigmoid')(dropout_out)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
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


# ✅ 主程序
if __name__ == "__main__":
    results = []
    csv_file_path = 'E3_KAN_GRU_model_results2020.csv'
    os.makedirs(os.path.dirname(csv_file_path) or ".", exist_ok=True)

    master_data_sizes = [12, 18]
    split_ratios = [1, 2, 5, 10, 25, "original"]  # 🔑 加入 original
    sampling_methods = ['random_undersample', 'tsmote', 'timegan', 'base']
    test_year = 2020

    for method in sampling_methods:
        for master_data_size in master_data_sizes:
            for split_ratio in split_ratios:
                print(f"\n🔧 Training | method = {method}, master_data_size = {master_data_size}, split_ratio = {split_ratio}")

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
                        model = KANS_GRU_model(input_shape)

                        ratio_str = f"ratio{split_ratio}" if split_ratio == "original" else f"ratio1to{split_ratio}"
                        model_save_path = f'实验三models/真KAN/KAN_GRU_best_model_{method}_master_data_size_{master_data_size}_{ratio_str}_run_{run}.keras'
                        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

                        checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max')

                        model.fit(
                            train_X, train_Y,
                            epochs=25,
                            batch_size=64,
                            validation_data=(val_X, val_Y),
                            callbacks=[checkpoint, TqdmCallback(verbose=1)],
                            verbose=0
                        )

                        # 🔑 测试集路径根据 split_ratio 选择
                        if split_ratio == "original":
                            test_x_path = f"data/test_merged_{test_year}_master{master_data_size}_ratio_original/testX.npy"
                            test_y_path = f"data/test_merged_{test_year}_master{master_data_size}_ratio_original/testY.npy"
                        else:
                            test_x_path = f"data/test_merged_{test_year}_master{master_data_size}_ratio1to{split_ratio}/testX.npy"
                            test_y_path = f"data/test_merged_{test_year}_master{master_data_size}_ratio1to{split_ratio}/testY.npy"

                        if not os.path.exists(test_x_path):
                            print(f"⚠️ Missing test data: {test_x_path}")
                            continue

                        test_X = np.load(test_x_path)
                        test_Y = np.load(test_y_path)

                        print(f"🔍 Evaluating {method} | size={master_data_size}, ratio={split_ratio}, run={run}")
                        result = evaluate_model_by_method(model_save_path, test_X, test_Y)

                        if result:
                            result.update(dict(
                                year=test_year,
                                master_data_size=master_data_size,
                                split_ratio=split_ratio,
                                run=run,
                                method=method
                            ))
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
