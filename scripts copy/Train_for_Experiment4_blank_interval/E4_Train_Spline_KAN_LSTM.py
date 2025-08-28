import tensorflow as tf
ops = tf
import os
import numpy as np
import keras
from keras import backend
from keras.src import initializers
from keras.src.layers import Layer, Dropout, LayerNormalization
from keras.models import Model, load_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Masking
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import pandas as pd
from tqdm.keras import TqdmCallback

# 固定参数
master_data_size_ori = 21
predict_month = 3
blank_intervals = list(range(1, 13))

# ========================
# 自定义 KANLayer
# ========================
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
        return {"grid_range": self.grid_range, "grid_size": self.grid_size, "spline_order": self.spline_order}


@keras.utils.register_keras_serializable(package="keras_efficient_kan", name="KANLinear")
class KANLinear(Layer):
    def __init__(self, units, grid_size=3, spline_order=3,
                 base_activation='relu', grid_range=[-1, 1],
                 dropout=0., use_bias=True, use_layernorm=True, **kwargs):
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
        self.layer_norm = LayerNormalization(axis=-1) if self.use_layernorm else None
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
        self.base_weight = self.add_weight("base_weight", [self.in_features, self.units], initializer='glorot_uniform', dtype=dtype)
        if self.use_bias:
            self.base_bias = self.add_weight("base_bias", [self.units], initializer="zeros", dtype=dtype)
        self.spline_weight = self.add_weight(
            "spline_weight", [self.in_features * (self.grid_size + self.spline_order), self.units],
            initializer='glorot_uniform', dtype=dtype
        )
        if self.use_layernorm: self.layer_norm.build(input_shape)
        self.built = True

    def call(self, x, training=None):
        input_shape = ops.shape(x)
        x = ops.cast(x, self.dtype)
        x_2d = ops.reshape(x, [-1, self.in_features])
        if self.use_layernorm: x_2d = self.layer_norm(x_2d)
        base_activation = getattr(tf.nn, self.base_activation_name)
        base_output = ops.matmul(base_activation(x_2d), self.base_weight)
        if self.use_bias: base_output = ops.add(base_output, self.base_bias)
        spline_output = ops.matmul(self.b_splines(x_2d), self.spline_weight)
        output_2d = self.dropout(base_output, training=training) + self.dropout(spline_output, training=training)
        new_shape = tf.concat([input_shape[:-1], [self.units]], axis=0)
        return ops.reshape(output_2d, new_shape)

    def b_splines(self, x):
        x_expanded = ops.expand_dims(x, -1)
        bases = ops.cast((x_expanded >= self.grid[..., :-1]) & (x_expanded < self.grid[..., 1:]), self.dtype)
        for k in range(1, self.spline_order + 1):
            left_denom = self.grid[..., k:-1] - self.grid[..., :-(k + 1)]
            right_denom = self.grid[..., k + 1:] - self.grid[..., 1:-k]
            left = (x_expanded - self.grid[..., :-(k + 1)]) / left_denom
            right = (self.grid[..., k + 1:] - x_expanded) / right_denom
            bases = left * bases[..., :-1] + right * bases[..., 1:]
        return ops.reshape(bases, [ops.shape(x)[0], -1])


# ========================
# 模型构建 KAN-LSTM
# ========================
def KANS_LSTM_model(input_shape, output_dim=1):
    inputs = Input(shape=input_shape)
    masked_input = Masking(mask_value=0.0)(inputs)
    lstm_out = LSTM(128, return_sequences=True)(masked_input)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out2 = LSTM(64, return_sequences=False)(lstm_out)
    kan_out = KANLinear(units=5, grid_size=3, spline_order=3)(lstm_out2)
    dense_out = Dense(64, activation='relu')(kan_out)
    dropout_out = Dropout(0.3)(dense_out)
    outputs = Dense(1, activation='sigmoid')(dropout_out)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ========================
# 模型评估
# ========================
def evaluate_model_by_method(model_path, test_X, test_Y):
    try:
        model = load_model(model_path, custom_objects={'KANLinear': KANLinear})
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
    csv_file_path = 'E4_KAN_LSTM_model_results.csv'
    os.makedirs(os.path.dirname(csv_file_path) or ".", exist_ok=True)

    train_year = 2019
    test_year = 2020
    split_ratios = [1, "original"]
    sampling_methods = ['random_undersample']

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
                train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=42, stratify=train_Y)

                try:
                    input_shape = (train_X.shape[1], train_X.shape[2])
                    model = KANS_LSTM_model(input_shape)

                    ratio_str = "ratio_original" if split_ratio == "original" else f"ratio1to{split_ratio}"
                    model_save_path = f"models/KAN_LSTM_best_model_{method}_blank{blank_interval}_{ratio_str}.keras"
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

                    print(f"🔍 Evaluating {method} | blank={blank_interval}, split_ratio={split_ratio}")
                    result = evaluate_model_by_method(model_save_path, test_X, test_Y)

                    if result:
                        result.update(dict(
                            train_year=train_year,
                            test_year=test_year,
                            blank_interval=blank_interval,
                            master_data_size_final=master_data_size_final,
                            split_ratio=split_ratio,
                            method=method
                        ))
                        results.append(result)
                        pd.DataFrame(results).to_csv(csv_file_path, index=False)

                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue
