import os
import numpy as np
import tensorflow as tf
from keras.models import Model,load_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Masking, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import seaborn as sns

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
def evaluate_model(interval, test_X, test_Y, run):
    try:
        # 加载保存好的模型
        model_path = f'实验一models/Hybrid_original_rnow__{interval}_run_{run}.h5'
        model = load_model(model_path)
    except FileNotFoundError:
        print(f"Model for interval {interval}, run {run} not found at {model_path}. Skipping this interval.")
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

# 主函数
if __name__ == "__main__":
    results = []
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

        # 训练并评估模型 10 次
        for run in range(1, 21):  # 每个 interval 运行 20 次
            print(f"Training interval {interval}, run {run}...")

            # Step 3: 构建和训练 GRU 模型
            input_shape = (train_GRU_X.shape[1], train_GRU_X.shape[2])
            model = masked_LSTM_model(input_shape)
            
            # 训练模型
            model.fit(train_GRU_X, train_GRU_y, epochs=20, batch_size=64, validation_data=(val_GRU_X, val_GRU_y))

            # 评估模型在训练集和验证集上的性能
            _, train_acc = model.evaluate(train_GRU_X, train_GRU_y, verbose=0)
            _, val_acc = model.evaluate(val_GRU_X, val_GRU_y, verbose=0)
            print(f'Train Accuracy: {train_acc:.3f}, Validation Accuracy: {val_acc:.3f}')

            # 保存模型
            model_save_path = f'实验一models/LSTM_original_rnow__{interval}_run_{run}.h5'
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            model.save(model_save_path)
            print(f"Model saved to '{model_save_path}'.")

            # Step 4: 在测试集上评估模型
            result = evaluate_model(interval, test_X, test_Y, run)
            if result:
                results.append(result)

    # 输出所有结果
    print("\nFinal Results:")
    for result in results:
        print(result)
