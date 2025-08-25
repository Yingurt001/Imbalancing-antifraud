import os
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, Input, Masking, Flatten, Attention
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def attention(input_shape, epochs=10, batch_size=64):
    inputs = Input(shape=input_shape)
    masked_input = Masking(mask_value=0.0)(inputs)
    lstm_out = LSTM(128, return_sequences=True)(masked_input)
    lstm_out2 = LSTM(64, return_sequences=True)(lstm_out)
    
    # 添加注意力层
    attention_out = Attention()([lstm_out2, lstm_out2])
    
    # Flatten层
    flattened_out = Flatten()(attention_out)  # 将输出展平为一维数组
    
    # 密集层
    dense_out = Dense(25, activation='sigmoid')(flattened_out)
    output = Dense(1, activation='sigmoid')(dense_out)

    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

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
            model = attention(input_shape)
            # 训练模型
            history = model.fit(train_GRU_X, train_GRU_y, epochs=20, batch_size=64, validation_data=(val_GRU_X, val_GRU_y), verbose=0)
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
        results_df.to_csv(f'/Users/zhangying/Documents/Imbalanced_Atten_LSTM/Results/Attention_results.csv', index=False)
        print("All results saved to 'GRU_results.csv'.")
