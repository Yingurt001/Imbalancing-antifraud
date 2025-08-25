import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),  ".."))
sys.path.insert(0, ROOT)
import numpy as np
from sklearn.neighbors import NearestNeighbors
from synth.timegan_train import train_timegan  # ✅ 只保留一次
import pandas as pd
import numpy as np
import random
import csv
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import NearestNeighbors

# ========= 可调：快速验证开关 =========
FRACTION = 1.0   # 例如 0.2 表示只取 20% loan 加速预处理
TG_EPOCHS = 1200 # TimeGAN 训练轮数（快速验证可设 600~1200；正式跑 3000+）
TG_H = 96
TG_Z = 24
TG_BATCH = 64
TG_LR = 5e-4
TG_PRINT_EVERY = 200
SEED = 42
# =====================================

# =====================================
# Step 1: Load and group data by loan ID
# =====================================
def load_and_process_data(input_file, fraction=1.0, seed=42):
    df = pd.read_csv(input_file)
    unique_loans = df['LOAN SEQUENCE NUMBER'].unique()

    if 0 < fraction < 1.0:
        np.random.seed(seed)
        sampled_loans = np.random.choice(unique_loans, size=int(len(unique_loans) * fraction), replace=False)
        df = df[df['LOAN SEQUENCE NUMBER'].isin(sampled_loans)]

    sorted_data = df.sort_values('LOAN SEQUENCE NUMBER')
    grouped_data = sorted_data.groupby('LOAN SEQUENCE NUMBER')
    return grouped_data, df


# =====================================
# Step 2: Build base train data
# =====================================
def build_train_data(grouped_data, master_data_size):
    train_data_X = []
    train_data_Y = []
    predict_month = 3
    all_group_sizes = []
    num_defaults = 0
    total_rows_all_groups = 0

    for _, group in grouped_data:
        group = group.sort_values('REMAINING MONTHS TO LEGAL MATURITY', ascending=False)
        group_size = len(group)
        total_rows_all_groups += group_size
        all_group_sizes.append(group_size)

        # 最后一列为标签
        if group.iloc[:, -1].sum() > 0:
            num_defaults += 1

        if group_size >= master_data_size + predict_month:
            # 去掉第一列ID，最后一列标签
            X = group.iloc[0:master_data_size, 1:-1].copy()
            Y = group.iloc[master_data_size:master_data_size + predict_month, -1]

            # 删除不希望作为特征的列（若存在）
            for col in ['ZERO BALANCE CODE', 'REMAINING MONTHS TO LEGAL MATURITY']:
                if col in X.columns:
                    X.drop(columns=col, inplace=True)

            # ✅ 强制数值 & 填缺，避免字符串/空值导致训练报错
            X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0)

            label = 1 if Y.sum() > 0 else 0
            train_data_X.append(X.values)  # shape (T, D)
            train_data_Y.append(label)

    return (
        np.array(train_data_X),   # (N, T, D)
        np.array(train_data_Y),   # (N,)
        total_rows_all_groups,
        len(all_group_sizes),
        sum(all_group_sizes),
        num_defaults
    )


# =====================================
# Step 3: Sampling Methods
# =====================================

def _print_class_info(prefix, Y):
    pos = int(np.sum(Y == 1))
    neg = int(np.sum(Y == 0))
    total = len(Y)
    rate = pos / total if total > 0 else 0.0
    print(f"{prefix} | total={total}, pos={pos}, neg={neg}, pos_rate={rate:.4f}")


def apply_tsmote_ts(
    X, Y,
    k_neighbors=5,
    n_synthetic=None,            # 生成多少少数类样本；默认补齐到与多数类相同
    alpha_mode="global",         # "global"：整段同一个alpha；"per_timestep"：每个t一个alpha
    noise_std=0.0,               # 可选：小高斯噪声 σ（建议0~0.01之间，数据已标准化时）
    smooth=False,                # 可选：对合成序列做一次轻微平滑
    random_state=42,
    metric="euclidean"
):
    """
    时序SMOTE（长度T、维度D一致）：
    1) 仅在少数类内部做kNN（基于扁平后的距离）选邻居
    2) 对整段序列逐时刻线性插值：x_syn[t] = (1-α)*x_i[t] + α*x_j[t]
       - alpha_mode="global": 整段一个α
       - alpha_mode="per_timestep": 每个t单独采样α_t，更有随机性
    3) 可选：加极小白噪声、简单平滑
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X)
    Y = np.asarray(Y)

    X_min = X[Y == 1]
    X_maj = X[Y == 0]
    n_min, n_maj = len(X_min), len(X_maj)
    if n_min == 0:
        print("❌ 少数类为0，无法做SMOTE。")
        return X, Y

    # 需要生成的数量
    if n_synthetic is None:
        n_synthetic = n_maj - n_min
    if n_synthetic <= 0:
        print("ℹ️ 已平衡或少数类不少，跳过时序SMOTE。")
        return X, Y

    # 近邻只在少数类内部
    k_use = min(k_neighbors + 1, n_min)  # 包含自身
    if k_use <= 1:
        print("⚠️ 少数类样本 < 2，无法kNN插值。")
        return X, Y

    N, T, D = X.shape
    X_min_flat = X_min.reshape(n_min, T * D)
    nn = NearestNeighbors(n_neighbors=k_use, metric=metric)
    nn.fit(X_min_flat)
    neigh_idx = nn.kneighbors(X_min_flat, return_distance=False)  # 第0列是自己

    syn_list = []
    for _ in range(n_synthetic):
        i = rng.integers(0, n_min)
        # 选一个邻居（排除自己）
        j = i if neigh_idx.shape[1] == 1 else rng.choice(neigh_idx[i][1:])

        xi = X_min[i]        # (T, D)
        xj = X_min[j]        # (T, D)

        if alpha_mode == "per_timestep":
            # 每个t一个alpha_t
            alpha = rng.random((T, 1))   # 广播到D
        else:
            # 整段一个alpha
            a = float(rng.random())
            alpha = np.full((T, 1), a, dtype=xi.dtype)

        x_syn = xi + alpha * (xj - xi)   # 逐时刻线性插值 (T,D)

        if noise_std and noise_std > 0:
            x_syn = x_syn + rng.normal(0.0, noise_std, size=x_syn.shape)

        if smooth:
            # 简单时域平滑（3点平均滤波），不引入未来信息
            x_pad = np.pad(x_syn, ((1,1),(0,0)), mode='edge')  # pad time 维
            x_syn = (x_pad[:-2] + x_pad[1:-1] + x_pad[2:]) / 3.0

        syn_list.append(x_syn)

    X_syn = np.stack(syn_list, axis=0)             # (n_synthetic, T, D)
    Y_syn = np.ones((n_synthetic,), dtype=Y.dtype) # 都是少数类

    X_aug = np.concatenate([X, X_syn], axis=0)
    Y_aug = np.concatenate([Y, Y_syn], axis=0)
    return X_aug, Y_aug

def generate_timegan_samples(X, Y, num_to_generate=None,
                             epochs=TG_EPOCHS, lr=TG_LR, Z=TG_Z, H=TG_H, batch=TG_BATCH, random_state=SEED):
    """
    用少数类样本训练一个极简 TimeGAN，然后采样补齐到与多数类相同数量（或自定义 num_to_generate）。
    - X: (N, T, D)  已经完成你项目里的预处理/缩放
    - Y: (N,)
    """
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # 1) 拆分少数/多数
    X_min = X[Y == 1]
    X_maj = X[Y == 0]
    n_min, n_maj = len(X_min), len(X_maj)
    if n_min == 0:
        print("❌ 少数类为 0，无法训练 TimeGAN")
        return X, Y

    if num_to_generate is None:
        num_to_generate = n_maj - n_min
    if num_to_generate <= 0:
        print("⚠️ 无需 TimeGAN 采样，少数类已足够")
        return X, Y

    # 2) 组建 DataLoader（只用少数类训练 TimeGAN）
    N, T, D = X.shape
    tensor_min = torch.tensor(X_min, dtype=torch.float32)
    ds = TensorDataset(tensor_min)
    if len(ds) < 2:
        print("⚠️ 少数类样本过少（<2），跳过 TimeGAN。")
        return X, Y
    batch = min(batch, len(ds))
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=len(ds) >= batch)

    # 3) 训练 TimeGAN（你项目里的极简实现）
    model = train_timegan((b for (b,) in dl), D=D, T=T,
                          epochs=epochs, lr=lr, Z=Z, H=H, print_every=TG_PRINT_EVERY)

    # 4) 采样需要的数量（跟随模型设备）
    device = next(model.parameters()).device
    with torch.no_grad():
        x_fake = model.synthesize(B=num_to_generate, T=T, device=str(device)).cpu().numpy()

    y_fake = np.ones((num_to_generate,), dtype=Y.dtype)

    # 5) 合并回去
    X_aug = np.concatenate([X, x_fake], axis=0)
    Y_aug = np.concatenate([Y, y_fake], axis=0)
    return X_aug, Y_aug


def apply_sampling(X, Y, method, random_state=SEED):
    """
    四种方法：
    - base: 不动
    - random_undersample: 欠采到与正样本同数
    - tsmote: 先欠采多数类到 1:2，再对少数类用 SMOTE 补齐
    - timegan: 先欠采多数类到 1:2，再用 TimeGAN 补齐
    """
    rng = np.random.default_rng(random_state)

    X = np.asarray(X)
    Y = np.asarray(Y)

    pos_idx = np.where(Y == 1)[0]
    neg_idx = np.where(Y == 0)[0]

    if method == "base":
        _print_class_info("BASE before", Y)
        return X, Y

    elif method == "random_undersample":
        # 欠采多数类到与少数类同数
        if len(neg_idx) > len(pos_idx):
            keep_neg = rng.choice(neg_idx, size=len(pos_idx), replace=False)
        else:
            keep_neg = neg_idx
        keep_idx = np.concatenate([pos_idx, keep_neg])
        rng.shuffle(keep_idx)
        X_new, Y_new = X[keep_idx], Y[keep_idx]
        _print_class_info("RUS after", Y_new)
        return X_new, Y_new

    elif method == "tsmote":
        # ① 欠采多数类到 1:2
        if len(neg_idx) > 2 * len(pos_idx):
            keep_neg = rng.choice(neg_idx, size=2 * len(pos_idx), replace=False)
        else:
            keep_neg = neg_idx
        keep_idx = np.concatenate([pos_idx, keep_neg])
        X_part, Y_part = X[keep_idx], Y[keep_idx]
        _print_class_info("tSMOTE step1 (after 1:2 undersample)", Y_part)

        # ② 在这个子集上做“时序感知”的插值SMOTE
        X_aug, Y_aug = apply_tsmote_ts(
            X_part, Y_part,
            k_neighbors=5,
            n_synthetic=None,          # 默认补到与多数类相同
            alpha_mode="per_timestep", # 更有多样性；想更稳用 "global"
            noise_std=0.0,             # 数据已标准化可设 0~0.01
            smooth=False,              # 如需更顺滑可设 True
            random_state=random_state
        )
        _print_class_info("tSMOTE step2 (after SMOTE-TS)", Y_aug)
        return X_aug, Y_aug

    elif method == "timegan":
        # ① 欠采多数类到 1:2
        if len(neg_idx) > 2 * len(pos_idx):
            keep_neg = rng.choice(neg_idx, size=2 * len(pos_idx), replace=False)
        else:
            keep_neg = neg_idx
        keep_idx = np.concatenate([pos_idx, keep_neg])
        X_part, Y_part = X[keep_idx], Y[keep_idx]
        _print_class_info("TimeGAN step1 (after 1:2 undersample)", Y_part)

        # ② TimeGAN 补齐
        X_aug, Y_aug = generate_timegan_samples(
            X_part, Y_part,
            epochs=TG_EPOCHS, lr=TG_LR, Z=TG_Z, H=TG_H, batch=TG_BATCH,
            random_state=random_state
        )
        _print_class_info("TimeGAN step2 (after synth)", Y_aug)
        return X_aug, Y_aug

    else:
        raise ValueError(f"Unknown sampling method: {method}")


# =====================================
# Step 4: Save processed train data to .npy files
# =====================================
def save_data(X, Y, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'trainX.npy'), X)
    np.save(os.path.join(output_dir, 'trainY.npy'), Y)
    print(f"✅ Saved to {output_dir} | Samples: {len(Y)}, Defaults: {int(Y.sum())}, Non-defaults: {len(Y) - int(Y.sum())}")


# =====================================
# Step 5: Main loop for batch processing
# =====================================
if __name__ == "__main__":
    # 固定随机种子，保证可复现
    random.seed(SEED)
    np.random.seed(SEED)

    input_files = [
        "data/processed_data/historical_data_time_2019Q1.csv",
        "data/processed_data/historical_data_time_2019Q2.csv",
        "data/processed_data/historical_data_time_2019Q3.csv",
        "data/processed_data/historical_data_time_2019Q4.csv"
    ]

    master_data_sizes = [9, 12, 15, 18, 21, 24, 27, 30, 33, 36]
    sampling_methods = ["random_undersample"]
    # 快速验证可临时改成：
    # input_files = ["data/processed_data/historical_data_time_2019Q1.csv"]
    # master_data_sizes = [12]
    # sampling_methods = ["timegan"]

    stats_csv_path = "train_data_statistics_summary_2019.csv"
    write_header = not os.path.exists(stats_csv_path)

    first_file = os.path.basename(input_files[0])
    year_str = next((part[:4] for part in first_file.split('_') if part[:4].isdigit()), "Unknown")

    for master_data_size in master_data_sizes:
        for method in sampling_methods:
            all_X, all_Y = [], []
            total_sample_size = 0
            total_num_loans = 0
            total_loan_length = 0
            total_num_defaults = 0

            for input_file in input_files:
                if not os.path.exists(input_file):
                    print(f"❌ File not found: {input_file}")
                    continue

                # ✅ 支持 FRACTION 抽样快速跑
                grouped_data, _ = load_and_process_data(input_file, fraction=FRACTION, seed=SEED)
                X, Y, sample_size, num_loans, loan_length, num_defaults = build_train_data(grouped_data, master_data_size)

                if len(X) == 0:
                    continue

                # 在每个季度上先做采样，再汇总
                sampled_X, sampled_Y = apply_sampling(X, Y, method, random_state=SEED)
                all_X.append(sampled_X)
                all_Y.append(sampled_Y)

                total_sample_size += sample_size
                total_num_loans += num_loans
                total_loan_length += loan_length
                total_num_defaults += num_defaults

            if all_X:
                final_X = np.concatenate(all_X, axis=0)
                final_Y = np.concatenate(all_Y, axis=0)

                # ✅ 加入年份区分
                output_path = f"data/npy_merged_{year_str}_master_data_size_{master_data_size}_{method}"
                save_data(final_X, final_Y, output_path)

                final_num_defaults = int(final_Y.sum())
                final_num_total = len(final_Y)
                default_rate = total_num_defaults / total_num_loans if total_num_loans > 0 else 0.0
                final_default_rate = final_num_defaults / final_num_total if final_num_total > 0 else 0.0
                avg_loan_length = total_loan_length / total_num_loans if total_num_loans > 0 else 0.0
                final_avg_loan_length = master_data_size

                with open(stats_csv_path, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    if write_header:
                        writer.writerow([
                            "year",
                            "master_data_size",
                            "method",
                            "original_sample_size",
                            "original_num_loans",
                            "avg_loan_length",
                            "num_defaults",
                            "default_rate",
                            "final_sample_size",
                            "final_num_loans",
                            "final_avg_loan_length",
                            "final_num_defaults",
                            "final_default_rate"
                        ])
                        write_header = False

                    writer.writerow([
                        year_str,
                        master_data_size,
                        method,
                        total_sample_size,
                        total_num_loans,
                        round(avg_loan_length, 2),
                        total_num_defaults,
                        round(default_rate, 4),
                        final_num_total,
                        final_num_total,
                        final_avg_loan_length,
                        final_num_defaults,
                        round(final_default_rate, 4)
                    ])

                print(f"📈 Stats saved to {stats_csv_path}")
            else:
                print(f"⚠️ No data to save for master_data_size={master_data_size}, method={method}")
