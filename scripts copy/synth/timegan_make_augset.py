# timegan_make_augset.py
# Build an augmented training set by fitting TimeGAN on minority-class sequences, then sampling.

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# Robust import whether running from repo root (as package) or inside folder
try:
    from synth.timegan_train import train_timegan
except ImportError:
    from timegan_train import train_timegan

def load_train_split(base_dir: str, master_size: int, method_src: str):
    X = np.load(Path(base_dir) / f"npy_merged_master_data_size_{master_size}_{method_src}" / "trainX.npy")
    y = np.load(Path(base_dir) / f"npy_merged_master_data_size_{master_size}_{method_src}" / "trainY.npy")
    return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="data", help="root data dir")
    parser.add_argument("--master_size", type=int, required=True)
    parser.add_argument("--ratio", type=int, required=True, help="target majority:minority ratio, e.g., 1:5 -> --ratio 5")
    parser.add_argument("--method_src", type=str, default="base", help="source train set method name")
    parser.add_argument("--out_method", type=str, default="timegan", help="output method folder name")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--Z", type=int, default=32)
    parser.add_argument("--H", type=int, default=128)
    args = parser.parse_args()

    X, y = load_train_split(args.base_dir, args.master_size, args.method_src)
    N, T, D = X.shape

    # decide minority class
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    if len(idx_pos) <= len(idx_neg):
        minority_label, min_idx, maj_idx = 1, idx_pos, idx_neg
    else:
        minority_label, min_idx, maj_idx = 0, idx_neg, idx_pos

    target_minority = max(1, int(len(maj_idx) / args.ratio))
    need = max(0, target_minority - len(min_idx))
    print(f"[master={args.master_size}] train N={N}, T={T}, D={D}\n"
          f"minority label={minority_label}, current={len(min_idx)}, majority={len(maj_idx)}, target={target_minority}, need_gen={need}")

    out_dir = Path(args.base_dir) / f"npy_merged_master_data_size_{args.master_size}_{args.out_method}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if need == 0:
        np.save(out_dir / "trainX.npy", X)
        np.save(out_dir / "trainY.npy", y)
        print(f"No need to generate. Saved passthrough to {out_dir}")
        return

    # build dataloader for minority-only sequences
    X_min = X[min_idx]  # (n_min, T, D), assumed scaled like in classifier
    tensor_min = torch.tensor(X_min, dtype=torch.float32)
    ds = TensorDataset(tensor_min)
    batch = min(args.batch, len(ds))
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=len(ds) >= batch)

    model = train_timegan((b for (b,) in dl), D=D, T=T, epochs=args.epochs, lr=args.lr, Z=args.Z, H=args.H)

    # follow model's actual device to synthesize
    device = next(model.parameters()).device
    need_batches = need
    with torch.no_grad():
        x_fake = model.synthesize(B=need_batches, T=T, device=str(device)).cpu().numpy()

    y_fake = np.full((need,), minority_label, dtype=y.dtype)

    X_new = np.concatenate([X, x_fake], axis=0)
    y_new = np.concatenate([y, y_fake], axis=0)

    np.save(out_dir / "trainX.npy", X_new)
    np.save(out_dir / "trainY.npy", y_new)

    meta = {
        "master_size": args.master_size,
        "ratio": args.ratio,
        "method_src": args.method_src,
        "out_method": args.out_method,
        "epochs": args.epochs,
        "batch": int(batch),
        "lr": args.lr,
        "Z": args.Z,
        "H": args.H,
        "need_generated": int(need),
        "original_minority": int(len(min_idx))
    }
    with open(out_dir / "_timegan_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved augmented set to {out_dir}\n- trainX.npy shape={X_new.shape}\n- trainY.npy shape={y_new.shape}")

if __name__ == "__main__":
    main()
