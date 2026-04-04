import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_preprocessing import WifiLTSChannelDataset


def set_seed(seed: int = 94) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_shape=(2, 64, 2),
        output_shape=(52, 2),
        hidden_dims=(256, 128, 128, 128),
        dropout=0.1,
    ):
        super().__init__()

        input_dim = int(np.prod(input_shape))
        output_dim = int(np.prod(output_shape))

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))

        self.output_shape = output_shape
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.net(x)
        return x.reshape(x.size(0), *self.output_shape)


def build_dataloaders(train_csv_path: str, val_csv_path: str, batch_size: int = 128):
    train_dataset = WifiLTSChannelDataset(train_csv_path)
    val_dataset = WifiLTSChannelDataset(val_csv_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def rmse_loss(pred, target, eps: float = 1e-8):
    mse = nn.functional.mse_loss(pred, target)
    return torch.sqrt(mse + eps)


def run_train_epoch(model, loader, device, optimizer):
    model.train()

    total_sq_error = 0.0
    total_count = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = rmse_loss(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        diff = pred - y
        total_sq_error += torch.square(diff).sum().item()
        total_count += diff.numel()

    return float(np.sqrt(total_sq_error / total_count))


def evaluate_rmse_mae(model, loader, device):
    model.eval()

    total_sq_error = 0.0
    total_abs_error = 0.0
    total_count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x)
            diff = pred - y

            total_sq_error += torch.square(diff).sum().item()
            total_abs_error += torch.abs(diff).sum().item()
            total_count += diff.numel()

    rmse = float(np.sqrt(total_sq_error / total_count))
    mae = float(total_abs_error / total_count)
    return rmse, mae


class EarlyStopping:
    def __init__(self, patience: int = 30, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, current_score: float) -> None:
        if current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def save_metric_plot(
    train_rmse_history,
    val_mae_history,
    save_path: str,
    best_rmse: float,
    best_mae: float,
    last_rmse: float,
    last_mae: float,
):
    epochs = range(1, len(train_rmse_history) + 1)
    plot_eps = 1e-12

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))

    axes[0].plot(epochs, np.maximum(train_rmse_history, plot_eps), label="Train RMSE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Train RMSE")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both")
    axes[0].legend()

    axes[1].plot(epochs, np.maximum(val_mae_history, plot_eps), label="Val MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("Validation MAE")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both")
    axes[1].legend()

    summary_text = (
        "[Validation metrics]\n"
        f"Best model  - RMSE: {best_rmse:.6f} | MAE: {best_mae:.6f}\n"
        f"Last model  - RMSE: {last_rmse:.6f} | MAE: {last_mae:.6f}"
    )
    fig.text(0.5, 0.02, summary_text, ha="center", va="bottom", fontsize=10)

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def resolve_results_dir(save_dir: str) -> str:
    save_dir = os.path.abspath(save_dir)
    if os.path.basename(save_dir) == "results":
        return save_dir
    return os.path.join(save_dir, "results")


def train_one_snr(
    snr_db: int,
    data_dir: str = "/home/jinx/project/CE01/data_set",
    save_dir: str = "/home/jinx/project/CE01/results",
    seed: int = 94,
    batch_size: int = 128,
    num_epochs: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    early_stopping_patience: int = 30,
    early_stopping_min_delta: float = 1e-6,
    hidden_dims=(256, 128, 128, 128, 64),
    dropout: float = 0.1,
):
    set_seed(seed)

    results_dir = resolve_results_dir(save_dir)
    os.makedirs(results_dir, exist_ok=True)

    train_csv_path = os.path.join(data_dir, f"wifi_lltf_dataset_{snr_db}db.csv")
    val_csv_path = os.path.join(data_dir, f"wifi_lltf_dataset_{snr_db}db_eval.csv")

    best_path = os.path.join(results_dir, f"best_model_{snr_db}db.pt")
    last_path = os.path.join(results_dir, f"last_model_{snr_db}db.pt")
    plot_path = os.path.join(results_dir, f"training_curve_{snr_db}db.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path,
        batch_size=batch_size,
    )

    model = MLPRegressor(
        input_shape=(2, 64, 2),
        output_shape=(52, 2),
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    early_stopper = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
    )

    best_val_mae = float("inf")
    best_val_rmse = None
    last_val_rmse = None
    last_val_mae = None

    train_rmse_history = []
    val_mae_history = []

    for epoch in range(1, num_epochs + 1):
        train_rmse = run_train_epoch(model, train_loader, device, optimizer)
        val_rmse, val_mae = evaluate_rmse_mae(model, val_loader, device)

        train_rmse_history.append(train_rmse)
        val_mae_history.append(val_mae)

        last_val_rmse = val_rmse
        last_val_mae = val_mae

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), best_path)

        print(
            f"SNR {snr_db:2d} dB | "
            f"Epoch [{epoch:03d}/{num_epochs:03d}] | "
            f"Train RMSE: {train_rmse:.6f} | "
            f"Val RMSE: {val_rmse:.6f} | "
            f"Val MAE: {val_mae:.6f}"
        )

        early_stopper.step(val_mae)
        if early_stopper.should_stop:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best Val MAE: {best_val_mae:.6f}"
            )
            break

    torch.save(model.state_dict(), last_path)

    save_metric_plot(
        train_rmse_history=train_rmse_history,
        val_mae_history=val_mae_history,
        save_path=plot_path,
        best_rmse=best_val_rmse,
        best_mae=best_val_mae,
        last_rmse=last_val_rmse,
        last_mae=last_val_mae,
    )

    print("\nTraining finished.")
    print(f"Best Val RMSE: {best_val_rmse:.6f}")
    print(f"Best Val MAE: {best_val_mae:.6f}")
    print(f"Last Val RMSE: {last_val_rmse:.6f}")
    print(f"Last Val MAE: {last_val_mae:.6f}")
    print(f"Best model saved to: {best_path}")
    print(f"Last model saved to: {last_path}")
    print(f"Training curve saved to: {plot_path}")

    return {
        "snr_db": snr_db,
        "best_val_rmse": best_val_rmse,
        "best_val_mae": best_val_mae,
        "last_val_rmse": last_val_rmse,
        "last_val_mae": last_val_mae,
        "best_model_path": best_path,
        "last_model_path": last_path,
        "plot_path": plot_path,
    }


def main(snr_db: int):
    return train_one_snr(snr_db=snr_db)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr_db", type=int, default=18)
    parser.add_argument("--data_dir", type=str, default="/home/jinx/project/CE01/data_set")
    parser.add_argument("--save_dir", type=str, default="/home/jinx/project/CE01/results")
    args = parser.parse_args()

    train_one_snr(
        snr_db=args.snr_db,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
    )
