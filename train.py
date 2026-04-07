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
        hidden_dims=(128, 128, 128),
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
                    nn.BatchNorm1d(h),
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


def nrmse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Global batch NRMSE:
        sqrt( sum((pred-target)^2) / sum(target^2) )

    Since the last dimension stores [real, imag], this is equivalent to
    complex-domain NRMSE.
    """
    num = torch.sum((pred - target) ** 2)
    den = torch.sum(target ** 2)
    return torch.sqrt(num / (den + eps) + eps)


def compute_batch_metrics(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    """
    Returns:
        rmse   = sqrt(mean squared error over all scalar elements)
        nrmse  = sqrt(sum squared error / sum target energy)
    """
    diff = pred - target
    sq_err = torch.sum(diff ** 2)
    target_energy = torch.sum(target ** 2)
    count = diff.numel()

    rmse = torch.sqrt(sq_err / max(count, 1) + eps)
    nrmse = torch.sqrt(sq_err / (target_energy + eps) + eps)
    return rmse, nrmse


def run_train_epoch(model, loader, device, optimizer):
    model.train()

    total_sq_error = 0.0
    total_target_energy = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = nrmse_loss(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        diff = pred - y
        total_sq_error += torch.square(diff).sum().item()
        total_target_energy += torch.square(y).sum().item()

    train_nrmse = float(np.sqrt(total_sq_error / max(total_target_energy, 1e-12)))
    return train_nrmse


def evaluate_nrmse_rmse(model, loader, device):
    model.eval()

    total_sq_error = 0.0
    total_target_energy = 0.0
    total_count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x)
            diff = pred - y

            total_sq_error += torch.square(diff).sum().item()
            total_target_energy += torch.square(y).sum().item()
            total_count += diff.numel()

    rmse = float(np.sqrt(total_sq_error / max(total_count, 1)))
    nrmse = float(np.sqrt(total_sq_error / max(total_target_energy, 1e-12)))
    return rmse, nrmse


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
    train_nrmse_history,
    val_nrmse_history,
    save_path: str,
    best_rmse: float,
    best_nrmse: float,
    last_rmse: float,
    last_nrmse: float,
):
    epochs = range(1, len(train_nrmse_history) + 1)
    plot_eps = 1e-12

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))

    axes[0].plot(epochs, np.maximum(train_nrmse_history, plot_eps), label="Train NRMSE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("NRMSE")
    axes[0].set_title("Train NRMSE")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both")
    axes[0].legend()

    axes[1].plot(epochs, np.maximum(val_nrmse_history, plot_eps), label="Val NRMSE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("NRMSE")
    axes[1].set_title("Validation NRMSE")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both")
    axes[1].legend()

    summary_text = (
        "[Validation metrics]\n"
        f"Best model NRMSE: {best_nrmse:.6f}\n"
        f"Last model NRMSE: {last_nrmse:.6f}"
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
    batch_size: int = 64,
    num_epochs: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    early_stopping_patience: int = 30,
    early_stopping_min_delta: float = 1e-6,
    hidden_dims=(128, 128, 128),
    dropout: float = 0.1,
):
    set_seed(seed)

    results_dir = resolve_results_dir(save_dir)
    os.makedirs(results_dir, exist_ok=True)

    train_csv_path = os.path.join(data_dir, f"wifi_lltf_dataset_{snr_db}db.csv")
    val_csv_path = os.path.join(data_dir, f"wifi_lltf_dataset_{snr_db}db_eval.csv")

    best_path = os.path.join(results_dir, f"best_model_{snr_db}db.pt")
    last_path = os.path.join(results_dir, f"last_model_{snr_db}db.pt")
    plot_path = os.path.join(results_dir, f"results_table_{snr_db}db.png")

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

    best_val_nrmse = float("inf")
    best_val_rmse = None
    last_val_rmse = None
    last_val_nrmse = None

    train_nrmse_history = []
    val_nrmse_history = []

    for epoch in range(1, num_epochs + 1):
        train_nrmse = run_train_epoch(model, train_loader, device, optimizer)
        val_rmse, val_nrmse = evaluate_nrmse_rmse(model, val_loader, device)

        train_nrmse_history.append(train_nrmse)
        val_nrmse_history.append(val_nrmse)

        last_val_rmse = val_rmse
        last_val_nrmse = val_nrmse

        if val_nrmse < best_val_nrmse:
            best_val_nrmse = val_nrmse
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), best_path)

        print(
            f"SNR {snr_db:2d} dB | "
            f"Epoch [{epoch:03d}/{num_epochs:03d}] | "
            f"Train NRMSE: {train_nrmse:.6f} | "
            f"Val NRMSE: {val_nrmse:.6f}"
        )

        early_stopper.step(val_nrmse)
        if early_stopper.should_stop:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best Val NRMSE: {best_val_nrmse:.6f}"
            )
            break

    torch.save(model.state_dict(), last_path)

    save_metric_plot(
        train_nrmse_history=train_nrmse_history,
        val_nrmse_history=val_nrmse_history,
        save_path=plot_path,
        best_rmse=best_val_rmse,
        best_nrmse=best_val_nrmse,
        last_rmse=last_val_rmse,
        last_nrmse=last_val_nrmse,
    )

    print("\nTraining finished.")
    print(f"Best Val NRMSE: {best_val_nrmse:.6f}")
    print(f"Last Val NRMSE: {last_val_nrmse:.6f}")
    print(f"Best model saved to: {best_path}")
    print(f"Last model saved to: {last_path}")
    print(f"Training curve saved to: {plot_path}")

    return {
        "snr_db": snr_db,
        "best_val_nrmse": best_val_nrmse,
        "last_val_nrmse": last_val_nrmse,
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
