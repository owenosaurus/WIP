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


def build_dataloaders(train_csv_path: str, eval_csv_path: str, batch_size: int = 128):
    train_dataset = WifiLTSChannelDataset(train_csv_path)
    eval_dataset = WifiLTSChannelDataset(eval_csv_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, eval_loader


def _complex_squared_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Convert [real, imag] pairs into per-complex-coefficient squared magnitude.
    If the last dimension is not 2, fall back to scalar-domain squared magnitude.
    """
    if x.ndim > 0 and x.size(-1) == 2:
        return torch.sum(x ** 2, dim=-1)
    return x ** 2


def _complex_abs(x: torch.Tensor) -> torch.Tensor:
    """
    Convert [real, imag] pairs into per-complex-coefficient magnitude.
    If the last dimension is not 2, fall back to scalar-domain absolute value.
    """
    if x.ndim > 0 and x.size(-1) == 2:
        return torch.sqrt(torch.sum(x ** 2, dim=-1))
    return torch.abs(x)


def rmse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Training loss: complex-domain RMSE
        RMSE = sqrt( mean( |pred - target|^2 ) )
    when the last dimension is [real, imag].

    The small eps keeps the square-root derivative numerically stable near zero.
    """
    sq_err = _complex_squared_norm(pred - target)
    return torch.sqrt(torch.mean(sq_err) + eps)


def run_train_epoch(model, loader, device, optimizer):
    model.train()

    total_sq_error = 0.0
    total_coeff_count = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = rmse_loss(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        sq_err_per_coeff = _complex_squared_norm(pred - y)
        total_sq_error += sq_err_per_coeff.sum().item()
        total_coeff_count += sq_err_per_coeff.numel()

    train_rmse = float(np.sqrt(total_sq_error / max(total_coeff_count, 1)))
    return train_rmse


def evaluate_nmae(model, loader, device):
    model.eval()

    total_abs_error = 0.0
    total_target_abs = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x)
            diff = pred - y

            total_abs_error += _complex_abs(diff).sum().item()
            total_target_abs += _complex_abs(y).sum().item()

    eval_nmae = float(total_abs_error / max(total_target_abs, 1e-12))
    return eval_nmae


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


def save_train_plot(
    train_rmse_history,
    eval_nmae_history,
    save_path: str,
    best_nmae: float,
    best_epoch=None,
):
    epochs = range(1, len(train_rmse_history) + 1)
    plot_eps = 1e-12

    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)

    axes[0].plot(
        epochs,
        np.maximum(train_rmse_history, plot_eps),
        label="Train RMSE",
    )
    if best_epoch is not None and 1 <= best_epoch <= len(train_rmse_history):
        best_train_rmse = max(train_rmse_history[best_epoch - 1], plot_eps)
        axes[0].scatter(
            [best_epoch],
            [best_train_rmse],
            color="red",
            marker="o",
            s=20,
            zorder=5,
        )
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Train RMSE")
    axes[0].grid(True, which="both")
    axes[0].legend()

    axes[1].plot(
        epochs,
        np.maximum(eval_nmae_history, plot_eps),
        label="Eval NMAE",
    )
    if best_epoch is not None and 1 <= best_epoch <= len(eval_nmae_history):
        best_eval_nmae_for_plot = max(eval_nmae_history[best_epoch - 1], plot_eps)
        axes[1].scatter(
            [best_epoch],
            [best_eval_nmae_for_plot],
            color="red",
            marker="o",
            s=20,
            zorder=5,
        )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("NMAE")
    axes[1].set_title("Evaluation NMAE")
    axes[1].grid(True, which="both")
    axes[1].legend()

    if best_epoch is not None:
        summary_text = f"Best Eval NMAE: {best_nmae:.6f} at epoch {best_epoch}"
    else:
        summary_text = f"Best Eval NMAE: {best_nmae:.6f}"
    fig.text(0.5, 0.01, summary_text, ha="center", va="bottom", fontsize=10)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
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
    num_epochs: int = 100,
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
    eval_csv_path = os.path.join(data_dir, f"wifi_lltf_dataset_{snr_db}db_eval.csv")

    plot_path = os.path.join(results_dir, f"training_plot_{snr_db}db.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, eval_loader = build_dataloaders(
        train_csv_path=train_csv_path,
        eval_csv_path=eval_csv_path,
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

    best_eval_nmae = float("inf")
    best_epoch = None

    train_rmse_history = []
    eval_nmae_history = []

    for epoch in range(1, num_epochs + 1):
        train_rmse = run_train_epoch(model, train_loader, device, optimizer)
        eval_nmae = evaluate_nmae(model, eval_loader, device)

        train_rmse_history.append(train_rmse)
        eval_nmae_history.append(eval_nmae)

        if eval_nmae < best_eval_nmae:
            best_eval_nmae = eval_nmae
            best_epoch = epoch

        print(
            f"SNR {snr_db:2d} dB | "
            f"Epoch [{epoch:03d}/{num_epochs:03d}] | "
            f"Train RMSE: {train_rmse:.6f} | "
            f"Eval NMAE: {eval_nmae:.6f}"
        )

        early_stopper.step(eval_nmae)
        if early_stopper.should_stop:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best Eval NMAE: {best_eval_nmae:.6f}"
            )
            break

    save_train_plot(
        train_rmse_history=train_rmse_history,
        eval_nmae_history=eval_nmae_history,
        save_path=plot_path,
        best_nmae=best_eval_nmae,
        best_epoch=best_epoch,
    )

    print("\nTraining finished.")
    print(f"Best Eval NMAE: {best_eval_nmae:.6f}")
    if best_epoch is not None:
        print(f"Best epoch: {best_epoch}")
    print(f"Training plot saved to: {plot_path}")

    return {
        "snr_db": snr_db,
        "best_eval_nmae": best_eval_nmae,
        "best_epoch": best_epoch,
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
