import argparse
import os
import random
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Project-local defaults
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(PROJECT_DIR, "data_set")
DEFAULT_SAVE_DIR = os.path.join(PROJECT_DIR, "results")
DEFAULT_TRAIN_CSV_PATTERN = "dataset_onetap_{snr_db}db.csv"
DEFAULT_EVAL_CSV_PATTERN = "dataset_onetap_{snr_db}db_eval.csv"


# ============================================================
# Basic utilities
# ============================================================

def set_seed(seed: int = 94) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_results_dir(save_dir: str) -> str:
    save_dir = os.path.abspath(save_dir)
    return save_dir if os.path.basename(save_dir) == "results" else os.path.join(save_dir, "results")


def parse_column_list(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


# ============================================================
# Dataset
# ============================================================

class LSChannelCFRDataset(Dataset):
    """
    Dataset for supervised channel estimation after FFT-LS.

    Expected sample:
        x: LS after FFT, 104-dimensional real vector
        y: true channel frequency response, 104-dimensional real vector

    Column resolution rule:
        1) If input_cols/target_cols are explicitly given, use them.
        2) Else, try common prefixes such as x0~x103 and y0~y103.
        3) Else, require exactly 208 numeric columns:
              first 104 columns  -> input x
              next 104 columns   -> label y

    Complex layout for metrics is handled outside the dataset.
    """

    def __init__(
        self,
        csv_path: str,
        input_dim: int = 104,
        target_dim: int = 104,
        input_cols: list[str] | None = None,
        target_cols: list[str] | None = None,
    ):
        super().__init__()

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self.csv_path = csv_path
        self.input_dim = input_dim
        self.target_dim = target_dim

        df = pd.read_csv(csv_path)

        input_cols = input_cols or self._find_prefixed_columns(
            df,
            prefixes=("x", "x_", "input", "input_", "ls", "ls_", "ls_est", "ls_est_"),
            dim=input_dim,
        )
        target_cols = target_cols or self._find_prefixed_columns(
            df,
            prefixes=("y", "y_", "target", "target_", "label", "label_", "h", "h_", "cfr", "cfr_"),
            dim=target_dim,
        )

        if input_cols is not None or target_cols is not None:
            if input_cols is None:
                raise ValueError(
                    "Target columns were found or provided, but input columns were not. "
                    "Pass --input_cols explicitly or rename input columns as x0~x103."
                )
            if target_cols is None:
                raise ValueError(
                    "Input columns were found or provided, but target columns were not. "
                    "Pass --target_cols explicitly or rename label columns as y0~y103."
                )

            missing_input = [col for col in input_cols if col not in df.columns]
            missing_target = [col for col in target_cols if col not in df.columns]
            if missing_input or missing_target:
                raise ValueError(
                    f"Missing columns. missing_input={missing_input}, "
                    f"missing_target={missing_target}"
                )

            x_np = df.loc[:, input_cols].to_numpy(dtype=np.float32)
            y_np = df.loc[:, target_cols].to_numpy(dtype=np.float32)
            self.column_mode = "named columns"
        else:
            numeric_df = df.select_dtypes(include=[np.number])
            unnamed_mask = numeric_df.columns.astype(str).str.startswith("Unnamed:")
            numeric_df = numeric_df.loc[:, ~unnamed_mask]

            required_dim = input_dim + target_dim
            found_dim = numeric_df.shape[1]
            if found_dim != required_dim:
                raise ValueError(
                    f"CSV must contain exactly {required_dim} numeric columns "
                    f"when input_cols/target_cols are not given. "
                    f"Found {found_dim} numeric columns in {csv_path}. "
                    f"This strict check prevents silent slicing of extra numeric columns. "
                    f"If your CSV has metadata or additional numeric columns, pass "
                    f"--input_cols and --target_cols explicitly."
                )

            x_np = numeric_df.iloc[:, :input_dim].to_numpy(dtype=np.float32)
            y_np = numeric_df.iloc[:, input_dim:required_dim].to_numpy(dtype=np.float32)
            self.column_mode = f"exactly {required_dim} numeric columns"

        if x_np.shape[1] != input_dim:
            raise ValueError(f"Input dimension mismatch: expected {input_dim}, got {x_np.shape[1]}")
        if y_np.shape[1] != target_dim:
            raise ValueError(f"Target dimension mismatch: expected {target_dim}, got {y_np.shape[1]}")
        if not np.isfinite(x_np).all():
            raise ValueError(f"Input contains NaN or Inf: {csv_path}")
        if not np.isfinite(y_np).all():
            raise ValueError(f"Target contains NaN or Inf: {csv_path}")

        self.x = torch.from_numpy(x_np).float()
        self.y = torch.from_numpy(y_np).float()

    @staticmethod
    def _find_prefixed_columns(
        df: pd.DataFrame,
        prefixes: Iterable[str],
        dim: int,
    ) -> list[str] | None:
        for prefix in prefixes:
            cols = [f"{prefix}{idx}" for idx in range(dim)]
            if all(col in df.columns for col in cols):
                return cols
        return None

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


# ============================================================
# Model
# ============================================================

class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int = 104,
        output_dim: int = 104,
        hidden_dims: tuple[int, ...] = (104, 104),
        dropout: float = 0.1,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        return self.net(x)


# ============================================================
# I/Q-domain metrics for 104-dimensional flattened CFR
# ============================================================

def _as_iq(x: torch.Tensor, iq_layout: str = "interleaved") -> torch.Tensor:
    """
    Convert a 104-dimensional flattened vector into (..., 52, 2).

    iq_layout="interleaved": [Re0, Im0, Re1, Im1, ..., Re51, Im51]
    iq_layout="ri_block":    [Re0, Re1, ..., Re51, Im0, Im1, ..., Im51]
    """
    if x.ndim > 0 and x.size(-1) == 2:
        return x

    if x.ndim > 0 and x.size(-1) == 104:
        if iq_layout == "interleaved":
            return x.reshape(*x.shape[:-1], 52, 2)
        if iq_layout == "ri_block":
            real = x[..., :52]
            imag = x[..., 52:]
            return torch.stack((real, imag), dim=-1)
        if iq_layout == "scalar":
            return x
        raise ValueError(f"Unsupported iq_layout: {iq_layout}")

    return x


def _iq_power(x: torch.Tensor, iq_layout: str = "interleaved") -> torch.Tensor:
    x_iq = _as_iq(x, iq_layout=iq_layout)
    if x_iq.ndim > 0 and x_iq.size(-1) == 2:
        return torch.sum(x_iq ** 2, dim=-1)
    return x_iq ** 2


def _iq_abs(x: torch.Tensor, iq_layout: str = "interleaved") -> torch.Tensor:
    return torch.sqrt(_iq_power(x, iq_layout=iq_layout).clamp_min(0.0))


def rmse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    iq_layout: str = "interleaved",
    eps: float = 1e-12,
) -> torch.Tensor:
    return torch.sqrt(torch.mean(_iq_power(pred - target, iq_layout=iq_layout)) + eps)


# ============================================================
# Data
# ============================================================

def build_dataloaders(
    train_csv_path: str,
    eval_csv_path: str,
    batch_size: int = 64,
    input_dim: int = 104,
    target_dim: int = 104,
    input_cols: list[str] | None = None,
    target_cols: list[str] | None = None,
):
    train_dataset = LSChannelCFRDataset(
        train_csv_path,
        input_dim=input_dim,
        target_dim=target_dim,
        input_cols=input_cols,
        target_cols=target_cols,
    )
    eval_dataset = LSChannelCFRDataset(
        eval_csv_path,
        input_dim=input_dim,
        target_dim=target_dim,
        input_cols=input_cols,
        target_cols=target_cols,
    )

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

    return train_loader, eval_loader, train_dataset, eval_dataset


# ============================================================
# Train / Evaluation
# ============================================================

def train_one_epoch(model, loader, optimizer, device, iq_layout: str = "interleaved") -> float:
    model.train()

    total_sq_error = 0.0
    total_count = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = rmse_loss(pred, y, iq_layout=iq_layout)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            sq_error = _iq_power(pred - y, iq_layout=iq_layout)
            total_sq_error += sq_error.sum().item()
            total_count += sq_error.numel()

    return float(np.sqrt(total_sq_error / max(total_count, 1)))


def evaluate(model, loader, device, iq_layout: str = "interleaved") -> tuple[float, float]:
    model.eval()

    total_sq_error = 0.0
    total_count = 0

    total_abs_error = 0.0
    total_target_abs = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x)
            diff = pred - y

            sq_error = _iq_power(diff, iq_layout=iq_layout)

            total_sq_error += sq_error.sum().item()
            total_count += sq_error.numel()

            total_abs_error += _iq_abs(diff, iq_layout=iq_layout).sum().item()
            total_target_abs += _iq_abs(y, iq_layout=iq_layout).sum().item()

    eval_rmse = float(np.sqrt(total_sq_error / max(total_count, 1)))
    eval_nmae = float(total_abs_error / max(total_target_abs, 1e-12))

    return eval_rmse, eval_nmae


# ============================================================
# Plot
# ============================================================

def _plot_metric(ax, epochs, values, title: str, ylabel: str, label: str, best_epoch):
    values = np.asarray(values, dtype=float)

    ax.plot(epochs, values, label=label)

    if best_epoch is not None and 1 <= best_epoch <= len(values):
        ax.scatter(
            [best_epoch],
            [values[best_epoch - 1]],
            color="red",
            marker="o",
            s=15,
            zorder=5,
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both")
    ax.legend()


def save_train_plot(
    history: dict,
    save_path: str,
    best_epoch: int | None,
    best_eval_nmae: float,
) -> None:
    epochs = np.arange(1, len(history["train_rmse"]) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(8, 9.5), sharex=True)

    _plot_metric(
        axes[0],
        epochs,
        history["train_rmse"],
        title="Train RMSE",
        ylabel="RMSE",
        label="Train RMSE",
        best_epoch=best_epoch,
    )

    _plot_metric(
        axes[1],
        epochs,
        history["eval_rmse"],
        title="Evaluation RMSE",
        ylabel="RMSE",
        label="Eval RMSE",
        best_epoch=best_epoch,
    )

    _plot_metric(
        axes[2],
        epochs,
        history["eval_nmae"],
        title="Evaluation NMAE",
        ylabel="NMAE",
        label="Eval NMAE",
        best_epoch=best_epoch,
    )

    axes[2].set_xlabel("Epoch")

    if best_epoch is None:
        summary = f"Best Eval NMAE: {best_eval_nmae:.6f}"
    else:
        summary = f"Best Eval NMAE: {best_eval_nmae:.6f} at epoch {best_epoch}"

    fig.text(0.5, 0.01, summary, ha="center", va="bottom", fontsize=10)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main training function
# ============================================================

def train_one_snr(
    snr_db: int,
    data_dir: str = DEFAULT_DATA_DIR,
    save_dir: str = DEFAULT_SAVE_DIR,
    seed: int = 94,
    batch_size: int = 64,
    num_epochs: int = 200,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 20,
    early_stopping_min_delta: float = 1e-6,
    hidden_dims: tuple[int, ...] = (104, 104),
    dropout: float = 0.1,
    input_dim: int = 104,
    output_dim: int = 104,
    iq_layout: str = "interleaved",
    train_csv_path: str | None = None,
    eval_csv_path: str | None = None,
    input_cols: list[str] | None = None,
    target_cols: list[str] | None = None,
):
    set_seed(seed)

    if iq_layout in {"interleaved", "ri_block"} and output_dim != 104:
        raise ValueError(
            f"iq_layout='{iq_layout}' assumes a 104-dimensional flattened complex CFR, "
            f"but output_dim={output_dim}. Use output_dim=104 or iq_layout='scalar'."
        )

    results_dir = resolve_results_dir(save_dir)
    os.makedirs(results_dir, exist_ok=True)

    if train_csv_path is None:
        train_csv_path = os.path.join(data_dir, DEFAULT_TRAIN_CSV_PATTERN.format(snr_db=snr_db))
    if eval_csv_path is None:
        eval_csv_path = os.path.join(data_dir, DEFAULT_EVAL_CSV_PATTERN.format(snr_db=snr_db))

    plot_path = os.path.join(results_dir, f"training_plot_{snr_db}db.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, eval_loader, train_dataset, eval_dataset = build_dataloaders(
        train_csv_path=train_csv_path,
        eval_csv_path=eval_csv_path,
        batch_size=batch_size,
        input_dim=input_dim,
        target_dim=output_dim,
        input_cols=input_cols,
        target_cols=target_cols,
    )

    model = MLPRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = {
        "train_rmse": [],
        "eval_rmse": [],
        "eval_nmae": [],
    }

    best_eval_nmae = float("inf")
    best_eval_rmse = None
    best_epoch = None

    early_stop_best = float("inf")
    epochs_without_improvement = 0

    print("Training configuration")
    print(f"  SNR: {snr_db} dB")
    print(f"  Device: {device}")
    print(f"  Train CSV: {train_csv_path}")
    print(f"  Eval CSV: {eval_csv_path}")
    print(f"  Train samples: {len(train_dataset)} | column mode: {train_dataset.column_mode}")
    print(f"  Eval samples:  {len(eval_dataset)} | column mode: {eval_dataset.column_mode}")
    print(f"  Model: {input_dim} -> {hidden_dims} -> {output_dim}")
    print(f"  I/Q metric layout: {iq_layout}")

    for epoch in range(1, num_epochs + 1):
        train_rmse = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            iq_layout=iq_layout,
        )
        eval_rmse, eval_nmae = evaluate(
            model,
            eval_loader,
            device,
            iq_layout=iq_layout,
        )

        history["train_rmse"].append(train_rmse)
        history["eval_rmse"].append(eval_rmse)
        history["eval_nmae"].append(eval_nmae)

        if eval_nmae < best_eval_nmae:
            best_eval_nmae = eval_nmae
            best_eval_rmse = eval_rmse
            best_epoch = epoch

        print(
            f"SNR {snr_db:2d} dB | "
            f"Epoch [{epoch:03d}/{num_epochs:03d}] | "
            f"Train RMSE: {train_rmse:.6f} | "
            f"Eval RMSE: {eval_rmse:.6f} | "
            f"Eval NMAE: {eval_nmae:.6f}"
        )

        improved_for_early_stop = eval_nmae < early_stop_best - early_stopping_min_delta

        if improved_for_early_stop:
            early_stop_best = eval_nmae
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best Eval NMAE: {best_eval_nmae:.6f}"
            )
            break

    save_train_plot(
        history=history,
        save_path=plot_path,
        best_epoch=best_epoch,
        best_eval_nmae=best_eval_nmae,
    )

    print("\nTraining finished.")
    print(f"Best Eval NMAE: {best_eval_nmae:.6f}")

    if best_eval_rmse is not None:
        print(f"Eval RMSE at best epoch: {best_eval_rmse:.6f}")

    if best_epoch is not None:
        print(f"Best epoch: {best_epoch}")

    print(f"Training plot saved to: {plot_path}")

    return {
        "snr_db": snr_db,
        "best_eval_nmae": best_eval_nmae,
        "best_eval_rmse": best_eval_rmse,
        "best_epoch": best_epoch,
        "plot_path": plot_path,
    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr_db", type=int, default=18)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)

    # Optional direct paths. If omitted, default dataset_onetap filename rule is used.
    parser.add_argument("--train_csv_path", type=str, default=None)
    parser.add_argument("--eval_csv_path", type=str, default=None)

    # Optional explicit CSV column lists, comma-separated.
    parser.add_argument("--input_cols", type=str, default=None)
    parser.add_argument("--target_cols", type=str, default=None)

    # For 104-dimensional flattened complex vectors.
    parser.add_argument(
        "--iq_layout",
        type=str,
        default="interleaved",
        choices=["interleaved", "ri_block", "scalar"],
        help=(
            "interleaved: [Re0, Im0, Re1, Im1, ...], "
            "ri_block: [Re0..Re51, Im0..Im51], "
            "scalar: treat all 104 dimensions independently"
        ),
    )

    args = parser.parse_args()

    train_one_snr(
        snr_db=args.snr_db,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        train_csv_path=args.train_csv_path,
        eval_csv_path=args.eval_csv_path,
        iq_layout=args.iq_layout,
        input_cols=parse_column_list(args.input_cols),
        target_cols=parse_column_list(args.target_cols),
    )
