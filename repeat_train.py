import argparse
import os
from typing import Iterable

import pandas as pd

from train import (
    resolve_results_dir,
    train_one_snr,
    parse_column_list,
    DEFAULT_DATA_DIR,
    DEFAULT_SAVE_DIR,
)

def _build_csv_path(
    data_dir: str,
    csv_pattern: str | None,
    snr_db: int,
) -> str | None:

    if csv_pattern is None or csv_pattern.strip() == "":
        return None

    filename = csv_pattern.format(snr_db=snr_db, snr=snr_db)
    return os.path.join(data_dir, filename)


def run_snr_sweep(
    snr_list: Iterable[int],
    data_dir: str = DEFAULT_DATA_DIR,
    save_dir: str = DEFAULT_SAVE_DIR,
    csv_name: str = "results_table.csv",
    train_csv_pattern: str | None = None,
    eval_csv_pattern: str | None = None,
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
    input_cols: list[str] | None = None,
    target_cols: list[str] | None = None,
):
    """
    Train the LS-to-CFR MLP for multiple SNR values and save the best NMAE summary.

    The output CSV keeps the same table format as the previous sweep code:
        one column: DNN_NMAE
        row order: sorted SNR list
    """
    results_dir = resolve_results_dir(save_dir)
    os.makedirs(results_dir, exist_ok=True)

    records = []

    for snr_db in snr_list:
        train_csv_path = _build_csv_path(data_dir, train_csv_pattern, snr_db)
        eval_csv_path = _build_csv_path(data_dir, eval_csv_pattern, snr_db)

        print("\n" + "=" * 70)
        print(f"Start training | SNR {snr_db} dB")
        print("=" * 70)

        result = train_one_snr(
            snr_db=snr_db,
            data_dir=data_dir,
            save_dir=save_dir,
            seed=seed,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            hidden_dims=tuple(hidden_dims),
            dropout=dropout,
            input_dim=input_dim,
            output_dim=output_dim,
            iq_layout=iq_layout,
            train_csv_path=train_csv_path,
            eval_csv_path=eval_csv_path,
            input_cols=input_cols,
            target_cols=target_cols,
        )

        records.append(
            {
                "snr_db": result["snr_db"],
                "best_eval_nmae": result["best_eval_nmae"],
                "best_eval_rmse": result["best_eval_rmse"],
                "best_epoch": result["best_epoch"],
                "plot_path": result["plot_path"],
            }
        )

    records.sort(key=lambda item: item["snr_db"])

    sorted_snr_list = [item["snr_db"] for item in records]
    nmae_values = [item["best_eval_nmae"] for item in records]

    # Keep the previous results_table.csv format.
    df = pd.DataFrame({"DNN_NMAE": nmae_values})

    csv_path = os.path.join(results_dir, csv_name)
    df.to_csv(csv_path, index=False)

    print("\nsweep finished.")
    print(f"Summary CSV saved to: {csv_path}")
    print(f"Row order (SNR_dB): {sorted_snr_list}")
    print(df)

    return df, csv_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--snr_list", type=int, nargs="+", default=[0, 3, 6, 9, 12, 15, 18])
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--csv_name", type=str, default="results_table.csv")

    # Optional per-SNR filename patterns.
    # If omitted, train_one_snr() uses the project-local default patterns:
    #   dataset_onetap_{snr_db}db.csv
    #   dataset_onetap_{snr_db}db_eval.csv
    parser.add_argument("--train_csv_pattern", type=str, default=None)
    parser.add_argument("--eval_csv_pattern", type=str, default=None)

    # Training hyperparameters. Defaults match train.py.
    parser.add_argument("--seed", type=int, default=94)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-6)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[104, 104])
    parser.add_argument("--dropout", type=float, default=0.1)

    # LS input and CFR label dimensions.
    parser.add_argument("--input_dim", type=int, default=104)
    parser.add_argument("--output_dim", type=int, default=104)

    # CSV column control. Use comma-separated names if automatic detection is not enough.
    parser.add_argument("--input_cols", type=str, default=None)
    parser.add_argument("--target_cols", type=str, default=None)

    # 104-dimensional flattened complex-vector layout for RMSE/NMAE.
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

    run_snr_sweep(
        snr_list=args.snr_list,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        csv_name=args.csv_name,
        train_csv_pattern=args.train_csv_pattern,
        eval_csv_pattern=args.eval_csv_pattern,
        seed=args.seed,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        iq_layout=args.iq_layout,
        input_cols=parse_column_list(args.input_cols),
        target_cols=parse_column_list(args.target_cols),
    )


if __name__ == "__main__":
    main()
