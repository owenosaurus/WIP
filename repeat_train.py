import argparse
import os

import pandas as pd

from train import train_one_snr


def run_snr_sweep(
    snr_list,
    data_dir: str = "/home/jinx/project/CE01/data_set",
    save_dir: str = "/home/jinx/project/CE01/results",
    csv_name: str = "results_table.csv",
):
    results = []

    for snr_db in snr_list:
        result = train_one_snr(
            snr_db=snr_db,
            data_dir=data_dir,
            save_dir=save_dir,
        )

        results.append(
            {
                "snr_db": result["snr_db"],
                "best_val_nrmse": result["best_val_nrmse"],
                "last_val_nrmse": result["last_val_nrmse"],
            }
        )

    df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, csv_name)
    df.to_csv(csv_path, index=False)

    print("\nSNR sweep finished.")
    print(f"Summary CSV saved to: {csv_path}")
    print(df)

    return df, csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr_list", type=int, nargs="+", default=[0, 3, 6, 9, 12, 15, 18])
    parser.add_argument("--data_dir", type=str, default="/home/jinx/project/CE01/data_set")
    parser.add_argument("--save_dir", type=str, default="/home/jinx/project/CE01/results")
    parser.add_argument("--csv_name", type=str, default="results_table.csv")
    args = parser.parse_args()

    run_snr_sweep(
        snr_list=args.snr_list,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        csv_name=args.csv_name,
    )


if __name__ == "__main__":
    main()
