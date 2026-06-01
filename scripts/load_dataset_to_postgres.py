from __future__ import annotations

import argparse

from src.data import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Load configured dataset to PostgreSQL")
    parser.add_argument("--dataset", required=True, help="base | variant_1 | variant_2 | samld")
    parser.add_argument("--from-source", default="csv", help="Source to read from, default: csv")
    parser.add_argument("--if-exists", default="replace", choices=["replace", "append", "fail"])
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--nrows", type=int, default=None, help="Optional row limit for test loads")
    args = parser.parse_args()

    loader = DataLoader()
    read_options = {"nrows": args.nrows} if args.nrows else {}
    df = loader.load_dataset(args.dataset, source_override=args.from_source, **read_options)
    loader.save_dataset_to_source(
        df,
        args.dataset,
        source_override="postgres",
        if_exists=args.if_exists,
        chunksize=args.chunksize,
        index=False,
    )

    print(f"OK: loaded dataset={args.dataset}, rows={len(df)}, destination=postgres")


if __name__ == "__main__":
    main()
