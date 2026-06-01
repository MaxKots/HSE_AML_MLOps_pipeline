from __future__ import annotations

import argparse

from src.data import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-check configured data source")
    parser.add_argument("--dataset", default="base")
    parser.add_argument("--source", default="csv", choices=["csv", "s3", "postgres", "api"])
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    loader = DataLoader()
    df = loader.load_dataset(args.dataset, source_override=args.source, nrows=args.limit)

    print(f"OK: dataset={args.dataset}, source={args.source}, rows={len(df)}, cols={len(df.columns)}")
    print(df.head().to_string())


if __name__ == "__main__":
    main()
