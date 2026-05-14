#!/usr/bin/env python3
from __future__ import annotations

import argparse

from src.pipelines.benchmark_jobs import build_benchmark_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Сборка сводных benchmark-таблиц")
    parser.add_argument("--metrics-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_benchmark_tables(
        metrics_dir=args.metrics_dir,
        output_dir=args.output_dir,
    )
    print(summary)


if __name__ == "__main__":
    main()
