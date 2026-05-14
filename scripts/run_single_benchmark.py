#!/usr/bin/env python3
from __future__ import annotations

import argparse

from src.pipelines.benchmark_jobs import run_single_benchmark_cli


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Запуск одного benchmark-job")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--train-dataset-name", required=True, choices=["base", "variant_1", "variant_2", "synthaml"])
    parser.add_argument("--eval-dataset-name", default=None)
    parser.add_argument("--model-type", required=True, choices=["lightgbm", "xgboost"])
    parser.add_argument("--feature-set", required=True, choices=["raw", "engineered", "prepared"])
    parser.add_argument("--output-file", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_single_benchmark_cli(
        job_id=args.job_id,
        scenario=args.scenario,
        dataset_label=args.dataset_label,
        train_dataset_name=args.train_dataset_name,
        eval_dataset_name=args.eval_dataset_name,
        model_type=args.model_type,
        feature_set=args.feature_set,
        output_file=args.output_file,
    )
    print(result)


if __name__ == "__main__":
    main()
