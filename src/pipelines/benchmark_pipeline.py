from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from src.data import DataLoader
from src.models.benchmark import AMLBenchmarkRunner
from src.utils.io import save_dataframe, save_json
from src.utils.logger import get_logger
from src.utils.paths import ensure_directories, get_artifacts_dir

logger = get_logger(__name__)


def run_benchmark_pipeline() -> dict[str, Any]:
    ensure_directories()

    loader = DataLoader()
    runner = AMLBenchmarkRunner()

    base_df = loader.load_dataset("base")
    variant_1_df = loader.load_dataset("variant_1")
    variant_2_df = loader.load_dataset("variant_2")

    results = []

    experiments = [
        # baseline without feature engineering
        {
            "experiment_name": "baseline_raw_lightgbm",
            "train_df": base_df,
            "test_df": None,
            "model_type": "lightgbm",
            "use_feature_engineering": False,
        },
        {
            "experiment_name": "baseline_raw_xgboost",
            "train_df": base_df,
            "test_df": None,
            "model_type": "xgboost",
            "use_feature_engineering": False,
        },

        # proposed approach
        {
            "experiment_name": "proposed_fe_lightgbm",
            "train_df": base_df,
            "test_df": None,
            "model_type": "lightgbm",
            "use_feature_engineering": True,
        },
        {
            "experiment_name": "proposed_fe_xgboost",
            "train_df": base_df,
            "test_df": None,
            "model_type": "xgboost",
            "use_feature_engineering": True,
        },

        # drift robustness
        {
            "experiment_name": "drift_variant_1_lightgbm",
            "train_df": base_df,
            "test_df": variant_1_df,
            "model_type": "lightgbm",
            "use_feature_engineering": True,
        },
        {
            "experiment_name": "drift_variant_2_lightgbm",
            "train_df": base_df,
            "test_df": variant_2_df,
            "model_type": "lightgbm",
            "use_feature_engineering": True,
        },
        {
            "experiment_name": "drift_variant_1_xgboost",
            "train_df": base_df,
            "test_df": variant_1_df,
            "model_type": "xgboost",
            "use_feature_engineering": True,
        },
        {
            "experiment_name": "drift_variant_2_xgboost",
            "train_df": base_df,
            "test_df": variant_2_df,
            "model_type": "xgboost",
            "use_feature_engineering": True,
        },
    ]

    for config in experiments:
        result = runner.run_single_experiment(**config)
        results.append(asdict(result))

    results_df = pd.DataFrame(results)

    metrics_dir = get_artifacts_dir() / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    csv_path = metrics_dir / "benchmark_results.csv"
    json_path = metrics_dir / "benchmark_results.json"

    save_dataframe(results_df, csv_path, index=False)
    save_json(results, json_path)

    summary = {
        "n_experiments": len(results),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "best_by_pr_auc": results_df.sort_values("pr_auc", ascending=False).iloc[0].to_dict(),
        "best_by_roc_auc": results_df.sort_values("roc_auc", ascending=False).iloc[0].to_dict(),
        "best_by_precision_at_100": results_df.sort_values("precision_at_100", ascending=False).iloc[0].to_dict(),
    }

    logger.info(f"Benchmark pipeline завершён. Проведено экспериментов: {len(results)}")
    return summary