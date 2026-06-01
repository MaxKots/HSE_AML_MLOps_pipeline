from pathlib import Path

import pandas as pd

paths = [
    Path("artifacts/metrics/benchmark_samld.csv"),
    Path("artifacts/metrics/benchmark_threshold_modes.csv"),
    Path("artifacts/metrics/benchmark_best_by_f1_threshold_modes.csv"),
]

columns = [
    "scenario",
    "model_type",
    "feature_set",
    "threshold_mode",
    "threshold",
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1_score",
    "false_positive_rate",
    "alert_rate",
    "positive_share",
    "precision_at_100",
    "recall_at_100",
    "lift_at_100",
    "precision_at_500",
    "recall_at_500",
    "lift_at_500",
]

for path in paths:
    print("\n" + "=" * 100)
    print(path)
    print("=" * 100)

    if not path.exists():
        print("Файл не найден")
        continue

    df = pd.read_csv(path)

    if "scenario" in df.columns:
        df = df[df["scenario"].astype(str).str.contains("SAML-D", case=False, na=False)]

    show_columns = [column for column in columns if column in df.columns]

    if df.empty:
        print("Нет строк SAML-D")
    else:
        print(df[show_columns].to_string(index=False))
