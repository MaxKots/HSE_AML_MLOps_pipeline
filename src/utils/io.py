from pathlib import Path
from typing import Any
import json
import shutil

import joblib
import numpy as np
import pandas as pd
import yaml


def _make_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _make_serializable(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [_make_serializable(item) for item in obj]

    if isinstance(obj, tuple):
        return [_make_serializable(item) for item in obj]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj


def read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable_data = _make_serializable(data)

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(serializable_data, file, allow_unicode=True, sort_keys=False)


def read_json(path: str | Path) -> dict[str, Any] | list[Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: dict[str, Any] | list[Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable_data = _make_serializable(data)

    with path.open("w", encoding="utf-8") as file:
        json.dump(serializable_data, file, ensure_ascii=False, indent=2)


def save_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".csv":
        df.to_csv(path, index=index)
    elif path.suffix == ".parquet":
        df.to_parquet(path, index=index)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {path.suffix}")


def load_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Неподдерживаемый формат файла: {path.suffix}")


def save_object(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_object(path: str | Path) -> Any:
    return joblib.load(path)


def copy_file(src: str | Path, dst: str | Path) -> None:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
