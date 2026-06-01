import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_dataset(path: str) -> pd.DataFrame:
    """
    Загружает датасет из CSV или Parquet.
    """
    path = Path(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(path)

    raise ValueError(f"Неподдерживаемый формат файла: {path.suffix}")


def prepare_feature_matrix(
    df: pd.DataFrame,
    target_col: str | None = None,
    drop_cols: list[str] | None = None,
    max_categories: int = 20,
) -> pd.DataFrame:
    """
    Подготавливает матрицу признаков:
    - удаляет целевую переменную и служебные колонки;
    - кодирует категориальные признаки;
    - исключает признаки с чрезмерно большим количеством категорий;
    - оставляет только числовую матрицу.
    """
    df = df.copy()

    if drop_cols is None:
        drop_cols = []

    cols_to_drop = list(drop_cols)

    if target_col and target_col in df.columns:
        cols_to_drop.append(target_col)

    existing_drop_cols = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=existing_drop_cols)

    # Удаляем полностью пустые признаки
    df = df.dropna(axis=1, how="all")

    # Разделяем признаки по типам
    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Оставляем только категориальные признаки с умеренным числом уникальных значений
    valid_categorical_cols = []
    for col in categorical_cols:
        unique_count = df[col].nunique(dropna=True)
        if 1 < unique_count <= max_categories:
            valid_categorical_cols.append(col)

    # Кодируем категориальные признаки
    if valid_categorical_cols:
        encoded = pd.get_dummies(
            df[valid_categorical_cols],
            prefix=valid_categorical_cols,
            dummy_na=True,
            drop_first=False,
        )
    else:
        encoded = pd.DataFrame(index=df.index)

    numeric_part = df[numeric_cols].copy()

    # Приводим bool к int
    for col in numeric_part.columns:
        if numeric_part[col].dtype == "bool":
            numeric_part[col] = numeric_part[col].astype(int)

    feature_matrix = pd.concat([numeric_part, encoded], axis=1)

    # Удаляем константные признаки
    nunique = feature_matrix.nunique(dropna=False)
    feature_matrix = feature_matrix.loc[:, nunique > 1]

    # Заполняем пропуски медианой для числовых колонок
    feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan)
    feature_matrix = feature_matrix.fillna(feature_matrix.median(numeric_only=True))

    return feature_matrix


def select_top_features(
    feature_matrix: pd.DataFrame,
    target: pd.Series | None = None,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Ограничивает число признаков для читаемости heatmap.

    Если передана целевая переменная, выбираются признаки с наибольшей
    абсолютной корреляцией с target.

    Если target не передан, выбираются признаки с наибольшей дисперсией.
    """
    if feature_matrix.shape[1] <= top_n:
        return feature_matrix

    if target is not None:
        temp = feature_matrix.copy()
        temp["_target"] = target.values

        corr_with_target = (
            temp.corr(numeric_only=True)["_target"]
            .drop("_target")
            .abs()
            .sort_values(ascending=False)
        )

        selected_cols = corr_with_target.head(top_n).index.tolist()
        return feature_matrix[selected_cols]

    variances = feature_matrix.var(numeric_only=True).sort_values(ascending=False)
    selected_cols = variances.head(top_n).index.tolist()

    return feature_matrix[selected_cols]


def plot_correlation_heatmap(
    feature_matrix: pd.DataFrame,
    output_path: str,
    title: str = "Тепловая матрица корреляций признаков",
    figsize: tuple[int, int] = (16, 12),
) -> None:
    """
    Строит и сохраняет тепловую матрицу корреляций признаков.
    """
    corr_matrix = feature_matrix.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(corr_matrix.values, aspect="auto")

    ax.set_title(title, fontsize=14, pad=20)

    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.index)))

    ax.set_xticklabels(corr_matrix.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr_matrix.index, fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Коэффициент корреляции", rotation=270, labelpad=20)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Построение тепловой матрицы признаков для ML/MLOps-диссертации"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Путь к CSV или Parquet-файлу",
    )

    parser.add_argument(
        "--output",
        default="artifacts/feature_heatmap.png",
        help="Путь для сохранения изображения",
    )

    parser.add_argument(
        "--target",
        default=None,
        help="Название целевой переменной",
    )

    parser.add_argument(
        "--drop-cols",
        nargs="*",
        default=[],
        help="Служебные колонки, которые нужно исключить",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Количество признаков для отображения на heatmap",
    )

    parser.add_argument(
        "--title",
        default="Тепловая матрица корреляций признаков",
        help="Заголовок графика",
    )

    args = parser.parse_args()

    df = load_dataset(args.input)

    target = None
    if args.target and args.target in df.columns:
        target = df[args.target]

    feature_matrix = prepare_feature_matrix(
        df=df,
        target_col=args.target,
        drop_cols=args.drop_cols,
    )

    feature_matrix = select_top_features(
        feature_matrix=feature_matrix,
        target=target,
        top_n=args.top_n,
    )

    plot_correlation_heatmap(
        feature_matrix=feature_matrix,
        output_path=args.output,
        title=args.title,
    )

    print(f"Heatmap сохранена: {args.output}")
    print(f"Количество отображённых признаков: {feature_matrix.shape[1]}")


if __name__ == "__main__":
    main()
