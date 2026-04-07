from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

from config.settings import settings
from src.utils.io import load_dataframe, read_yaml
from src.utils.paths import get_artifacts_dir

st.set_page_config(
    page_title="AML Dashboard",
    page_icon="🚨",
    layout="wide",
)

API_URL = settings.dashboard_api_url

INTEGER_COLUMNS = [
    "prev_address_months_count",
    "current_address_months_count",
    "customer_age",
    "days_since_request",
    "zip_count_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "email_is_free",
    "phone_home_valid",
    "phone_mobile_valid",
    "bank_months_count",
    "has_other_cards",
    "foreign_request",
    "keep_alive_session",
    "device_distinct_emails_8w",
    "device_fraud_count",
    "month",
]

FLOAT_COLUMNS = [
    "income",
    "name_email_similarity",
    "intended_balcon_amount",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "credit_risk_score",
    "proposed_credit_limit",
    "session_length_in_minutes",
]

STRING_COLUMNS = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]


@st.cache_data
def load_local_dataset(dataset_name: str) -> pd.DataFrame:
    dataset_map = {
        "Base": settings.base_dataset,
        "Variant I": settings.drift_dataset_1,
        "Variant II": settings.drift_dataset_2,
    }

    path = dataset_map[dataset_name]
    return load_dataframe(path)


def check_api_health() -> dict:
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {
            "status": "error",
            "is_model_loaded": False,
            "loaded_model_type": None,
            "error": str(exc),
        }


def prepare_transactions_payload(df: pd.DataFrame) -> list[dict]:
    working_df = df.drop(columns=["fraud_bool"], errors="ignore").copy()

    for column in INTEGER_COLUMNS:
        if column in working_df.columns:
            working_df[column] = pd.to_numeric(working_df[column], errors="coerce").round().astype("Int64")

    for column in FLOAT_COLUMNS:
        if column in working_df.columns:
            working_df[column] = pd.to_numeric(working_df[column], errors="coerce").astype(float)

    for column in STRING_COLUMNS:
        if column in working_df.columns:
            working_df[column] = working_df[column].astype(str)

    records = working_df.where(pd.notnull(working_df), None).to_dict(orient="records")
    return records


def score_batch(df: pd.DataFrame) -> list[dict]:
    payload = {
        "transactions": prepare_transactions_payload(df)
    }

    response = requests.post(
        f"{API_URL}/predict_batch",
        json=payload,
        timeout=120,
    )

    if not response.ok:
        raise ValueError(f"API error {response.status_code}: {response.text}")

    return response.json()["items"]


def load_drift_summary() -> list[dict]:
    reports_dir = get_artifacts_dir() / "reports"
    if not reports_dir.exists():
        return []

    summaries = []
    for path in sorted(reports_dir.glob("drift_*.yaml")):
        try:
            report = read_yaml(path)
            metrics = report["metrics"][0]["result"]

            summaries.append(
                {
                    "report_file": path.name,
                    "dataset_drift": metrics.get("dataset_drift"),
                    "share_of_drifted_columns": metrics.get("share_of_drifted_columns"),
                    "number_of_drifted_columns": metrics.get("number_of_drifted_columns"),
                    "number_of_columns": metrics.get("number_of_columns"),
                }
            )
        except Exception:
            continue

    return summaries


def recommendation_to_label(recommendation: str) -> str:
    mapping = {
        "red": "🔴 block",
        "yellow": "🟡 review",
        "green": "🟢 pass",
    }
    return mapping.get(recommendation, recommendation)


def main() -> None:
    st.title("🚨 AML MLOps Dashboard")
    st.caption("Интерактивный интерфейс для анализа подозрительных финансовых операций")

    with st.sidebar:
        st.header("Настройки")
        dataset_name = st.selectbox("Датасет", ["Base", "Variant I", "Variant II"], index=0)
        sample_size = st.slider("Размер выборки", min_value=10, max_value=500, value=100, step=10)
        run_scoring = st.button("Запустить batch scoring", type="primary")

        st.markdown("---")
        st.write("API URL:", API_URL)

    st.subheader("Состояние API")
    health = check_api_health()

    if health.get("status") == "ok":
        st.success(f"API доступен. Загружена модель: {health.get('loaded_model_type')}")
    else:
        st.error(f"API недоступен: {health.get('error', 'unknown error')}")

    tab1, tab2, tab3 = st.tabs(["Скоринг транзакций", "Drift monitoring", "О системе"])

    with tab1:
        st.subheader("Скоринг транзакций")

        df = load_local_dataset(dataset_name).head(sample_size)
        st.write("Пример входных данных:")
        st.dataframe(df.head(10), use_container_width=True)

        if run_scoring:
            with st.spinner("Выполняется scoring через FastAPI..."):
                try:
                    scored_items = score_batch(df)
                    scored_df = pd.DataFrame(scored_items)

                    scored_df["recommendation_label"] = scored_df["recommendation"].apply(recommendation_to_label)

                    st.success(f"Скоринг завершён. Обработано записей: {len(scored_df)}")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Всего записей", len(scored_df))
                    col2.metric("Красные", int((scored_df["recommendation"] == "red").sum()))
                    col3.metric("Жёлтые", int((scored_df["recommendation"] == "yellow").sum()))

                    st.markdown("### Результаты batch scoring")
                    filter_values = st.multiselect(
                        "Фильтр по рекомендации",
                        options=["red", "yellow", "green"],
                        default=["red", "yellow", "green"],
                    )

                    filtered_df = scored_df[scored_df["recommendation"].isin(filter_values)].copy()
                    filtered_df = filtered_df.sort_values("prediction_score", ascending=False)

                    st.dataframe(
                        filtered_df[
                            ["row_index", "prediction_score", "prediction_label", "recommendation_label"]
                        ],
                        use_container_width=True,
                    )

                    st.markdown("### Детальный просмотр транзакции")
                    selected_row_index = st.selectbox(
                        "Выберите строку",
                        options=filtered_df["row_index"].tolist(),
                    )

                    selected_item = filtered_df[filtered_df["row_index"] == selected_row_index].iloc[0]

                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Score", f"{selected_item['prediction_score']:.4f}")
                    col_b.metric("Label", int(selected_item["prediction_label"]))
                    col_c.metric("Recommendation", recommendation_to_label(selected_item["recommendation"]))

                    st.markdown("#### Top positive factors")
                    positive_df = pd.DataFrame(selected_item["top_positive_factors"])
                    if not positive_df.empty:
                        st.dataframe(positive_df, use_container_width=True)
                    else:
                        st.info("Положительные факторы риска отсутствуют")

                    st.markdown("#### Top negative factors")
                    negative_df = pd.DataFrame(selected_item["top_negative_factors"])
                    if not negative_df.empty:
                        st.dataframe(negative_df, use_container_width=True)
                    else:
                        st.info("Факторы, снижающие риск, отсутствуют")

                    st.markdown("#### Human-readable explanation")
                    for reason in selected_item["human_readable_reasons"]:
                        st.write(f"- {reason}")

                    st.markdown("#### Исходные поля выбранной записи")
                    original_row = df.iloc[int(selected_row_index)]
                    st.json(original_row.to_dict())

                except Exception as exc:
                    st.error(f"Ошибка batch scoring: {exc}")

    with tab2:
        st.subheader("Drift monitoring")

        drift_summaries = load_drift_summary()
        if not drift_summaries:
            st.info("Drift-отчёты пока не найдены. Сначала выполнить pipeline drift-check.")
        else:
            drift_df = pd.DataFrame(drift_summaries)
            st.dataframe(drift_df, use_container_width=True)

            latest_report = drift_df.iloc[-1]
            st.markdown("### Последний drift report")
            st.write(f"Файл: `{latest_report['report_file']}`")
            st.write(f"Dataset drift: `{latest_report['dataset_drift']}`")
            st.write(f"Доля drifted columns: `{latest_report['share_of_drifted_columns']}`")
            st.write(
                f"Drifted columns: `{latest_report['number_of_drifted_columns']}` "
                f"из `{latest_report['number_of_columns']}`"
            )

    with tab3:
        st.subheader("О системе")
        st.markdown(
            """
            **Технологический стек**
            - Airflow: оркестрация пайплайнов
            - MLflow: трекинг экспериментов и реестр моделей
            - LightGBM / XGBoost: обучение моделей
            - SHAP: интерпретация решений
            - Evidently: детекция дрейфа
            - FastAPI: REST API для инференса
            - Streamlit: интерактивный дашборд
            """
        )

        st.markdown("### Артефакты")
        artifacts_dir = get_artifacts_dir()
        st.write(f"Путь к артефактам: `{artifacts_dir}`")

        feature_spec_path = artifacts_dir / "metrics" / "feature_spec.yaml"
        if feature_spec_path.exists():
            feature_spec = read_yaml(feature_spec_path)
            st.write(f"Количество признаков: {len(feature_spec.get('feature_columns', []))}")
            st.write(f"Категориальных: {len(feature_spec.get('categorical_columns', []))}")
            st.write(f"Числовых: {len(feature_spec.get('numerical_columns', []))}")


if __name__ == "__main__":
    main()
