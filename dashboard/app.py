from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

from config.settings import settings
from src.data import DataLoader
from src.utils.io import load_dataframe, read_yaml
from src.utils.paths import get_artifacts_dir

st.set_page_config(
    page_title="AML Dashboard",
    page_icon="🚨",
    layout="wide",
)

API_URL = settings.dashboard_api_url
AIRFLOW_API_URL = settings.airflow_api_url
AIRFLOW_USERNAME = settings.airflow_username
AIRFLOW_PASSWORD = settings.airflow_password

DAG_OPTIONS = {
    "Training": "aml_training_dag",
    "Benchmark": "aml_benchmark_dag",
    "Drift": "aml_drift_dag",
}

DATASET_OPTIONS = {
    "Base": "base",
    "Variant I": "variant_1",
    "Variant II": "variant_2",
    "SAML-D": "samld",
}

SOURCE_OPTIONS = {
    "CSV": "csv",
    "S3 / MinIO": "s3",
    "PostgreSQL": "postgres",
}

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


@st.cache_data(show_spinner=False)
def preview_dataset_from_source(dataset_name: str, source: str, limit: int) -> pd.DataFrame:
    loader = DataLoader()
    return loader.load_dataset(dataset_name, source_override=source, nrows=limit)


def trigger_airflow_dag(dag_id: str, conf: dict) -> dict:
    response = requests.post(
        f"{AIRFLOW_API_URL}/api/v1/dags/{dag_id}/dagRuns",
        auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),
        json={"conf": conf},
        timeout=20,
    )

    if response.status_code not in (200, 201):
        raise ValueError(f"Airflow API error {response.status_code}: {response.text}")

    return response.json()


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

    tab1, tab2, tab3, tab4 = st.tabs(["Скоринг транзакций", "Drift monitoring", "Запуск пайплайнов", "О системе"])

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
        st.subheader("Источник данных и запуск Airflow DAG")
        st.caption("Dashboard выбирает источник данных и передаёт параметры запуска в Airflow. Вычисления выполняются в DAG, а не внутри Streamlit.")

        col_left, col_right = st.columns(2)

        with col_left:
            selected_dataset_label = st.selectbox(
                "Датасет для проверки загрузки",
                options=list(DATASET_OPTIONS.keys()),
                index=0,
                key="source_dataset_select",
            )
            selected_source_label = st.selectbox(
                "Источник данных",
                options=list(SOURCE_OPTIONS.keys()),
                index=0,
                key="source_type_select",
            )
            preview_limit = st.number_input(
                "Количество строк для preview",
                min_value=10,
                max_value=5000,
                value=100,
                step=10,
            )

            dataset_code = DATASET_OPTIONS[selected_dataset_label]
            source_code = SOURCE_OPTIONS[selected_source_label]

            if st.button("Проверить загрузку", key="preview_dataset_button"):
                with st.spinner("Загружаю preview из выбранного источника..."):
                    try:
                        preview_df = preview_dataset_from_source(
                            dataset_name=dataset_code,
                            source=source_code,
                            limit=int(preview_limit),
                        )
                        st.success(
                            f"Данные загружены: rows={len(preview_df)}, cols={len(preview_df.columns)}"
                        )
                        st.dataframe(preview_df.head(20), use_container_width=True)
                    except Exception as exc:
                        st.error(f"Ошибка загрузки: {exc}")

        with col_right:
            scenario_label = st.selectbox(
                "Пайплайн",
                options=list(DAG_OPTIONS.keys()),
                index=0,
                key="dag_scenario_select",
            )

            dag_id = DAG_OPTIONS[scenario_label]
            st.write(f"Airflow DAG: `{dag_id}`")
            st.write(f"Airflow API: `{AIRFLOW_API_URL}`")

            if scenario_label == "Drift":
                reference_label = st.selectbox("Reference dataset", ["Base"], index=0)
                current_label = st.selectbox("Current dataset", ["Variant I", "Variant II"], index=0)
                dag_conf = {
                    "reference_dataset_name": DATASET_OPTIONS[reference_label],
                    "current_dataset_name": DATASET_OPTIONS[current_label],
                    "source": SOURCE_OPTIONS[selected_source_label],
                }
            elif scenario_label == "Benchmark":
                dag_conf = {
                    "source": SOURCE_OPTIONS[selected_source_label],
                }
            else:
                dag_conf = {
                    "dataset_name": DATASET_OPTIONS[selected_dataset_label],
                    "source": SOURCE_OPTIONS[selected_source_label],
                }

            st.markdown("#### Конфигурация запуска")
            st.json(dag_conf)

            if st.button("Запустить DAG", type="primary", key="trigger_dag_button"):
                with st.spinner("Отправляю запрос в Airflow..."):
                    try:
                        dag_run = trigger_airflow_dag(dag_id=dag_id, conf=dag_conf)
                        st.success(f"DAG `{dag_id}` запущен")
                        st.json(dag_run)
                    except Exception as exc:
                        st.error(f"Ошибка запуска DAG: {exc}")

    with tab4:
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
