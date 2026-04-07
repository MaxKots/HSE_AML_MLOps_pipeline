# AML Pipeline

Сквозной MLOps-пайплайн для выявления сомнительных финансовых операций.

Проект реализует полный жизненный цикл ML-модели для задачи AML: загрузку и валидацию данных, feature engineering, обучение моделей, интерпретацию решений, мониторинг дрейфа, serving через REST API и визуализацию результатов в интерактивном дашборде.

## Основные возможности

- загрузка данных из локальных файлов;
- валидация схемы и контроль корректности входных данных;
- feature engineering для табличных транзакционных данных;
- обучение моделей LightGBM и XGBoost;
- логирование экспериментов и метрик в MLflow;
- локальная и глобальная интерпретация решений с помощью SHAP;
- мониторинг дрейфа данных с помощью Evidently;
- online и batch scoring через FastAPI;
- визуализация результатов и explanations в Streamlit dashboard;
- оркестрация пайплайнов через Apache Airflow.

## Технологический стек

- Python 3.10+
- Pandas, NumPy, scikit-learn
- LightGBM, XGBoost
- MLflow
- SHAP
- Evidently
- FastAPI
- Streamlit
- Apache Airflow
- Docker, Docker Compose
- PostgreSQL
- MinIO

## Структура проекта

![schema](https://github.com/MaxKots/HSE_AML_MLOps_pipeline/blob/main/.assets/schema.svg)

## Входные данные

Для локального запуска необходимо поместить датасеты в директорию `data/raw/`:

- `Base.csv`
- `Variant I.csv`
- `Variant II.csv`

Используемый target-признак:
- `fraud_bool`

## Рекомендуемый локальный запуск через venv

### 1. Создание виртуального окружения

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Обновление pip-инструментов

```bash
python -m pip install -upgrade pip setuptools wheel
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Подготовка датасетов

Разместить файлы:

```text
data/raw/Base.csv
data/raw/Variant I.csv
data/raw/Variant II.csv
```

## Пошаговый запуск пайплайна

### Проверка данных

```bash
python scripts/run_data_check.py
```

### Генерация признаков

```bash
python scripts/run_feature_engineering.py
```

### Обучение модели

```bash
python scripts/run_train.py
```

### Проверка дрейфа

```bash
python scripts/run_drift_check.py
```

### Генерация SHAP-объяснений

```bash
python scripts/run_explain.py
```

### Запуск benchmark-экспериментов

```bash
python scripts/run_benchmark.py
```

## Запуск API

```bash
python scripts/run_api.py
```

После запуска документация API будет доступна по адресу:

```text
http://localhost:8000/docs
```

![streamlit](https://github.com/MaxKots/HSE_AML_MLOps_pipeline/blob/main/.assets/1.jpg)

Доступные основные эндпоинты:

- `/health`
- `/ready`
- `/predict`
- `/predict_batch`

## Запуск dashboard

В отдельном терминале с активированным тем же `venv`:

```bash
python scripts/run_dashboard.py
```

После запуска dashboard будет доступен по адресу:

```text
http://localhost:8501
```

## Запуск тестов

```bash
pytest -q
```

## Запуск через Makefile

```bash
make data-check
make features
make train
make drift
make explain
make api
make dashboard
make benchmark
```

## Запуск через Docker Compose

Базовая контейнерная инфраструктура:

```bash
docker compose up -build
```

В контейнерном режиме могут быть доступны:

- MLflow: `http://localhost:5000`
- Airflow: `http://localhost:8080`
- API: `http://localhost:8000`
- Dashboard: `http://localhost:8501`

## Основные артефакты

После выполнения пайплайна сохраняются:

- обработанные датасеты в `data/processed/`;
- отчёты по дрейфу в `artifacts/reports/`;
- feature spec в `artifacts/metrics/feature_spec.yaml`;
- training summary в `artifacts/metrics/training_summary.json`;
- benchmark results в `artifacts/metrics/benchmark_results.csv`;
- model bundles в `artifacts/models/`;
- SHAP-артефакты в `artifacts/shap/`;
- prediction logs в `artifacts/predictions/`.

## Назначение основных scripts

- `run_data_check.py` — проверка загрузки и валидации данных;
- `run_feature_engineering.py` — построение engineered features;
- `run_train.py` — запуск training pipeline;
- `run_drift_check.py` — запуск drift monitoring;
- `run_explain.py` — генерация SHAP-объяснений;
- `run_api.py` — запуск FastAPI сервиса;
- `run_dashboard.py` — запуск Streamlit dashboard;
- `run_benchmark.py` — запуск серии экспериментов;
- `promote_model.py` — перевод версии модели в Production в MLflow Registry.

## Практические рекомендации

Рекомендуемый порядок локальной отладки:

1. `run_data_check.py`
2. `run_feature_engineering.py`
3. `run_train.py`
4. `run_drift_check.py`
5. `run_explain.py`
6. `run_api.py`
7. `run_dashboard.py`

Такой порядок позволяет последовательно проверить:
- корректность данных;
- корректность feature engineering;
- обучение и сохранение модели;
- мониторинг дрейфа;
- explainability;
- online serving и пользовательский интерфейс.

## Контекст проекта

Проект разработан в рамках магистерской диссертации по теме построения сквозного ML-пайплайна для выявления сомнительных финансовых операций. Реализация ориентирована на воспроизводимость, интерпретируемость и готовность к интеграции в MLOps-контур.
