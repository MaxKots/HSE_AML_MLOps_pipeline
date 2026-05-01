# AML Pipeline

Сквозной MLOps-проект для выявления сомнительных финансовых операций.

Проект покрывает полный жизненный цикл ML-решения:
- загрузку и валидацию данных;
- feature engineering;
- обучение моделей;
- логирование экспериментов в MLflow;
- интерпретацию через SHAP;
- мониторинг дрейфа;
- REST API для инференса;
- Streamlit dashboard;
- оркестрацию пайплайнов через Apache Airflow.

## Стек

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

## Архитектура и общий процесс

![schema](https://github.com/MaxKots/HSE_AML_MLOps_pipeline/blob/main/.assets/schema.svg)

## Что делает проект

- загружает AML-датасеты из локальных файлов;
- проверяет корректность схемы и целевой колонки;
- строит дополнительные признаки для транзакционных данных;
- обучает модели LightGBM и XGBoost;
- сохраняет артефакты моделей и метрик;
- логирует эксперименты в MLflow;
- строит SHAP-объяснения;
- считает drift-отчёты;
- предоставляет online и batch scoring через API;
- визуализирует результаты в dashboard;
- запускает training workflow через Airflow DAG'и.

## Структура проекта

```text
config/        конфиги проекта
data/raw/      исходные датасеты
data/processed/обработанные данные
src/           основной код
scripts/       точки запуска
dags/          DAG'и Airflow
artifacts/     модели, метрики, отчёты, predictions, SHAP
logs/          логи сервисов и Airflow
docker/        docker-init и служебные файлы
```

## Входные данные

Для запуска необходимо положить в `data/raw/` следующие файлы:

- `Base.csv`
- `Variant I.csv`
- `Variant II.csv`
- `synthaml_alerts.csv`
- `synthaml_transactions.csv`

Целевая колонка для основного датасета:

- `fraud_bool`

## Экспериментальная база

В проекте используются две группы открытых наборов данных.

- `Base.csv`, `Variant I.csv`, `Variant II.csv` — табличные экспериментальные данные для обучения базовых моделей, анализа дрейфа и проверки воспроизводимости ML/MLOps-контура.
- `synthaml_alerts.csv` и `synthaml_transactions.csv` — AML-ориентированная экспериментальная база SynthAML, используемая для дополнительной проверки архитектуры на данных, связанных с задачами противодействия отмыванию доходов.

Для SynthAML исходные данные имеют двухтабличную структуру: таблицу алертов и таблицу транзакций. Перед обучением они агрегируются по идентификатору алерта в плоский набор признаков.

## Быстрый старт

Подробная инструкция находится в файле `docs/setup.md`.

Если нужен полный пошаговый гайд по:
- virtual environment;
- Docker Compose;
- Airflow;
- MLflow;
- проверке работы;
- очистке проекта;

смотри `docs/setup.md`.

## Локальный запуск через venv

Создание окружения:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Установка зависимостей:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Базовые команды:

```bash
python scripts/run_data_check.py
python scripts/run_feature_engineering.py
python scripts/run_train.py
python scripts/run_drift_check.py
python scripts/run_explain.py
python scripts/run_api.py
python scripts/run_dashboard.py
```

Запуск бенчмарка:
```
docker-compose exec api python scripts/run_benchmark.py
```

Бенчмарк сравнивает модели LightGBM и XGBoost на нескольких экспериментальных сценариях:

- базовые эксперименты на Base.csv;
- проверка устойчивости на Variant I.csv и Variant II.csv;
- дополнительная AML-ориентированная проверка на SynthAML.

Результаты сохраняются в artifacts/metrics/benchmark_results.csv и artifacts/metrics/benchmark_results.json

## Запуск через Docker Compose

Базовый запуск инфраструктуры:

```bash
docker-compose up -d
```

Если используется Airflow, в файле `.env` должна быть задана переменная:

```env
AIRFLOW_UID=1000
```

Если твой UID на Linux отличается, замени `1000` на результат команды:

```bash
id -u
```

Это нужно для корректной работы контейнеров Airflow с примонтированными локальными директориями и чтобы не возникали ошибки прав доступа.

Типовые адреса сервисов:

- MLflow: `http://localhost:5000`
- Airflow: `http://localhost:8080`
- API: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`

## Интерфейс dashboard

![streamlit](https://github.com/MaxKots/HSE_AML_MLOps_pipeline/blob/main/.assets/1.jpg)

## Оркестрация в Airflow

Airflow используется для запуска и мониторинга DAG'ов пайплайна, включая training workflow.

![airflow](https://github.com/MaxKots/HSE_AML_MLOps_pipeline/blob/main/.assets/2.jpg)

## Эксперименты в MLflow

MLflow используется для логирования запусков, параметров, метрик и артефактов моделей.

![mlflow](https://github.com/MaxKots/HSE_AML_MLOps_pipeline/blob/main/.assets/3.jpg)

## Основные артефакты

После выполнения пайплайна сохраняются:

- обработанные датасеты: `data/processed/`
- drift-отчёты: `artifacts/reports/`
- feature spec: `artifacts/metrics/feature_spec.yaml`
- training summary: `artifacts/metrics/training_summary.json`
- model bundles: `artifacts/models/`
- SHAP-артефакты: `artifacts/shap/`
- prediction logs: `artifacts/predictions/`

## Основные сценарии

Проверка данных:

```bash
python scripts/run_data_check.py
```

Обучение:

```bash
python scripts/run_train.py
```

Запуск API:

```bash
python scripts/run_api.py
```

Запуск dashboard:

```bash
python scripts/run_dashboard.py
```

Проверка Airflow DAG:

```bash
docker-compose exec airflow-webserver airflow dags list
```

## Полезные ссылки

- полный гайд по установке и запуску: `SETUP.md`
- API docs после запуска: `http://localhost:8000/docs`
- Airflow UI после запуска: `http://localhost:8080`
- MLflow UI после запуска: `http://localhost:5000`
- Dashboard после запуска: `http://localhost:8501`

## Очистка

Остановить контейнеры:

```bash
docker-compose down
```

Полная очистка вместе с volumes:

```bash
docker-compose down -v
```

Удаление локальных артефактов:

```bash
rm -rf artifacts/*
rm -rf data/processed/*
```
