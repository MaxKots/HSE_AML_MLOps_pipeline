import pandas as pd

from src.models.benchmark import AMLBenchmarkRunner

def make_dataset(n_rows: int = 120) -> pd.DataFrame:
rows = []
for i in range(n_rows):
fraud = 1 if i % 10 == 0 else 0
rows.append(
{
"fraud_bool": fraud,
"income": 0.9 if fraud else 0.2,
"name_email_similarity": 0.1 if fraud else 0.9,
"prev_address_months_count": -1 if fraud else 24,
"current_address_months_count": 2 if fraud else 36,
"customer_age": 20 if fraud else 40,
"days_since_request": i % 30,
"intended_balcon_amount": 50 if fraud else 5,
"payment_type": "B" if fraud else "A",
"zip_count_4w": 50 if fraud else 5,
"velocity_6h": 800 if fraud else 100,
"velocity_24h": 1500 if fraud else 200,
"velocity_4w": 3000 if fraud else 800,
"bank_branch_count_8w": 1 if fraud else 3,
"date_of_birth_distinct_emails_4w": 3 if fraud else 1,
"employment_status": "self" if fraud else "employed",
"credit_risk_score": 800 if fraud else 200,
"email_is_free": 1,
"housing_status": "rent" if fraud else "own",
"phone_home_valid": 0 if fraud else 1,
"phone_mobile_valid": 0 if fraud else 1,
"bank_months_count": 2 if fraud else 24,
"has_other_cards": 0 if fraud else 1,
"proposed_credit_limit": 8000 if fraud else 1000,
"foreign_request": 1 if fraud else 0,
"source": "app" if fraud else "web",
"session_length_in_minutes": 1 if fraud else 10,
"device_os": "android" if fraud else "ios",
"keep_alive_session": 0 if fraud else 1,
"device_distinct_emails_8w": 5 if fraud else 1,
"device_fraud_count": 3 if fraud else 0,
"month": 1 + (i // 30),
}
)
return pd.DataFrame(rows)

def test_benchmark_runner_returns_metrics() -> None:
runner = AMLBenchmarkRunner()



train_df = make_dataset()
test_df = make_dataset()

result = runner.run_single_experiment(
    experiment_name="test_experiment",
    train_df=train_df,
    test_df=test_df,
    model_type="lightgbm",
    use_feature_engineering=True,
)

assert result.roc_auc >= 0.5
assert result.pr_auc >= 0.0
assert result.train_rows > 0
assert result.test_rows > 0

