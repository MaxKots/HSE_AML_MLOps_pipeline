import pandas as pd

from src.data.transformers import DataTransformer
from src.features.engineering import FeatureEngineer

def make_feature_dataframe() -> pd.DataFrame:
return pd.DataFrame(
{
"fraud_bool": [0, 1, 0],
"income": [0.1, 0.8, 0.5],
"name_email_similarity": [0.9, 0.2, 0.4],
"prev_address_months_count": [-1, 24, 12],
"current_address_months_count": [36, -1, 6],
"customer_age": [30, 40, 20],
"days_since_request": [1, 3, 2],
"intended_balcon_amount": [10.0, -1, 5.0],
"payment_type": ["A", "B", "A"],
"zip_count_4w": [5, 20, 7],
"velocity_6h": [100, 300, 120],
"velocity_24h": [200, 600, 180],
"velocity_4w": [800, 2000, 900],
"bank_branch_count_8w": [1, 2, 1],
"date_of_birth_distinct_emails_4w": [1, 2, 1],
"employment_status": ["employed", "self", "student"],
"credit_risk_score": [100, 300, 150],
"email_is_free": [1, 1, 0],
"housing_status": ["rent", "own", "rent"],
"phone_home_valid": [1, 0, 1],
"phone_mobile_valid": [1, 0, 1],
"bank_months_count": [12, -1, 3],
"has_other_cards": [0, 1, 0],
"proposed_credit_limit": [1000, 5000, 1500],
"foreign_request": [0, 1, 0],
"source": ["web", "app", "web"],
"session_length_in_minutes": [5.0, -1, 10.0],
"device_os": ["ios", "android", "ios"],
"keep_alive_session": [1, 0, 1],
"device_distinct_emails_8w": [1, 3, 2],
"device_fraud_count": [0, 2, 0],
"month": [1, 2, 1],
}
)

def test_feature_engineer_adds_expected_columns() -> None:
df = make_feature_dataframe()



transformer = DataTransformer()
transformed = transformer.transform(df)

feature_engineer = FeatureEngineer()
result = feature_engineer.build_features(transformed)

expected_columns = [
    "event_id",
    "event_time",
    "event_hour",
    "event_day_of_week",
    "credit_limit_to_income_ratio",
    "velocity_24h_to_6h_ratio",
    "income_x_credit_risk",
    "velocity_mean",
    "risk_low_similarity_free_email",
]

for column in expected_columns:
    assert column in result.dataframe.columns

def test_feature_engineer_returns_non_empty_feature_lists() -> None:
df = make_feature_dataframe()



transformer = DataTransformer()
transformed = transformer.transform(df)

feature_engineer = FeatureEngineer()
result = feature_engineer.build_features(transformed)

assert len(result.feature_columns) > 0
assert len(result.categorical_columns) > 0
assert len(result.numerical_columns) > 0
assert "fraud_bool" not in result.feature_columns

