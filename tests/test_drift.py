import pandas as pd

from src.data.transformers import DataTransformer
from src.features.engineering import FeatureEngineer
from src.monitoring.drift import AMLDriftDetector

def make_reference_dataframe(n_rows: int = 100) -> pd.DataFrame:
rows = []
for i in range(n_rows):
rows.append(
{
"fraud_bool": 1 if i % 10 == 0 else 0,
"income": 0.2,
"name_email_similarity": 0.9,
"prev_address_months_count": 24,
"current_address_months_count": 36,
"customer_age": 40,
"days_since_request": i % 30,
"intended_balcon_amount": 5.0,
"payment_type": "A",
"zip_count_4w": 5,
"velocity_6h": 100,
"velocity_24h": 200,
"velocity_4w": 800,
"bank_branch_count_8w": 3,
"date_of_birth_distinct_emails_4w": 1,
"employment_status": "employed",
"credit_risk_score": 200,
"email_is_free": 0,
"housing_status": "own",
"phone_home_valid": 1,
"phone_mobile_valid": 1,
"bank_months_count": 24,
"has_other_cards": 1,
"proposed_credit_limit": 1000,
"foreign_request": 0,
"source": "web",
"session_length_in_minutes": 10.0,
"device_os": "ios",
"keep_alive_session": 1,
"device_distinct_emails_8w": 1,
"device_fraud_count": 0,
"month": 1 + (i // 30),
}
)
return pd.DataFrame(rows)

def make_current_dataframe_with_drift(n_rows: int = 100) -> pd.DataFrame:
rows = []
for i in range(n_rows):
rows.append(
{
"fraud_bool": 1 if i % 4 == 0 else 0,
"income": 0.8,
"name_email_similarity": 0.2,
"prev_address_months_count": 2,
"current_address_months_count": 3,
"customer_age": 20,
"days_since_request": i % 30,
"intended_balcon_amount": 50.0,
"payment_type": "B",
"zip_count_4w": 60,
"velocity_6h": 900,
"velocity_24h": 1600,
"velocity_4w": 3200,
"bank_branch_count_8w": 1,
"date_of_birth_distinct_emails_4w": 3,
"employment_status": "self",
"credit_risk_score": 800,
"email_is_free": 1,
"housing_status": "rent",
"phone_home_valid": 0,
"phone_mobile_valid": 0,
"bank_months_count": 2,
"has_other_cards": 0,
"proposed_credit_limit": 8000,
"foreign_request": 1,
"source": "app",
"session_length_in_minutes": 1.0,
"device_os": "android",
"keep_alive_session": 0,
"device_distinct_emails_8w": 5,
"device_fraud_count": 3,
"month": 1 + (i // 30),
}
)
return pd.DataFrame(rows)

def test_drift_detector_finds_drift() -> None:
transformer = DataTransformer()
feature_engineer = FeatureEngineer()
drift_detector = AMLDriftDetector()



reference_raw = make_reference_dataframe()
current_raw = make_current_dataframe_with_drift()

reference_features = feature_engineer.build_features(transformer.transform(reference_raw))
current_features = feature_engineer.build_features(transformer.transform(current_raw))

result = drift_detector.detect_drift(
    reference_df=reference_features.dataframe,
    current_df=current_features.dataframe,
    report_name="test_drift_report",
)

assert result.total_columns_checked > 0
assert result.number_of_drifted_columns >= 1

