import pandas as pd

from src.data.validators import DataValidator

def make_valid_dataframe() -> pd.DataFrame:
return pd.DataFrame(
{
"fraud_bool": [0, 1],
"income": [0.1, 0.9],
"name_email_similarity": [0.8, 0.2],
"prev_address_months_count": [12, 24],
"current_address_months_count": [36, 48],
"customer_age": [30, 40],
"days_since_request": [1, 2],
"intended_balcon_amount": [10.0, 20.0],
"payment_type": ["A", "B"],
"zip_count_4w": [5, 6],
"velocity_6h": [100, 200],
"velocity_24h": [300, 400],
"velocity_4w": [1000, 2000],
"bank_branch_count_8w": [1, 2],
"date_of_birth_distinct_emails_4w": [1, 1],
"employment_status": ["employed", "self"],
"credit_risk_score": [100, 200],
"email_is_free": [1, 0],
"housing_status": ["rent", "own"],
"phone_home_valid": [1, 0],
"phone_mobile_valid": [1, 1],
"bank_months_count": [12, 18],
"has_other_cards": [0, 1],
"proposed_credit_limit": [1000, 2000],
"foreign_request": [0, 1],
"source": ["web", "app"],
"session_length_in_minutes": [5.5, 8.2],
"device_os": ["ios", "android"],
"keep_alive_session": [1, 0],
"device_distinct_emails_8w": [1, 2],
"device_fraud_count": [0, 1],
"month": [1, 2],
}
)

def test_validator_accepts_valid_dataframe() -> None:
validator = DataValidator()
df = make_valid_dataframe()



report = validator.run_full_validation(df)

assert report.is_valid is True
assert report.row_count == 2

def test_validator_rejects_wrong_target_values() -> None:
validator = DataValidator()
df = make_valid_dataframe()
df["fraud_bool"] = [0, 2]



try:
    validator.run_full_validation(df)
    assert False, "Ожидалось исключение"
except ValueError as exc:
    assert "должна быть бинарной" in str(exc)

