from __future__ import annotations

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    income: float
    name_email_similarity: float
    prev_address_months_count: int | float
    current_address_months_count: int | float
    customer_age: int
    days_since_request: int
    intended_balcon_amount: float
    payment_type: str
    zip_count_4w: int
    velocity_6h: float
    velocity_24h: float
    velocity_4w: float
    bank_branch_count_8w: int
    date_of_birth_distinct_emails_4w: int
    employment_status: str
    credit_risk_score: float
    email_is_free: int
    housing_status: str
    phone_home_valid: int
    phone_mobile_valid: int
    bank_months_count: int | float
    has_other_cards: int
    proposed_credit_limit: float
    foreign_request: int
    source: str
    session_length_in_minutes: float
    device_os: str
    keep_alive_session: int
    device_distinct_emails_8w: int
    device_fraud_count: int
    month: int


class PredictionFactor(BaseModel):
    feature: str
    shap_value: float
    abs_shap_value: float
    direction: str


class PredictionResponse(BaseModel):
    prediction_score: float
    prediction_label: int
    recommendation: str
    top_positive_factors: list[PredictionFactor]
    top_negative_factors: list[PredictionFactor]
    human_readable_reasons: list[str]


class BatchPredictionRequest(BaseModel):
    transactions: list[TransactionRequest] = Field(default_factory=list)


class BatchPredictionItemResponse(BaseModel):
    row_index: int
    prediction_score: float
    prediction_label: int
    recommendation: str
    top_positive_factors: list[PredictionFactor]
    top_negative_factors: list[PredictionFactor]
    human_readable_reasons: list[str]


class BatchPredictionResponse(BaseModel):
    items: list[BatchPredictionItemResponse]


class HealthResponse(BaseModel):
    status: str
    is_model_loaded: bool
    loaded_model_type: str | None = None
