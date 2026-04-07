import pandas as pd

from src.data.transformers import DataTransformer
from src.explainability.shap_explainer import AMLShapExplainer
from src.features.engineering import FeatureEngineer
from src.models.train import AMLModelTrainer

def make_training_dataframe(n_rows: int = 120) -> pd.DataFrame:
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

def test_shap_explanations_can_be_built() -> None:
raw_df = make_training_dataframe()



transformer = DataTransformer()
transformed_df = transformer.transform(raw_df)

feature_engineer = FeatureEngineer()
feature_result = feature_engineer.build_features(transformed_df)

trainer = AMLModelTrainer()
summary = trainer.train_and_select_best(
    df=feature_result.dataframe,
    categorical_columns=feature_result.categorical_columns,
    numerical_columns=feature_result.numerical_columns,
    candidate_models=["lightgbm"],
)

explainer = AMLShapExplainer(summary["best_bundle_path"])
explanations = explainer.explain_rows(
    feature_result.dataframe.head(5),
    row_indices=[0],
    top_k=3,
)

assert len(explanations) == 1
assert 0.0 <= explanations[0].prediction_score <= 1.0
assert len(explanations[0].raw_top_features) > 0

