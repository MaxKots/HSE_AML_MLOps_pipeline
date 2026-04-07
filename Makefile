.PHONY: install test data-check features train drift explain api dashboard promote benchmark

install:
	pip install -r requirements.txt

test:
	pytest -q

data-check:
	python scripts/run_data_check.py

features:
	python scripts/run_feature_engineering.py

train:
	python scripts/run_train.py

drift:
	python scripts/run_drift_check.py

explain:
	python scripts/run_explain.py

api:
	python scripts/run_api.py

dashboard:
	python scripts/run_dashboard.py

promote:
	python scripts/promote_model.py

benchmark:
	python scripts/run_benchmark.py
