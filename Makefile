.PHONY: up down seed train promote api test k8s-apply

up:
	docker compose up -d

down:
	docker compose down -v

seed:
	docker compose exec spark spark-submit /opt/data/prepare.py

train:
	docker compose exec spark spark-submit \
		--packages org.mlflow:mlflow-spark:2.13.0 \
		/opt/jobs/train_als.py

promote:
	docker compose exec spark python /opt/jobs/promote_model.py

api:
	docker compose up -d --build api

test:
	docker compose exec api pytest /app/tests -q

k8s-apply:
	kubectl apply -f serving/k8s/deployment.yaml
