install:
	python3 -m pip install -r requirements.txt

run:
	uvicorn app.main:app --reload

test:
	pytest -q

evaluate:
	EMBEDDING_PROVIDER=local-hash LLM_PROVIDER=extractive python scripts/evaluate.py

docker:
	docker build -t sentinelrag . && docker run --rm -p 8000:8000 sentinelrag

compose-up:
	docker compose up --build

compose-down:
	docker compose down -v
