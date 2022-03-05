build:
	docker build -t dbx_scalable_dl --file=Dockerfile.dev .

test: build
	docker run -v $(PWD):/app dbx_scalable_dl pytest tests/unit --cov --cov-report html --cov-report xml