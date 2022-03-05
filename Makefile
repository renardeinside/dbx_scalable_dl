build:
	docker build -t dbx_scalable_dl --file=Dockerfile.dev .

test: build
	docker run dbx_scalable_dl pytest tests/unit --cov