local-build:
	docker build -t dbx_scalable_dl --file=Dockerfile.dev .

local-test: local-build
	docker run -v $(PWD):/app dbx_scalable_dl pytest tests/unit --cov --cov-report html --cov-report xml

ci-test:
	docker pull ghcr.io/renardeinside/dbx_scalable_dl:latest
	docker run -v $(PWD):/app ghcr.io/renardeinside/dbx_scalable_dl:latest pytest tests/unit --cov --cov-report html --cov-report xml


