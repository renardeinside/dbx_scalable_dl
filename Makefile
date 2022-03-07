ifneq (,$(wildcard ./.env))
    include .env
    export
endif

local-build:
	docker build -t dbx_scalable_dl --file=Dockerfile.dev .

local-test: local-build
	docker run -v $(PWD):/app dbx_scalable_dl pytest tests/unit --cov --cov-report html --cov-report xml

ci-test:
	docker pull ghcr.io/renardeinside/dbx_scalable_dl:latest
	docker run -v $(PWD):/app ghcr.io/renardeinside/dbx_scalable_dl:latest pytest tests/unit --cov --cov-report html --cov-report xml


local-it-model-builder:
	dbx deploy --job=dbx-scalable-dl-model-builder-dev #--files-only
	dbx launch --job=dbx-scalable-dl-model-builder-dev --trace --as-run-submit

local-it-data-loader:
	dbx deploy --job=dbx-scalable-dl-data-loader-dev #--files-only
	dbx launch --job=dbx-scalable-dl-data-loader-dev --trace #--as-run-submit

deploy-cw-integration:
	databricks --profile=e2-demo-aws fs cp --recursive --overwrite scripts/cloud-watch/ dbfs:/init_scripts/cloud-watch/