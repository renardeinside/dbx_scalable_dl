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


model-builder-from-local:
	echo "Launching model builder with size $(size)"
	dbx deploy --job=dbx-scalable-dl-model-builder-$(size)x --files-only
	dbx launch --job=dbx-scalable-dl-model-builder-$(size)x --as-run-submit


data-loader-from-local:
	dbx deploy --job=dbx-scalable-dl-data-loader --files-only
	dbx launch --job=dbx-scalable-dl-data-loader --as-run-submit