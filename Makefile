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


# local here means that these actions are handly to launch from local machine
local-model-builder-sn:
	dbx deploy --job=dbx-scalable-dl-model-builder-sn --files-only
	dbx launch --job=dbx-scalable-dl-model-builder-sn --as-run-submit

local-model-builder-mn:
	dbx deploy --job=dbx-scalable-dl-model-builder-mn --files-only
	dbx launch --job=dbx-scalable-dl-model-builder-mn --as-run-submit

local-data-loader:
	dbx deploy --job=dbx-scalable-dl-data-loader --files-only
	dbx launch --job=dbx-scalable-dl-data-loader --as-run-submit