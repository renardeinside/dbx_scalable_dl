ifneq (,$(wildcard ./.env))
    include .env
    export
endif

local-build:
	docker build -t nocturne --file=Dockerfile.dev .

local-test: local-build
	docker run -v $(PWD):/app nocturne pytest tests/unit --cov --cov-report html --cov-report xml

ci-test:
	docker run -v $(PWD):/app ghcr.io/renardeinside/nocturne:ci pytest tests/unit --cov --cov-report html --cov-report xml
