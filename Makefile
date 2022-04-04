ifneq (,$(wildcard ./.env))
    include .env
    export
endif

local-build-base:
	docker build -t nocturne-base --file=docker/Dockerfile.base .

local-build-dev: local-build-base
	docker build -t nocturne-dev --file=docker/Dockerfile.dev .

local-test: local-build-dev
	docker run -v $(PWD):/usr/src/project nocturne-dev python -m pytest tests/unit --cov --cov-report html --cov-report xml


build-ci:
	docker build -t nocturne-ci --file=docker/Dockerfile.ci .

ci-test:
	docker run -v $(PWD):/usr/src/project nocturne-ci /bin/bash -c \
		"pip install -e .;python -m pytest tests/unit --cov --cov-report html --cov-report xml"

