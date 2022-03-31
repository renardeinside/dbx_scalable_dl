ifneq (,$(wildcard ./.env))
    include .env
    export
endif

local-build-base:
	docker build -t nocturne-base --file=docker/Dockerfile.base .

#local-test: local-build
#	docker run -v $(PWD):/usr/src/project nocturne pytest tests/unit #--cov --cov-report html --cov-report xml
#
#ci-test:
#	docker run -v $(PWD):/usr/src/project ghcr.io/renardeinside/nocturne:ci pytest tests/unit --cov --cov-report html --cov-report xml
