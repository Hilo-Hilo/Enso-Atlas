.PHONY: docker-build docker-up docker-down docker-logs docker-shell

COMPOSE_FILE ?= docker/docker-compose.yaml
COMPOSE ?= docker compose -f $(COMPOSE_FILE)

docker-build:
	$(COMPOSE) build

docker-up:
	$(COMPOSE) up -d --build

docker-down:
	$(COMPOSE) down

docker-logs:
	$(COMPOSE) logs -f

docker-shell:
	$(COMPOSE) exec enso-atlas bash
