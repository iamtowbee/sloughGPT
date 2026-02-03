.PHONY: help setup train sample chat api ui train-ui lint format test

help:
	@echo "Common commands:"
	@echo "  make setup      # install dependencies"
	@echo "  make train      # run training"
	@echo "  make sample     # run sampling"
	@echo "  make chat       # run chat"
	@echo "  make api        # run API server"
	@echo "  make train-ui   # run training UI"

setup:
	@bash setup.sh

train:
	@python train.py

sample:
	@python sample.py

chat:
	@python chat.py

api:
	@python api_server.py

train-ui:
	@python train_ui.py

lint:
	@ruff check .

format:
	@ruff format .

test:
	@pytest
