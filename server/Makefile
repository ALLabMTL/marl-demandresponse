VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
CONFIG_FILE = $(DIR)
HOST = 0.0.0.0
PORT = 5000
DIR = ./app
CONFIG_FILE = $(DIR)/config.json
DOCKER_NAME = server

SHELL := /bin/bash
.ONESHELL:

.PHONY: help run setup test lint clean build run-docker

# Defines the default target that `make` will to try to make, or in the case of a phony target, execute the specified commands
# This target is executed whenever we just type `make`
.DEFAULT_GOAL = help

help:
	@echo "---------------HELP-----------------"
	@echo "To test the project type make test"
	@echo "To run the project type make run, all dependencies will be installed"
	@echo "To clean the project type make clean"
	@echo "To run the production version using docker, type make run-docker"
	@echo "To simply build the docker image, type make build"
	@echo "To run the linting tool, type make lint"
	@echo "------------------------------------"


run: setup
	VERBOSE=true $(PYTHON) $(DIR)/main.py

setup: requirements.txt
	python3 -m venv $(VENV)
	source $(VENV)/bin/activate
	$(PIP) install -r requirements.txt

test: setup
	$(PYTHON) -m pytest -s

lint: setup
	$(PYTHON) -m pylint $(DIR)
	$(PYTHON) -m autoflake --in-place --recursive --remove-all-unused-imports --remove-unused-variables $(DIR)
	$(PYTHON) -m isort --profile black $(DIR)
	$(PYTHON) -m black --target-version py38 $(DIR)

clean:
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf logs/*.log

# build:
# 	docker build . -t $(DOCKER_NAME):test --network host

# run-docker: build
# 	docker run -t --rm --network="host" --privileged $(DOCKER_NAME):test 
