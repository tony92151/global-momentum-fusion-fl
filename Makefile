.ONESHELL:
ENV_NAME = gmfenv
PYTHON=./$(ENV_NAME)/bin/python3
PIP=./$(ENV_NAME)/bin/pip3

create_env: install_python_dep

create_python_env:
	@echo "Create gmfenv..."
	python3 -m venv $(ENV_NAME)
	@echo "New env: $(PYTHON)"

install_python_dep: create_python_env
	@echo "Install python dep..."
	@sleep 1
	$(PIP) install -r requirements.txt