SRC_DIR=src
TEST_DIR=testes

install:
	pip install -r ambiente/requirements.txt

run:
	python -m $(SRC_DIR).cli

test:
	pytest -v $(TEST_DIR)

coverage:
	pytest --cov=$(SRC_DIR) --cov-report=term-missing $(TEST_DIR)

clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage

version:
	python -m $(SRC_DIR).cli version
