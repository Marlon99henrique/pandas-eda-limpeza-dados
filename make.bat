@echo off
SET SRC_DIR=src
SET TEST_DIR=testes
SET VENV_DIR=.venv

REM Se o ambiente virtual não existir, cria
IF NOT EXIST %VENV_DIR% (
    echo Criando ambiente virtual em %VENV_DIR%...
    py -m venv %VENV_DIR%
)

REM Ativa o ambiente virtual
CALL %VENV_DIR%\Scripts\activate.bat

IF "%1"=="install" (
    pip install --upgrade pip setuptools wheel
    pip install -r ambiente\requirements.txt
) ELSE IF "%1"=="run" (
    python -m %SRC_DIR%.cli
) ELSE IF "%1"=="test" (
    pytest -v %TEST_DIR%
) ELSE IF "%1"=="coverage" (
    pytest --cov=%SRC_DIR% --cov-report=term-missing %TEST_DIR%
) ELSE IF "%1"=="clean" (
    rmdir /S /Q __pycache__ .pytest_cache htmlcov
    del /Q .coverage
) ELSE IF "%1"=="version" (
    python -m %SRC_DIR%.cli version
) ELSE (
    echo Opções disponíveis: install, run, test, coverage, clean, version
)
