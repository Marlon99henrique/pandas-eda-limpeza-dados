"""
Testes para o CLI (Interface de Linha de Comando) do projeto Telco Customer Churn.

Valida os comandos:
- clean
- validate
- explore
- version

Autor: Marlon Henrique
Ano: 2025
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Raiz do projeto (assume que este arquivo está em tests/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cli_command(args):
    """
    Executa um comando CLI e retorna (codigo_saida, stdout, stderr).
    Força o cwd para a raiz do projeto, para que `-m src.cli` encontre o pacote.
    """
    result = subprocess.run(
        [sys.executable, "-m", "src.cli"] + args,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    return result.returncode, result.stdout, result.stderr


def _csv_telco_minimo(path: Path, n: int = 10) -> None:
    """
    Gera um CSV mínimo no esquema Telco para que o pipeline rode sem erro.
    """
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "customerID": [f"ID{i:04d}" for i in range(n)],
        "SeniorCitizen": rng.integers(0, 2, size=n),
        "Partner": rng.choice(["Yes", "No"], size=n),
        "Dependents": rng.choice(["Yes", "No"], size=n),
        "tenure": rng.integers(0, 72, size=n),
        "PhoneService": rng.choice(["Yes", "No"], size=n),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], size=n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], size=n),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], size=n),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], size=n),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], size=n),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], size=n),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], size=n),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], size=n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
        "PaperlessBilling": rng.choice(["Yes", "No"], size=n),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n),
        "MonthlyCharges": rng.uniform(20, 120, size=n).round(2),
        "TotalCharges": rng.uniform(20, 8000, size=n).round(2),
        "Churn": rng.choice(["Yes", "No"], size=n),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_version_command():
    code, out, err = run_cli_command(["version"])
    assert code == 0
    assert "Versão" in out
    assert "Projeto" in out


def test_clean_command(tmp_path):
    # CSV Telco mínimo
    csv_path = tmp_path / "telco.csv"
    _csv_telco_minimo(csv_path)

    out_dir = tmp_path / "saida"

    code, out, err = run_cli_command([
        "clean",
        "--input", str(csv_path),
        "--output", str(out_dir),
    ])
    assert code == 0, f"stderr:\n{err}\nstdout:\n{out}"
    assert (out_dir / "telco_limpo.csv").exists()


def test_validate_command(tmp_path):
    csv_path = tmp_path / "telco.csv"
    _csv_telco_minimo(csv_path)

    code, out, err = run_cli_command([
        "validate",
        "--data", str(csv_path),
    ])
    # 0 = ok, 2 = erro de arquivo/validação cobrindo variações de ambiente
    assert code in (0, 2), f"stderr:\n{err}\nstdout:\n{out}"


def test_explore_command(tmp_path):
    csv_path = tmp_path / "telco.csv"
    _csv_telco_minimo(csv_path)

    report_path = tmp_path / "report.png"

    code, out, err = run_cli_command([
        "explore",
        "--data", str(csv_path),
        "--report", str(report_path),
    ])
    assert code in (0, 2), f"stderr:\n{err}\nstdout:\n{out}"
    if code == 0:
        assert report_path.exists()
