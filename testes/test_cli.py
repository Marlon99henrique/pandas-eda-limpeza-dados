"""
Testes para o CLI (Interface de Linha de Comando) do projeto Telco Customer Churn.

Aqui validamos se os principais comandos da CLI executam corretamente:
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


def run_cli_command(args):
    """
    Executa um comando CLI e retorna (codigo_saida, stdout, stderr).
    """
    result = subprocess.run(
        [sys.executable, "-m", "src.cli"] + args,
        capture_output=True,
        text=True
    )
    return result.returncode, result.stdout, result.stderr


def test_version_command():
    code, out, err = run_cli_command(["version"])
    assert code == 0
    assert "Versão" in out
    assert "Projeto" in out


def test_clean_command(tmp_path):
    # cria CSV falso de entrada
    csv_path = tmp_path / "fake.csv"
    csv_path.write_text("col1,col2\n1,2\n3,4\n")

    out_dir = tmp_path / "saida"

    code, out, err = run_cli_command([
        "clean",
        "--input", str(csv_path),
        "--output", str(out_dir)
    ])
    assert code == 0
    assert (out_dir / "telco_limpo.csv").exists()


def test_validate_command(tmp_path):
    # cria CSV falso de entrada
    csv_path = tmp_path / "fake.csv"
    csv_path.write_text("col1,col2\n1,2\n3,4\n")

    code, out, err = run_cli_command([
        "validate",
        "--data", str(csv_path)
    ])
    assert code in (0, 2)  # 0 se validou, 2 se schema/validação falhar
    # Não precisa checar saída exata ainda


def test_explore_command(tmp_path):
    # cria CSV falso de entrada
    csv_path = tmp_path / "fake.csv"
    csv_path.write_text("col1,col2\n1,2\n3,4\n")

    report_path = tmp_path / "report.png"

    code, out, err = run_cli_command([
        "explore",
        "--data", str(csv_path),
        "--report", str(report_path)
    ])
    assert code in (0, 2)
    if code == 0:
        assert report_path.exists()
