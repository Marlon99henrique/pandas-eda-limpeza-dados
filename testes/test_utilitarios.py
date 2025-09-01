"""
Testes unitários para o módulo src/utils.py
Autor: Marlon Henrique
Ano: 2025
"""

from pathlib import Path
import os
import tempfile
import shutil

import pytest
import pandas as pd
import numpy as np

from src.utils import (
    configurar_ambiente_visualizacao,
    carregar_dados,
    salvar_dados,
    calcular_estatisticas_descritivas,
    gerar_resumo_dataset,
    verificar_duplicatas,
    dividir_dataset_temporal,
    carregar_configuracao,
    salvar_configuracao,
    criar_diretorios_projeto,
    tempo_execucao,
    amostrar_dataset,
)

# -------------------------
# Fixtures básicas
# -------------------------
@pytest.fixture
def dados_teste():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "nome": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "idade": [25, 30, 35, 40, 45],
        "cidade": ["SP", "RJ", "SP", "MG", "SP"],
        "salario": [5000.0, 6000.0, 7000.0, 8000.0, 9000.0],
        "data_contratacao": pd.date_range("2023-01-01", periods=5, freq="M"),
    })


@pytest.fixture
def dados_com_duplicatas():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 1, 2],
        "nome": ["Alice", "Bob", "Charlie", "David", "Eve", "Alice", "Bob"],
        "idade": [25, 30, 35, 40, 45, 25, 30],
    })


@pytest.fixture
def dados_com_ausentes():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "nome": ["Alice", "Bob", None, "David", "Eve"],
        "idade": [25, None, 35, 40, 45],
        "salario": [5000.0, 6000.0, None, 8000.0, 9000.0],
    })


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    try:
        yield d
    finally:
        shutil.rmtree(d)


# -------------------------
# Testes
# -------------------------
def test_configurar_ambiente_visualizacao():
    configurar_ambiente_visualizacao(estilo="ggplot", contexto="notebook")


def test_carregar_salvar_dados_csv(dados_teste, temp_dir):
    caminho = Path(temp_dir) / "dados" / "x.csv"
    salvar_dados(dados_teste, caminho, formato="csv")
    assert caminho.exists()
    df2 = carregar_dados(caminho, formato="csv")
    pd.testing.assert_frame_equal(dados_teste, df2)


def test_carregar_salvar_dados_parquet(dados_teste, temp_dir):
    # pula se não houver engine instalada
    try:
        import pyarrow  # noqa: F401
    except Exception:
        try:
            import fastparquet  # noqa: F401
        except Exception:
            pytest.skip("Sem engine parquet (pyarrow/fastparquet).")

    caminho = Path(temp_dir) / "dados" / "x.parquet"
    salvar_dados(dados_teste, caminho, formato="parquet")
    assert caminho.exists()
    df2 = carregar_dados(caminho, formato="parquet")
    pd.testing.assert_frame_equal(
        dados_teste.reset_index(drop=True),
        df2.reset_index(drop=True),
        check_dtype=False,
    )


def test_carregar_dados_arquivo_inexistente(temp_dir):
    with pytest.raises(FileNotFoundError):
        carregar_dados(Path(temp_dir) / "nao_existe.csv")


def test_calcular_estatisticas_descritivas(dados_teste):
    stats = calcular_estatisticas_descritivas(dados_teste)
    assert "geral" in stats and "ausentes" in stats and "numericas" in stats
    assert stats["geral"]["linhas"] == 5
    assert stats["geral"]["colunas"] == 6
    assert stats["ausentes"]["total_ausentes"] == 0
    assert "idade" in stats["numericas"]
    assert stats["numericas"]["idade"]["min"] == 25
    assert stats["numericas"]["idade"]["max"] == 45


def test_calcular_estatisticas_com_ausentes(dados_com_ausentes):
    stats = calcular_estatisticas_descritivas(dados_com_ausentes)
    assert stats["ausentes"]["total_ausentes"] == 3
    assert stats["ausentes"]["colunas_com_ausentes"] == 3
    assert stats["numericas"]["idade"]["ausentes"] == 1


def test_gerar_resumo_dataset_imprime(dados_teste, capsys):
    gerar_resumo_dataset(dados_teste, "Resumo Teste")
    out = capsys.readouterr().out
    assert "Resumo Teste" in out
    assert "Dimensões:" in out


def test_verificar_duplicatas_completo(dados_com_duplicatas):
    info = verificar_duplicatas(dados_com_duplicatas)
    assert info["duplicatas_completas"] == 2
    assert info["linhas_unicas"] == 5


def test_verificar_duplicatas_subset(dados_com_duplicatas):
    info = verificar_duplicatas(dados_com_duplicatas, subset=["id"])
    assert "duplicatas_subset" in info
    assert info["duplicatas_subset"] >= 2


def test_verificar_duplicatas_sem_duplicatas(dados_teste):
    info = verificar_duplicatas(dados_teste)
    assert info["duplicatas_completas"] == 0
    assert info["linhas_unicas"] == 5


def test_dividir_dataset_temporal(dados_teste):
    treino, teste = dividir_dataset_temporal(dados_teste.copy(), "data_contratacao", "2023-03-15")
    assert len(treino) + len(teste) == len(dados_teste)
    assert (treino["data_contratacao"] < pd.Timestamp("2023-03-15")).all()
    assert (teste["data_contratacao"] >= pd.Timestamp("2023-03-15")).all()


def test_carregar_salvar_configuracao(temp_dir):
    cfg = {"geral": {"versao": "0.1.0"}, "parametros": {"seed": 42}}
    caminho = Path(temp_dir) / "config" / "parametros.yaml"
    salvar_configuracao(cfg, caminho)
    lido = carregar_configuracao(caminho)
    assert lido == cfg


def test_carregar_configuracao_inexistente(temp_dir):
    caminho = Path(temp_dir) / "nao_existe.yaml"
    cfg = carregar_configuracao(caminho)
    assert cfg == {}


def test_criar_diretorios_projeto(tmp_path: Path):
    """
    A função cria pastas com prefixo '../'. Para cair dentro de tmp_path,
    mudamos o CWD para tmp_path/'workspace' antes de chamar.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    cwd_old = Path.cwd()
    try:
        os.chdir(workspace)
        criar_diretorios_projeto()
        # '../dados' relativo a 'workspace' => tmp_path/'dados'
        assert (tmp_path / "dados" / "brutos").exists()
        assert (tmp_path / "relatorios" / "figuras").exists()
    finally:
        os.chdir(cwd_old)


def test_tempo_execucao_decorator(caplog):
    @tempo_execucao
    def func_teste():
        import time
        time.sleep(0.05)
        return "ok"

    caplog.clear()
    result = func_teste()
    assert result == "ok"
    # decorator usa logging; checamos a mensagem
    assert any("executado em" in rec.message for rec in caplog.records)


def test_amostrar_dataset_proporcao(dados_teste):
    amostra = amostrar_dataset(dados_teste, tamanho=0.6, random_state=42)
    assert len(amostra) == int(len(dados_teste) * 0.6)


def test_amostrar_dataset_tamanho_fixo(dados_teste):
    amostra = amostrar_dataset(dados_teste, tamanho=3, random_state=42)
    assert len(amostra) == 3


def test_amostrar_dataset_estratificada(dados_teste):
    dados = dados_teste.copy()
    dados["categoria"] = ["A", "B", "A", "B", "A"]  # 60% / 40%
    amostra = amostrar_dataset(dados, tamanho=0.6, estratificar="categoria", random_state=42)
    cont = amostra["categoria"].value_counts(normalize=True)
    assert abs(cont.get("A", 0) - 0.6) < 0.25
    assert abs(cont.get("B", 0) - 0.4) < 0.25
