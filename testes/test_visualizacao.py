"""
Testes unitários para o módulo src/visualizacao.py

Autor: Marlon Henrique
Ano: 2025
"""

import pytest
import pandas as pd
import numpy as np

# usar backend headless para evitar problemas em CI/sem GUI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
import tempfile
import shutil

# importa diretamente do pacote
from src.visualizacao import (
    configurar_estilo_graficos,
    plotar_distribuicoes_antes_depois,
    criar_heatmap_correlacao,
    plotar_valores_ausentes,
    visualizar_churn_por_categoria,
    criar_grafico_importancia_variaveis,
    plotar_boxplots_numericos,
    criar_grafico_interativo,
    salvar_grafico,
)

class TestVisualizacao:
    """Classe de testes para o módulo visualizacao"""

    @pytest.fixture
    def dados_teste(self):
        """Fixture para criar dados de teste para visualização."""
        np.random.seed(42)
        dados = {
            "idade": np.random.normal(35, 10, 100),
            "salario": np.random.normal(5000, 1500, 100),
            "tempo_empresa": np.random.randint(1, 20, 100),
            "cidade": np.random.choice(["SP", "RJ", "MG", "RS"], 100),
            "departamento": np.random.choice(["TI", "RH", "Vendas", "Marketing"], 100),
            "ativo": np.random.choice([True, False], 100),
            "churn": np.random.choice(["Sim", "Não"], 100, p=[0.3, 0.7]),
            "score": np.random.uniform(0, 1, 100),
        }
        return pd.DataFrame(dados)

    @pytest.fixture
    def dados_com_ausentes(self):
        """Fixture para criar dados com valores ausentes para teste."""
        dados = {
            "coluna_num": [1, 2, None, 4, 5, None, 7, 8, 9, 10],
            "coluna_cat": ["A", "B", None, "A", "B", "C", None, "A", "B", "C"],
            "coluna_num2": [None, 20, 30, 40, None, 60, 70, 80, 90, 100],
        }
        return pd.DataFrame(dados)

    @pytest.fixture
    def temp_dir(self):
        """Diretório temporário para testes de arquivos."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_configurar_estilo_graficos(self):
        configurar_estilo_graficos()
        assert plt.rcParams["figure.figsize"] == (12, 8)
        assert plt.rcParams["font.size"] == 12

    def test_plotar_distribuicoes_antes_depois(self, dados_teste):
        dados_antes = dados_teste.copy()
        dados_antes["idade"] = dados_antes["idade"] * 1.5
        fig = plotar_distribuicoes_antes_depois(
            dados_antes, dados_teste, ["idade", "salario", "tempo_empresa"], "Comparação Distribuições"
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 3
        plt.close(fig)

    def test_criar_heatmap_correlacao(self, dados_teste):
        fig = criar_heatmap_correlacao(dados_teste, metodo="pearson", annot=True, titulo="Mapa de Correlação Teste")
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_plotar_valores_ausentes_sem_ausentes(self, dados_teste):
        fig = plotar_valores_ausentes(dados_teste, "Análise de Valores Ausentes")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plotar_valores_ausentes_com_ausentes(self, dados_com_ausentes):
        fig = plotar_valores_ausentes(dados_com_ausentes, "Análise de Valores Ausentes")
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_visualizar_churn_por_categoria(self, dados_teste):
        fig = visualizar_churn_por_categoria(
            dados_teste, ["cidade", "departamento"], target="churn", titulo="Churn por Categoria"
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_criar_grafico_importancia_variaveis(self, dados_teste):
        fig = criar_grafico_importancia_variaveis(
            dados_teste, target="churn", metodo="mutual_info", top_n=5, titulo="Importância de Variáveis"
        )
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert len(ax.patches) > 0
        plt.close(fig)

    def test_plotar_boxplots_numericos(self, dados_teste):
        fig = plotar_boxplots_numericos(
            dados_teste, ["idade", "salario", "tempo_empresa"], target="churn", titulo="Boxplots por Churn"
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 3
        plt.close(fig)

    def test_criar_grafico_interativo(self, dados_teste):
        fig = criar_grafico_interativo(
            dados_teste, x_col="idade", y_col="salario", color_col="cidade", titulo="Gráfico Interativo Teste"
        )
        # é um objeto plotly.graph_objects.Figure
        assert hasattr(fig, "update_layout")

    def test_salvar_grafico_matplotlib(self, temp_dir):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.set_title("Gráfico Teste")
        caminho = Path(temp_dir) / "grafico_teste.png"
        salvar_grafico(fig, caminho, formato="png", dpi=100)
        assert caminho.exists() and caminho.stat().st_size > 0
        plt.close(fig)

    def test_salvar_grafico_plotly(self, dados_teste, temp_dir):
        fig = criar_grafico_interativo(dados_teste, x_col="idade", y_col="salario", titulo="Teste Plotly")
        caminho_html = Path(temp_dir) / "grafico_teste.html"
        salvar_grafico(fig, caminho_html, formato="html")
        assert caminho_html.exists() and caminho_html.stat().st_size > 0

    def test_salvar_grafico_formato_nao_suportado(self, temp_dir):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        caminho = Path(temp_dir) / "grafico_teste.xyz"
        with pytest.raises(Exception):
            salvar_grafico(fig, caminho, formato="xyz")
        plt.close(fig)

    def test_visualizacao_dados_vazios(self):
        df = pd.DataFrame()
        fig = plotar_valores_ausentes(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_visualizacao_dados_um_elemento(self):
        df = pd.DataFrame({"coluna": [1], "target": ["A"]})
        fig = plotar_valores_ausentes(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---- smoke tests avulsos (execução direta) ----
def test_configurar_estilo_graficos_smoke():
    configurar_estilo_graficos()

def test_plotar_distribuicoes_antes_depois_smoke():
    d1 = pd.DataFrame({"coluna": [1, 2, 3, 4, 5]})
    d2 = pd.DataFrame({"coluna": [2, 3, 4, 5, 6]})
    fig = plotar_distribuicoes_antes_depois(d1, d2, ["coluna"])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_criar_heatmap_correlacao_smoke():
    dados = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 3, 4, 5, 6], "z": [1, 1, 1, 1, 1]})
    fig = criar_heatmap_correlacao(dados)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plotar_valores_ausentes_smoke():
    dados = pd.DataFrame({"coluna": [1, None, 3, 4, None]})
    fig = plotar_valores_ausentes(dados)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_salvar_grafico_smoke():
    with tempfile.TemporaryDirectory() as td:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        caminho = Path(td) / "teste.png"
        salvar_grafico(fig, caminho)
        assert caminho.exists()
        plt.close(fig)
