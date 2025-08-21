"""
Testes unitários para o módulo visualizacao.py

Testes para garantir que as funções de visualização funcionam corretamente
e produzem os gráficos e visualizações esperados.

Autor: Marlon Henrique
Data: 2025
Versão: 1.0.0
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import tempfile
import shutil

# Adicionar o src ao path para importar os módulos
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from visualizacao import (
    configurar_estilo_graficos,
    plotar_distribuicoes_antes_depois,
    criar_heatmap_correlacao,
    plotar_valores_ausentes,
    visualizar_churn_por_categoria,
    criar_grafico_importancia_variaveis,
    plotar_boxplots_numericos,
    criar_grafico_interativo,
    salvar_grafico
)

class TestVisualizacao:
    """Classe de testes para o módulo visualizacao"""
    
    @pytest.fixture
    def dados_teste(self):
        """
        Fixture para criar dados de teste para visualização.
        """
        np.random.seed(42)
        dados = {
            'idade': np.random.normal(35, 10, 100),
            'salario': np.random.normal(5000, 1500, 100),
            'tempo_empresa': np.random.randint(1, 20, 100),
            'cidade': np.random.choice(['SP', 'RJ', 'MG', 'RS'], 100),
            'departamento': np.random.choice(['TI', 'RH', 'Vendas', 'Marketing'], 100),
            'ativo': np.random.choice([True, False], 100),
            'churn': np.random.choice(['Sim', 'Não'], 100, p=[0.3, 0.7]),
            'score': np.random.uniform(0, 1, 100)
        }
        return pd.DataFrame(dados)
    
    @pytest.fixture
    def dados_com_ausentes(self):
        """
        Fixture para criar dados com valores ausentes para teste.
        """
        dados = {
            'coluna_num': [1, 2, None, 4, 5, None, 7, 8, 9, 10],
            'coluna_cat': ['A', 'B', None, 'A', 'B', 'C', None, 'A', 'B', 'C'],
            'coluna_num2': [None, 20, 30, 40, None, 60, 70, 80, 90, 100]
        }
        return pd.DataFrame(dados)
    
    @pytest.fixture
    def temp_dir(self):
        """
        Fixture para criar diretório temporário para testes de arquivos.
        """
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)  # Limpar após o teste

    def test_configurar_estilo_graficos(self):
        """
        Testa se a configuração do estilo de gráficos funciona.
        """
        # Deve executar sem erros
        try:
            configurar_estilo_graficos()
            assert True
        except Exception as e:
            pytest.fail(f"configurar_estilo_graficos falhou: {e}")
        
        # Verificar se as configurações foram aplicadas
        assert plt.rcParams['figure.figsize'] == (12, 8)
        assert plt.rcParams['font.size'] == 12
        
        print("✅ configurar_estilo_graficos: Configura estilo corretamente")

    def test_plotar_distribuicoes_antes_depois(self, dados_teste):
        """
        Testa a plotagem de distribuições antes/depois.
        """
        # Criar dados "antes" com problemas
        dados_antes = dados_teste.copy()
        dados_antes['idade'] = dados_antes['idade'] * 1.5  # Alterar distribuição
        
        # Criar figura
        fig = plotar_distribuicoes_antes_depois(
            dados_antes, 
            dados_teste, 
            ['idade', 'salario', 'tempo_empresa'],
            "Comparação Distribuições"
        )
        
        # Verificar se a figura foi criada corretamente
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 3  # Deve ter pelo menos 3 subplots
        
        # Verificar se os eixos têm títulos
        for ax in fig.axes[:3]:  # Primeiros 3 eixos
            if ax.get_title():  # Se tem título
                assert 'Distribuição' in ax.get_title()
        
        plt.close(fig)
        print("✅ plotar_distribuicoes_antes_depois: Cria figura corretamente")

    def test_criar_heatmap_correlacao(self, dados_teste):
        """
        Testa a criação de heatmap de correlação.
        """
        fig = criar_heatmap_correlacao(
            dados_teste, 
            metodo='pearson',
            annot=True,
            titulo="Mapa de Correlação Teste"
        )
        
        # Verificar se a figura foi criada corretamente
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1  # Deve ter 1 axe
        
        # Verificar título
        ax = fig.axes[0]
        assert "Correlação" in ax.get_title()
        
        plt.close(fig)
        print("✅ criar_heatmap_correlacao: Cria heatmap corretamente")

    def test_plotar_valores_ausentes_sem_ausentes(self, dados_teste):
        """
        Testa a plotagem de valores ausentes quando não há ausentes.
        """
        fig = plotar_valores_ausentes(dados_teste, "Análise de Valores Ausentes")
        
        # Verificar se a figura foi criada
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)
        print("✅ plotar_valores_ausentes: Lida sem ausentes corretamente")

    def test_plotar_valores_ausentes_com_ausentes(self, dados_com_ausentes):
        """
        Testa a plotagem de valores ausentes quando há ausentes.
        """
        fig = plotar_valores_ausentes(dados_com_ausentes, "Análise de Valores Ausentes")
        
        # Verificar se a figura foi criada com múltiplos subplots
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 2  # Deve ter vários subplots
        
        plt.close(fig)
        print("✅ plotar_valores_ausentes: Visualiza ausentes corretamente")

    def test_visualizar_churn_por_categoria(self, dados_teste):
        """
        Testa a visualização de churn por categoria.
        """
        fig = visualizar_churn_por_categoria(
            dados_teste,
            ['cidade', 'departamento'],
            target='churn',
            titulo="Churn por Categoria"
        )
        
        # Verificar se a figura foi criada
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 2  # Deve ter vários subplots
        
        # Verificar se os eixos têm títulos relacionados a churn
        for ax in fig.axes:
            if ax.get_title() and hasattr(ax, 'get_title'):
                title = ax.get_title()
                if title:  # Verificar se não é string vazia
                    assert 'Churn' in title or 'Taxa' in title
        
        plt.close(fig)
        print("✅ visualizar_churn_por_categoria: Cria gráficos corretamente")

    def test_criar_grafico_importancia_variaveis(self, dados_teste):
        """
        Testa a criação de gráfico de importância de variáveis.
        """
        fig = criar_grafico_importancia_variaveis(
            dados_teste,
            target='churn',
            metodo='mutual_info',
            top_n=5,
            titulo="Importância de Variáveis"
        )
        
        # Verificar se a figura foi criada
        assert isinstance(fig, plt.Figure)
        
        # Verificar se é um gráfico de barras horizontais
        ax = fig.axes[0]
        assert len(ax.patches) > 0  # Deve ter barras
        
        plt.close(fig)
        print("✅ criar_grafico_importancia_variaveis: Cria gráfico corretamente")

    def test_plotar_boxplots_numericos(self, dados_teste):
        """
        Testa a criação de boxplots para variáveis numéricas.
        """
        fig = plotar_boxplots_numericos(
            dados_teste,
            ['idade', 'salario', 'tempo_empresa'],
            target='churn',
            titulo="Boxplots por Churn"
        )
        
        # Verificar se a figura foi criada
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 3  # Deve ter vários subplots
        
        plt.close(fig)
        print("✅ plotar_boxplots_numericos: Cria boxplots corretamente")

    def test_criar_grafico_interativo(self, dados_teste):
        """
        Testa a criação de gráfico interativo.
        """
        fig = criar_grafico_interativo(
            dados_teste,
            x_col='idade',
            y_col='salario',
            color_col='cidade',
            titulo="Gráfico Interativo Teste"
        )
        
        # Verificar se a figura foi criada
        assert hasattr(fig, 'update_layout')  # Verificar se é figura plotly
        
        print("✅ criar_grafico_interativo: Cria gráfico interativo corretamente")

    def test_salvar_grafico_matplotlib(self, dados_teste, temp_dir):
        """
        Testa o salvamento de gráfico matplotlib.
        """
        # Criar gráfico simples
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.set_title("Gráfico Teste")
        
        # Salvar gráfico
        caminho = Path(temp_dir) / "grafico_teste.png"
        salvar_grafico(fig, caminho, formato='png', dpi=100)
        
        # Verificar se arquivo foi criado
        assert caminho.exists()
        assert caminho.stat().st_size > 0
        
        plt.close(fig)
        print("✅ salvar_grafico: Salva gráfico matplotlib corretamente")

    def test_salvar_grafico_plotly(self, dados_teste, temp_dir):
        """
        Testa o salvamento de gráfico plotly.
        """
        # Criar gráfico plotly
        fig = criar_grafico_interativo(
            dados_teste,
            x_col='idade',
            y_col='salario',
            titulo="Teste Plotly"
        )
        
        # Salvar como HTML
        caminho_html = Path(temp_dir) / "grafico_teste.html"
        salvar_grafico(fig, caminho_html, formato='html')
        
        # Verificar se arquivo foi criado
        assert caminho_html.exists()
        assert caminho_html.stat().st_size > 0
        
        print("✅ salvar_grafico: Salva gráfico plotly corretamente")

    def test_salvar_grafico_formato_nao_suportado(self, dados_teste, temp_dir):
        """
        Testa comportamento ao tentar salvar em formato não suportado.
        """
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        caminho = Path(temp_dir) / "grafico_teste.xyz"
        
        # Deve levantar exceção ou falhar silenciosamente
        try:
            salvar_grafico(fig, caminho, formato='xyz')
            # Se não levantou exceção, verificar se arquivo não foi criado
            assert not caminho.exists()
        except Exception as e:
            # É aceitável levantar exceção para formato não suportado
            assert "formato" in str(e).lower() or "suportado" in str(e).lower()
        
        plt.close(fig)
        print("✅ salvar_grafico: Lida com formato não suportado corretamente")

    def test_visualizacao_dados_vazios(self):
        """
        Testa visualizações com DataFrame vazio.
        """
        dados_vazios = pd.DataFrame()
        
        # Testar cada função com dados vazios
        try:
            fig = plotar_valores_ausentes(dados_vazios)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
        except Exception as e:
            pytest.fail(f"plotar_valores_ausentes falhou com dados vazios: {e}")
        
        print("✅ Visualizações: Lidam com dados vazios corretamente")

    def test_visualizacao_dados_um_elemento(self):
        """
        Testa visualizações com DataFrame de um elemento.
        """
        dados_um = pd.DataFrame({'coluna': [1], 'target': ['A']})
        
        # Testar funções que devem funcionar com poucos dados
        try:
            fig = plotar_valores_ausentes(dados_um)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
        except Exception as e:
            pytest.fail(f"plotar_valores_ausentes falhou com um elemento: {e}")
        
        print("✅ Visualizações: Lidam com um elemento corretamente")

# Funções de teste individuais para facilitar execução específica
def test_configurar_estilo_graficos():
    """Teste individual para configurar_estilo_graficos"""
    configurar_estilo_graficos()
    print("✅ teste_configurar_estilo_graficos: Passou")

def test_plotar_distribuicoes_antes_depois():
    """Teste individual para plotar_distribuicoes_antes_depois"""
    dados_antes = pd.DataFrame({'coluna': [1, 2, 3, 4, 5]})
    dados_depois = pd.DataFrame({'coluna': [2, 3, 4, 5, 6]})
    fig = plotar_distribuicoes_antes_depois(dados_antes, dados_depois, ['coluna'])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    print("✅ teste_plotar_distribuicoes_antes_depois: Passou")

def test_criar_heatmap_correlacao():
    """Teste individual para criar_heatmap_correlacao"""
    dados = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [2, 3, 4, 5, 6],
        'z': [1, 1, 1, 1, 1]
    })
    fig = criar_heatmap_correlacao(dados)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    print("✅ teste_criar_heatmap_correlacao: Passou")

def test_plotar_valores_ausentes():
    """Teste individual para plotar_valores_ausentes"""
    dados = pd.DataFrame({'coluna': [1, None, 3, 4, None]})
    fig = plotar_valores_ausentes(dados)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    print("✅ teste_plotar_valores_ausentes: Passou")

def test_salvar_grafico():
    """Teste individual para salvar_grafico"""
    with tempfile.TemporaryDirectory() as temp_dir:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        caminho = Path(temp_dir) / "teste.png"
        salvar_grafico(fig, caminho)
        assert caminho.exists()
        plt.close(fig)
    print("✅ teste_salvar_grafico: Passou")

if __name__ == "__main__":
    # Executar testes individualmente
    test_configurar_estilo_graficos()
    test_plotar_distribuicoes_antes_depois()
    test_criar_heatmap_correlacao()
    test_plotar_valores_ausentes()
    test_salvar_grafico()
    
    print("\n🎉 Todos os testes de visualização passaram!")