"""
Testes unitários para o módulo utils.py

Testes para garantir que as funções utilitárias funcionam corretamente
e mantêm o comportamento esperado.

Autor: Marlon Henrique
Data: 2025
Versão: 1.0.0
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil
import yaml

# Adicionar o src ao path para importar os módulos
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils import (
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
    amostrar_dataset
)

class TestUtils:
    """Classe de testes para o módulo utils"""
    
    @pytest.fixture
    def dados_teste(self):
        """
        Fixture para criar dados de teste.
        """
        dados = {
            'id': [1, 2, 3, 4, 5],
            'nome': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'idade': [25, 30, 35, 40, 45],
            'cidade': ['SP', 'RJ', 'SP', 'MG', 'SP'],
            'salario': [5000.0, 6000.0, 7000.0, 8000.0, 9000.0],
            'data_contratacao': pd.date_range('2023-01-01', periods=5, freq='M')
        }
        return pd.DataFrame(dados)
    
    @pytest.fixture
    def dados_com_duplicatas(self):
        """
        Fixture para criar dados com duplicatas.
        """
        dados = {
            'id': [1, 2, 3, 4, 5, 1, 2],
            'nome': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Alice', 'Bob'],
            'idade': [25, 30, 35, 40, 45, 25, 30]
        }
        return pd.DataFrame(dados)
    
    @pytest.fixture
    def dados_com_ausentes(self):
        """
        Fixture para criar dados com valores ausentes.
        """
        dados = {
            'id': [1, 2, 3, 4, 5],
            'nome': ['Alice', 'Bob', None, 'David', 'Eve'],
            'idade': [25, None, 35, 40, 45],
            'salario': [5000.0, 6000.0, None, 8000.0, 9000.0]
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

    def test_configurar_ambiente_visualizacao(self):
        """
        Testa se a configuração do ambiente de visualização funciona.
        """
        # Deve executar sem erros
        try:
            configurar_ambiente_visualizacao()
            assert True
        except Exception as e:
            pytest.fail(f"configurar_ambiente_visualizacao falhou: {e}")
        
        print("✅ configurar_ambiente_visualizacao: Executa sem erros")

    def test_carregar_salvar_dados_csv(self, dados_teste, temp_dir):
        """
        Testa carregar e salvar dados em formato CSV.
        """
        caminho_csv = Path(temp_dir) / "test_data.csv"
        
        # Salvar dados
        salvar_dados(dados_teste, caminho_csv, formato='csv')
        assert caminho_csv.exists()
        
        # Carregar dados
        df_carregado = carregar_dados(caminho_csv, formato='csv')
        
        # Verificar se os dados são iguais
        pd.testing.assert_frame_equal(dados_teste, df_carregado)
        
        print("✅ carregar_dados/salvar_dados: CSV funciona corretamente")

    def test_carregar_salvar_dados_parquet(self, dados_teste, temp_dir):
        """
        Testa carregar e salvar dados em formato Parquet.
        """
        caminho_parquet = Path(temp_dir) / "test_data.parquet"
        
        # Salvar dados
        salvar_dados(dados_teste, caminho_parquet, formato='parquet')
        assert caminho_parquet.exists()
        
        # Carregar dados
        df_carregado = carregar_dados(caminho_parquet, formato='parquet')
        
        # Verificar se os dados são iguais (ignorando tipos de datetime)
        pd.testing.assert_frame_equal(
            dados_teste.reset_index(drop=True), 
            df_carregado.reset_index(drop=True),
            check_dtype=False
        )
        
        print("✅ carregar_dados/salvar_dados: Parquet funciona corretamente")

    def test_carregar_dados_arquivo_inexistente(self, temp_dir):
        """
        Testa comportamento ao tentar carregar arquivo inexistente.
        """
        caminho_inexistente = Path(temp_dir) / "nao_existe.csv"
        
        with pytest.raises(FileNotFoundError):
            carregar_dados(caminho_inexistente)
        
        print("✅ carregar_dados: Lança erro para arquivo inexistente")

    def test_calcular_estatisticas_descritivas(self, dados_teste):
        """
        Testa cálculo de estatísticas descritivas.
        """
        estatisticas = calcular_estatisticas_descritivas(dados_teste)
        
        # Verificar estrutura do resultado
        assert 'geral' in estatisticas
        assert 'ausentes' in estatisticas
        assert 'numericas' in estatisticas
        
        # Verificar valores específicos
        assert estatisticas['geral']['linhas'] == 5
        assert estatisticas['geral']['colunas'] == 6
        assert estatisticas['ausentes']['total_ausentes'] == 0
        
        # Verificar estatísticas numéricas
        assert 'idade' in estatisticas['numericas']
        assert estatisticas['numericas']['idade']['media'] == 35.0
        assert estatisticas['numericas']['idade']['min'] == 25
        assert estatisticas['numericas']['idade']['max'] == 45
        
        print("✅ calcular_estatisticas_descritivas: Calcula estatísticas corretamente")

    def test_calcular_estatisticas_com_ausentes(self, dados_com_ausentes):
        """
        Testa cálculo de estatísticas com dados ausentes.
        """
        estatisticas = calcular_estatisticas_descritivas(dados_com_ausentes)
        
        # Verificar detecção de valores ausentes
        assert estatisticas['ausentes']['total_ausentes'] == 3
        assert estatisticas['ausentes']['colunas_com_ausentes'] == 3
        
        # Verificar que estatísticas são calculadas apesar de ausentes
        assert 'idade' in estatisticas['numericas']
        assert estatisticas['numericas']['idade']['ausentes'] == 1
        
        print("✅ calcular_estatisticas_descritivas: Lida com valores ausentes corretamente")

    def test_gerar_resumo_dataset(self, dados_teste, capsys):
        """
        Testa geração de resumo do dataset.
        """
        # Deve executar sem erros e produzir output
        gerar_resumo_dataset(dados_teste, "Teste Resumo")
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Verificar se o resumo contém informações esperadas
        assert "Dimensões:" in output
        assert "5 linhas" in output
        assert "6 colunas" in output
        assert "Tipos de dados:" in output
        
        print("✅ gerar_resumo_dataset: Gera resumo corretamente")

    def test_verificar_duplicatas_completo(self, dados_com_duplicatas):
        """
        Testa verificação de duplicatas completas.
        """
        resultado = verificar_duplicatas(dados_com_duplicatas)
        
        # Verificar resultados
        assert resultado['duplicatas_completas'] == 2
        assert resultado['percentual_completas'] == (2 / 7 * 100)
        assert resultado['linhas_unicas'] == 5
        
        print("✅ verificar_duplicatas: Detecta duplicatas completas corretamente")

    def test_verificar_duplicatas_subset(self, dados_com_duplicatas):
        """
        Testa verificação de duplicatas em subset de colunas.
        """
        resultado = verificar_duplicatas(dados_com_duplicatas, subset=['id'])
        
        # Verificar resultados
        assert resultado['duplicatas_subset'] == 2
        assert resultado['subset'] == ['id']
        
        print("✅ verificar_duplicatas: Detecta duplicatas em subset corretamente")

    def test_verificar_duplicatas_sem_duplicatas(self, dados_teste):
        """
        Testa verificação de duplicatas quando não há duplicatas.
        """
        resultado = verificar_duplicatas(dados_teste)
        
        # Verificar resultados
        assert resultado['duplicatas_completas'] == 0
        assert resultado['percentual_completas'] == 0.0
        assert resultado['linhas_unicas'] == 5
        
        print("✅ verificar_duplicatas: Lida sem duplicatas corretamente")

    def test_dividir_dataset_temporal(self, dados_teste):
        """
        Testa divisão temporal do dataset.
        """
        treino, teste = dividir_dataset_temporal(
            dados_teste, 
            'data_contratacao', 
            '2023-03-15'
        )
        
        # Verificar divisão
        assert len(treino) > 0
        assert len(teste) > 0
        assert len(treino) + len(teste) == len(dados_teste)
        
        # Verificar que todas as datas de treino são anteriores
        assert all(treino['data_contratacao'] < pd.Timestamp('2023-03-15'))
        
        # Verificar que todas as datas de teste são posteriores ou iguais
        assert all(teste['data_contratacao'] >= pd.Timestamp('2023-03-15'))
        
        print("✅ dividir_dataset_temporal: Divide dados temporalmente corretamente")

    def test_carregar_salvar_configuracao(self, temp_dir):
        """
        Testa carregar e salvar configuração YAML.
        """
        config_test = {
            'parametros': {
                'limite_ausentes': 5.0,
                'seed': 42,
                'colunas_numericas': ['idade', 'salario']
            },
            'modelo': {
                'n_estimators': 100,
                'max_depth': 10
            }
        }
        
        caminho_config = Path(temp_dir) / "config_test.yaml"
        
        # Salvar configuração
        salvar_configuracao(config_test, caminho_config)
        assert caminho_config.exists()
        
        # Carregar configuração
        config_carregada = carregar_configuracao(caminho_config)
        
        # Verificar se configurações são iguais
        assert config_carregada == config_test
        
        print("✅ carregar_configuracao/salvar_configuracao: YAML funciona corretamente")

    def test_carregar_configuracao_inexistente(self, temp_dir):
        """
        Testa carregar configuração de arquivo inexistente.
        """
        caminho_inexistente = Path(temp_dir) / "nao_existe.yaml"
        
        config = carregar_configuracao(caminho_inexistente)
        
        # Deve retornar dicionário vazio
        assert config == {}
        
        print("✅ carregar_configuracao: Retorna vazio para arquivo inexistente")

    def test_criar_diretorios_projeto(self, temp_dir):
        """
        Testa criação de diretórios do projeto.
        """
        # Mudar para diretório temporário
        original_dir = Path.cwd()
        try:
            os.chdir(temp_dir)
            
            # Criar diretórios
            criar_diretorios_projeto()
            
            # Verificar se diretórios foram criados
            diretorios_esperados = [
                'dados/brutos',
                'dados/processados',
                'dados/externos',
                'notebooks',
                'src',
                'testes',
                'docs',
                'relatorios/figuras',
                'ambiente',
                'config'
            ]
            
            for diretorio in diretorios_esperados:
                assert (Path(temp_dir) / diretorio).exists(), f"Diretório {diretorio} não criado"
                
        finally:
            os.chdir(original_dir)
        
        print("✅ criar_diretorios_projeto: Cria estrutura de diretórios corretamente")

    def test_tempo_execucao_decorator(self, capsys):
        """
        Testa o decorator de tempo de execução.
        """
        @tempo_execucao
        def funcao_teste():
            import time
            time.sleep(0.1)  # 100ms
            return "sucesso"
        
        # Executar função decorada
        resultado = funcao_teste()
        
        # Verificar resultado e output
        assert resultado == "sucesso"
        
        captured = capsys.readouterr()
        output = captured.out
        assert "funcao_teste" in output
        assert "segundos" in output
        
        print("✅ tempo_execucao: Decorator funciona corretamente")

    def test_amostrar_dataset_proporcao(self, dados_teste):
        """
        Testa amostragem por proporção.
        """
        amostra = amostrar_dataset(dados_teste, tamanho=0.6, random_state=42)
        
        # Verificar tamanho da amostra
        assert len(amostra) == 3  # 60% de 5 = 3
        assert len(amostra) / len(dados_teste) == 0.6
        
        print("✅ amostrar_dataset: Amostragem por proporção funciona corretamente")

    def test_amostrar_dataset_tamanho_fixo(self, dados_teste):
        """
        Testa amostragem por tamanho fixo.
        """
        amostra = amostrar_dataset(dados_teste, tamanho=3, random_state=42)
        
        # Verificar tamanho da amostra
        assert len(amostra) == 3
        
        print("✅ amostrar_dataset: Amostragem por tamanho fixo funciona corretamente")

    def test_amostrar_dataset_estratificada(self, dados_teste):
        """
        Testa amostragem estratificada.
        """
        # Adicionar coluna categórica para estratificação
        dados_teste['categoria'] = ['A', 'B', 'A', 'B', 'A']
        
        amostra = amostrar_dataset(
            dados_teste, 
            tamanho=0.6, 
            estratificar='categoria',
            random_state=42
        )
        
        # Verificar que a amostra mantém proporções
        contagem_categorias = amostra['categoria'].value_counts()
        proporcao_a = contagem_categorias.get('A', 0) / len(amostra)
        proporcao_b = contagem_categorias.get('B', 0) / len(amostra)
        
        # Proporções devem ser similares às originais (3A:2B = 60%:40%)
        assert abs(proporcao_a - 0.6) < 0.2  # Tolerância de 20%
        assert abs(proporcao_b - 0.4) < 0.2  # Tolerância de 20%
        
        print("✅ amostrar_dataset: Amostragem estratificada funciona corretamente")

# Funções de teste individuais para facilitar execução específica
def test_configurar_ambiente_visualizacao():
    """Teste individual para configurar_ambiente_visualizacao"""
    configurar_ambiente_visualizacao()
    print("✅ teste_configurar_ambiente_visualizacao: Passou")

def test_calcular_estatisticas_descritivas():
    """Teste individual para calcular_estatisticas_descritivas"""
    dados = pd.DataFrame({'coluna': [1, 2, 3, 4, 5]})
    stats = calcular_estatisticas_descritivas(dados)
    assert stats['geral']['linhas'] == 5
    print("✅ teste_calcular_estatisticas_descritivas: Passou")

def test_verificar_duplicatas():
    """Teste individual para verificar_duplicatas"""
    dados = pd.DataFrame({'id': [1, 2, 3, 1, 2]})
    resultado = verificar_duplicatas(dados)
    assert resultado['duplicatas_completas'] == 2
    print("✅ teste_verificar_duplicatas: Passou")

def test_criar_diretorios_projeto():
    """Teste individual para criar_diretorios_projeto"""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        try:
            criar_diretorios_projeto()
            assert Path('dados/brutos').exists()
        finally:
            os.chdir(original_dir)
    print("✅ teste_criar_diretorios_projeto: Passou")

def test_tempo_execucao():
    """Teste individual para tempo_execucao"""
    @tempo_execucao
    def funcao_rapida():
        return "ok"
    
    assert funcao_rapida() == "ok"
    print("✅ teste_tempo_execucao: Passou")

def test_amostrar_dataset():
    """Teste individual para amostrar_dataset"""
    dados = pd.DataFrame({'x': range(10), 'y': range(10)})
    amostra = amostrar_dataset(dados, tamanho=5, random_state=42)
    assert len(amostra) == 5
    print("✅ teste_amostrar_dataset: Passou")

if __name__ == "__main__":
    # Executar testes individualmente
    test_configurar_ambiente_visualizacao()
    test_calcular_estatisticas_descritivas()
    test_verificar_duplicatas()
    test_criar_diretorios_projeto()
    test_tempo_execucao()
    test_amostrar_dataset()
    
    print("\n🎉 Todos os testes de utils passaram!")