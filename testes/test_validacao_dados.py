"""
Testes unitários para o módulo validacao_dados.py

Testes para garantir que as funções de validação e controle de qualidade
funcionam corretamente e detectam problemas apropriadamente.

Autor: Marlon Henrique
Data: 2025
Versão: 1.0.0
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Adicionar o src ao path para importar os módulos
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from validacao_dados import (
    ValidadorDados,
    validar_estrutura_dataset,
    verificar_valores_ausentes,
    validar_tipos_dados,
    verificar_consistencia_categorica,
    gerar_relatorio_validacao,
    TipoValidacao,
    ResultadoValidacao
)

class TestValidacaoDados:
    """Classe de testes para o módulo validacao_dados"""
    
    @pytest.fixture
    def dados_validos(self):
        """
        Fixture para criar dados válidos de referência.
        """
        dados = {
            'id': [1, 2, 3, 4, 5],
            'nome': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'idade': [25, 30, 35, 40, 45],
            'cidade': ['SP', 'RJ', 'SP', 'MG', 'SP'],
            'ativo': [True, True, False, True, False],
            'data_criacao': pd.date_range('2023-01-01', periods=5, freq='D')
        }
        return pd.DataFrame(dados)
    
    @pytest.fixture
    def dados_com_problemas(self):
        """
        Fixture para criar dados com vários problemas.
        """
        dados = {
            'id': [1, 2, 3, 4, 5, 6],  # Coluna extra
            'nome': ['Alice', 'Bob', None, 'David', 'Eve', 'Frank'],  # Ausente
            'idade': ['25', '30', '35', '40', '45', '60'],  # String instead of int
            'cidade': ['SP', 'RJ', 'SP', 'MG', 'XX', 'SP'],  # Valor inválido
            'salario': [5000, 6000, 7000, 8000, 9000, -1000],  # Valor negativo
            'status': ['Ativo', 'Inativo', 'Ativo', 'Pendente', 'Ativo', 'Ativo']  # Valor inválido
        }
        return pd.DataFrame(dados)
    
    @pytest.fixture
    def dados_tipos_incorretos(self):
        """
        Fixture para criar dados com tipos incorretos.
        """
        dados = {
            'id': ['1', '2', '3', '4', '5'],  # Deveria ser numérico
            'idade': [25, 30, 35, 40, 45],  # Correto
            'preco': ['10.5', '20.3', '15.7', '25.1', '30.9'],  # Deveria ser float
            'ativo': ['True', 'False', 'True', 'False', 'True']  # Deveria ser booleano
        }
        return pd.DataFrame(dados)

    def test_validador_inicializacao(self, dados_validos):
        """
        Testa a inicialização do ValidadorDados.
        """
        validador = ValidadorDados(dados_validos, "Test Dataset")
        
        assert validador.df.shape == dados_validos.shape
        assert validador.nome_dataset == "Test Dataset"
        assert len(validador.resultados) == 0
        
        print("✅ ValidadorDados: Inicialização correta")

    def test_validar_estrutura_valida(self, dados_validos):
        """
        Testa validação de estrutura com dados válidos.
        """
        validador = ValidadorDados(dados_validos)
        colunas_esperadas = ['id', 'nome', 'idade', 'cidade', 'ativo', 'data_criacao']
        
        resultado = validador.validar_estrutura(colunas_esperadas, min_linhas=3)
        
        assert resultado.sucesso == True
        assert "válida" in resultado.mensagem.lower()
        assert resultado.tipo == TipoValidacao.ESTRUTURA
        
        print("✅ validar_estrutura: Dados válidos passam na validação")

    def test_validar_estrutura_faltando_colunas(self, dados_validos):
        """
        Testa validação de estrutura com colunas faltando.
        """
        validador = ValidadorDados(dados_validos)
        colunas_esperadas = ['id', 'nome', 'idade', 'cidade', 'ativo', 'data_criacao', 'coluna_inexistente']
        
        resultado = validador.validar_estrutura(colunas_esperadas)
        
        assert resultado.sucesso == False
        assert "faltantes" in resultado.mensagem.lower()
        assert 'coluna_inexistente' in resultado.mensagem
        
        print("✅ validar_estrutura: Detecta colunas faltantes")

    def test_validar_estrutura_poucas_linhas(self, dados_validos):
        """
        Testa validação de estrutura com poucas linhas.
        """
        validador = ValidadorDados(dados_validos)
        colunas_esperadas = ['id', 'nome', 'idade', 'cidade', 'ativo', 'data_criacao']
        
        resultado = validador.validar_estrutura(colunas_esperadas, min_linhas=10)
        
        assert resultado.sucesso == False
        assert "linhas" in resultado.mensagem.lower()
        
        print("✅ validar_estrutura: Detecta poucas linhas")

    def test_validar_tipos_corretos(self, dados_validos):
        """
        Testa validação de tipos com tipos corretos.
        """
        validador = ValidadorDados(dados_validos)
        mapeamento_tipos = {
            'id': 'int',
            'nome': 'object',
            'idade': 'int',
            'cidade': 'object',
            'ativo': 'bool'
        }
        
        resultado = validador.validar_tipos(mapeamento_tipos)
        
        assert resultado.sucesso == True
        assert resultado.tipo == TipoValidacao.TIPOS
        
        print("✅ validar_tipos: Tipos corretos passam na validação")

    def test_validar_tipos_incorretos(self, dados_tipos_incorretos):
        """
        Testa validação de tipos com tipos incorretos.
        """
        validador = ValidadorDados(dados_tipos_incorretos)
        mapeamento_tipos = {
            'id': 'int',      # Esperado: int, Encontrado: object
            'idade': 'int',   # Correto
            'preco': 'float', # Esperado: float, Encontrado: object
            'ativo': 'bool'   # Esperado: bool, Encontrado: object
        }
        
        resultado = validador.validar_tipos(mapeamento_tipos)
        
        assert resultado.sucesso == False
        assert len(resultado.detalhes['tipos_incorretos']) == 3
        
        print("✅ validar_tipos: Detecta tipos incorretos")

    def test_verificar_valores_ausentes_sem_ausentes(self, dados_validos):
        """
        Testa verificação de valores ausentes quando não há ausentes.
        """
        validador = ValidadorDados(dados_validos)
        
        resultado = validador.verificar_valores_ausentes()
        
        assert resultado.sucesso == True
        assert resultado.detalhes['total_ausentes'] == 0
        assert resultado.tipo == TipoValidacao.AUSENTES
        
        print("✅ verificar_valores_ausentes: Sem ausentes passa na validação")

    def test_verificar_valores_ausentes_com_ausentes(self, dados_com_problemas):
        """
        Testa verificação de valores ausentes quando há ausentes.
        """
        validador = ValidadorDados(dados_com_problemas)
        
        resultado = validador.verificar_valores_ausentes()
        
        assert resultado.sucesso == False
        assert resultado.detalhes['total_ausentes'] > 0
        assert 'nome' in resultado.detalhes['ausentes_por_coluna']
        
        print("✅ verificar_valores_ausentes: Detecta valores ausentes")

    def test_verificar_valores_ausentes_limite_percentual(self, dados_com_problemas):
        """
        Testa verificação de valores ausentes com limite percentual.
        """
        validador = ValidadorDados(dados_com_problemas)
        
        resultado = validador.verificar_valores_ausentes(limite_percentual=10.0)
        
        # Verificar se detecta colunas que excedem o limite
        colunas_com_excesso = [
            col for col, info in resultado.detalhes['ausentes_por_coluna'].items()
            if info['excede_limite']
        ]
        
        assert len(colunas_com_excesso) > 0
        
        print("✅ verificar_valores_ausentes: Respeita limite percentual")

    def test_verificar_consistencia_categorica_valida(self, dados_validos):
        """
        Testa verificação de consistência categórica com dados válidos.
        """
        validador = ValidadorDados(dados_validos)
        valores_esperados = ['SP', 'RJ', 'MG']
        
        resultado = validador.verificar_consistencia_categorica('cidade', valores_esperados)
        
        assert resultado.sucesso == True
        assert resultado.tipo == TipoValidacao.CATEGORIAS
        
        print("✅ verificar_consistencia_categorica: Dados válidos passam")

    def test_verificar_consistencia_categorica_invalida(self, dados_com_problemas):
        """
        Testa verificação de consistência categórica com dados inválidos.
        """
        validador = ValidadorDados(dados_com_problemas)
        valores_esperados = ['SP', 'RJ', 'MG']
        
        resultado = validador.verificar_consistencia_categorica('cidade', valores_esperados)
        
        assert resultado.sucesso == False
        assert 'XX' in str(resultado.detalhes['valores_inesperados'])
        
        print("✅ verificar_consistencia_categorica: Detecta valores inesperados")

    def test_verificar_consistencia_categorica_coluna_inexistente(self, dados_validos):
        """
        Testa verificação de consistência categórica com coluna inexistente.
        """
        validador = ValidadorDados(dados_validos)
        valores_esperados = ['A', 'B', 'C']
        
        resultado = validador.verificar_consistencia_categorica('coluna_inexistente', valores_esperados)
        
        assert resultado.sucesso == False
        assert "não encontrada" in resultado.mensagem.lower()
        
        print("✅ verificar_consistencia_categorica: Lida com coluna inexistente")

    def test_validar_range_numerico_valido(self, dados_validos):
        """
        Testa validação de range numérico com dados válidos.
        """
        validador = ValidadorDados(dados_validos)
        
        resultado = validador.validar_range_numerico('idade', min_val=18, max_val=100)
        
        assert resultado.sucesso == True
        assert resultado.tipo == TipoValidacao.RANGE
        
        print("✅ validar_range_numerico: Dados válidos passam")

    def test_validar_range_numerico_invalido(self, dados_com_problemas):
        """
        Testa validação de range numérico com dados inválidos.
        """
        validador = ValidadorDados(dados_com_problemas)
        
        resultado = validador.validar_range_numerico('salario', min_val=0)
        
        assert resultado.sucesso == False
        assert resultado.detalhes['valores_fora_range'] > 0
        
        print("✅ validar_range_numerico: Detecta valores fora do range")

    def test_validar_range_numerico_coluna_nao_numerica(self, dados_validos):
        """
        Testa validação de range numérico com coluna não numérica.
        """
        validador = ValidadorDados(dados_validos)
        
        resultado = validador.validar_range_numerico('nome', min_val=0, max_val=100)
        
        assert resultado.sucesso == False
        assert "não é numérica" in resultado.mensagem.lower()
        
        print("✅ validar_range_numerico: Lida com coluna não numérica")

    def test_verificar_duplicatas_sem_duplicatas(self, dados_validos):
        """
        Testa verificação de duplicatas sem duplicatas.
        """
        validador = ValidadorDados(dados_validos)
        
        resultado = validador.verificar_duplicatas()
        
        assert resultado.sucesso == True
        assert resultado.detalhes['duplicatas'] == 0
        assert resultado.tipo == TipoValidacao.DUPLICATAS
        
        print("✅ verificar_duplicatas: Sem duplicatas passa")

    def test_verificar_duplicatas_com_duplicatas(self, dados_com_problemas):
        """
        Testa verificação de duplicatas com duplicatas.
        """
        # Adicionar duplicatas
        dados_duplicados = pd.concat([dados_com_problemas, dados_com_problemas.head(2)])
        validador = ValidadorDados(dados_duplicados)
        
        resultado = validador.verificar_duplicatas()
        
        assert resultado.sucesso == False
        assert resultado.detalhes['duplicatas'] == 2
        
        print("✅ verificar_duplicatas: Detecta duplicatas")

    def test_verificar_duplicatas_subset(self, dados_com_problemas):
        """
        Testa verificação de duplicatas em subset de colunas.
        """
        validador = ValidadorDados(dados_com_problemas)
        
        resultado = validador.verificar_duplicatas(subset=['id'])
        
        assert resultado.detalhes['tipo'] == 'parciais'
        assert 'id' in resultado.detalhes['subset']
        
        print("✅ verificar_duplicatas: Funciona com subset")

    def test_validar_consistencia_cruzada_valida(self, dados_validos):
        """
        Testa validação de consistência cruzada com regras válidas.
        """
        validador = ValidadorDados(dados_validos)
        
        regras = [
            {
                'condicao': 'idade >= 18',
                'mensagem': 'Idade deve ser >= 18'
            },
            {
                'condicao': 'ativo == True or idade < 50',
                'mensagem': 'Regra de negócio específica'
            }
        ]
        
        resultado = validador.validar_consistencia_cruzada(regras)
        
        assert resultado.sucesso == True
        assert resultado.tipo == TipoValidacao.CONSISTENCIA
        
        print("✅ validar_consistencia_cruzada: Regras válidas passam")

    def test_validar_consistencia_cruzada_invalida(self, dados_com_problemas):
        """
        Testa validação de consistência cruzada com regras inválidas.
        """
        validador = ValidadorDados(dados_com_problemas)
        
        regras = [
            {
                'condicao': 'salario > 0',
                'mensagem': 'Salário deve ser positivo'
            }
        ]
        
        resultado = validador.validar_consistencia_cruzada(regras)
        
        assert resultado.sucesso == False
        assert len(resultado.detalhes['regras_violadas']) > 0
        
        print("✅ validar_consistencia_cruzada: Detecta regras violadas")

    def test_gerar_relatorio_validacao(self, dados_validos):
        """
        Testa geração de relatório de validação.
        """
        validador = ValidadorDados(dados_validos, "Test Dataset")
        
        # Executar algumas validações
        validador.validar_estrutura(['id', 'nome', 'idade', 'cidade', 'ativo', 'data_criacao'])
        validador.verificar_valores_ausentes()
        validador.verificar_duplicatas()
        
        relatorio = validador.gerar_relatorio_validacao()
        
        # Verificar estrutura do relatório
        assert 'estatisticas_gerais' in relatorio
        assert 'resultados_por_tipo' in relatorio
        assert 'resumo' in relatorio
        assert 'detalhes' in relatorio
        assert relatorio['dataset'] == "Test Dataset"
        
        # Verificar estatísticas
        assert relatorio['estatisticas_gerais']['total_validacoes'] == 3
        assert relatorio['estatisticas_gerais']['validacoes_sucesso'] == 3
        assert relatorio['estatisticas_gerais']['taxa_sucesso'] == 100.0
        
        print("✅ gerar_relatorio_validacao: Gera relatório completo")

    def test_imprimir_relatorio(self, dados_validos, capsys):
        """
        Testa impressão do relatório de validação.
        """
        validador = ValidadorDados(dados_validos, "Test Dataset")
        
        # Executar algumas validações
        validador.validar_estrutura(['id', 'nome', 'idade', 'cidade', 'ativo', 'data_criacao'])
        validador.verificar_valores_ausentes()
        
        validador.imprimir_relatorio()
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Verificar se o relatório foi impresso
        assert "RELATÓRIO DE VALIDAÇÃO" in output
        assert "Test Dataset" in output
        assert "ESTRUTURA" in output
        assert "AUSENTES" in output
        
        print("✅ imprimir_relatorio: Imprime relatório formatado")

    def test_validar_estrutura_dataset_func(self, dados_validos):
        """
        Testa a função de conveniência validar_estrutura_dataset.
        """
        colunas_esperadas = ['id', 'nome', 'idade', 'cidade', 'ativo', 'data_criacao']
        
        resultado = validar_estrutura_dataset(dados_validos, colunas_esperadas)
        
        assert resultado == True
        
        print("✅ validar_estrutura_dataset: Função de conveniência funciona")

    def test_verificar_valores_ausentes_func(self, dados_com_problemas):
        """
        Testa a função de conveniência verificar_valores_ausentes.
        """
        resultado = verificar_valores_ausentes(dados_com_problemas)
        
        assert 'ausentes_por_coluna' in resultado
        assert 'total_ausentes' in resultado
        assert resultado['total_ausentes'] > 0
        
        print("✅ verificar_valores_ausentes: Função de conveniência funciona")

    def test_validar_tipos_dados_func(self, dados_validos):
        """
        Testa a função de conveniência validar_tipos_dados.
        """
        mapeamento_tipos = {
            'id': 'int',
            'nome': 'object',
            'idade': 'int'
        }
        
        resultado = validar_tipos_dados(dados_validos, mapeamento_tipos)
        
        assert resultado == True
        
        print("✅ validar_tipos_dados: Função de conveniência funciona")

    def test_verificar_consistencia_categorica_func(self, dados_validos):
        """
        Testa a função de conveniência verificar_consistencia_categorica.
        """
        valores_esperados = ['SP', 'RJ', 'MG']
        
        resultado = verificar_consistencia_categorica(dados_validos, 'cidade', valores_esperados)
        
        assert resultado == True
        
        print("✅ verificar_consistencia_categorica: Função de conveniência funciona")

    def test_gerar_relatorio_validacao_func(self, dados_validos):
        """
        Testa a função de conveniência gerar_relatorio_validacao.
        """
        relatorio = gerar_relatorio_validacao(dados_validos)
        
        assert 'estatisticas_gerais' in relatorio
        assert relatorio['estatisticas_gerais']['total_validacoes'] >= 2  # Ausentes + Duplicatas
        
        print("✅ gerar_relatorio_validacao: Função de conveniência funciona")

# Funções de teste individuais para facilitar execução específica
def test_validador_inicializacao():
    """Teste individual para inicialização do Validador"""
    dados = pd.DataFrame({'coluna': [1, 2, 3]})
    validador = ValidadorDados(dados, "Test")
    assert validador.nome_dataset == "Test"
    print("✅ teste_validador_inicializacao: Passou")

def test_validar_estrutura_valida():
    """Teste individual para validação de estrutura válida"""
    dados = pd.DataFrame({'id': [1, 2], 'nome': ['A', 'B']})
    validador = ValidadorDados(dados)
    resultado = validador.validar_estrutura(['id', 'nome'])
    assert resultado.sucesso == True
    print("✅ teste_validar_estrutura_valida: Passou")

def test_verificar_valores_ausentes():
    """Teste individual para verificação de valores ausentes"""
    dados = pd.DataFrame({'coluna': [1, None, 3]})
    validador = ValidadorDados(dados)
    resultado = validador.verificar_valores_ausentes()
    assert resultado.sucesso == False
    print("✅ teste_verificar_valores_ausentes: Passou")

def test_verificar_consistencia_categorica():
    """Teste individual para verificação de consistência categórica"""
    dados = pd.DataFrame({'categoria': ['A', 'B', 'C']})
    validador = ValidadorDados(dados)
    resultado = validador.verificar_consistencia_categorica('categoria', ['A', 'B'])
    assert resultado.sucesso == False
    print("✅ teste_verificar_consistencia_categorica: Passou")

def test_gerar_relatorio_validacao():
    """Teste individual para geração de relatório"""
    dados = pd.DataFrame({'coluna': [1, 2, 3]})
    validador = ValidadorDados(dados)
    validador.verificar_valores_ausentes()
    relatorio = validador.gerar_relatorio_validacao()
    assert relatorio['estatisticas_gerais']['total_validacoes'] == 1
    print("✅ teste_gerar_relatorio_validacao: Passou")

if __name__ == "__main__":
    # Executar testes individualmente
    test_validador_inicializacao()
    test_validar_estrutura_valida()
    test_verificar_valores_ausentes()
    test_verificar_consistencia_categorica()
    test_gerar_relatorio_validacao()
    
    print("\n🎉 Todos os testes de validação de dados passaram!")