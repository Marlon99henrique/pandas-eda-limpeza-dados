"""
Testes unitários para o módulo limpeza_dados.py

Testes para garantir que as funções de limpeza e transformação de dados
funcionam corretamente e mantêm a qualidade dos dados.

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

from limpeza_dados import (
    diagnosticar_problemas,
    corrigir_tipos_dados,
    tratar_valores_ausentes,
    normalizar_categoricas,
    criar_novas_variaveis,
    validar_qualidade_dados,
    pipeline_limpeza_completa
)

class TestLimpezaDados:
    """Classe de testes para o módulo limpeza_dados"""
    
    @pytest.fixture
    def dados_brutos_simulados(self):
        """
        Fixture para criar dados brutos simulados com problemas comuns.
        """
        dados = {
            'customerID': ['1', '2', '3', '4', '5'],
            'gender': ['Male', 'Female', 'male', 'FEMALE', 'Male'],
            'SeniorCitizen': ['0', '1', '0', '1', '2'],  # Valor inconsistente
            'Partner': ['Yes', 'No', 'yes', 'NO', 'Yes'],
            'Dependents': ['Yes', 'No', 'No', None, 'Yes'],  # Valor ausente
            'tenure': ['12', '24', '0', '36', '48'],
            'PhoneService': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'MonthlyCharges': ['29.85', '56.95', '53.85', '42.30', '70.70'],
            'TotalCharges': ['29.85', '1366.80', '0', '1522.80', None],  # Valor ausente
            'Churn': ['No', 'No', 'Yes', 'No', 'Yes']
        }
        return pd.DataFrame(dados)
    
    @pytest.fixture
    def dados_limpos_simulados(self):
        """
        Fixture para criar dados limpos de referência.
        """
        dados = {
            'customerID': ['1', '2', '3', '4', '5'],
            'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'SeniorCitizen': pd.Categorical(['0', '1', '0', '1', '1']),
            'Partner': pd.Categorical(['Yes', 'No', 'Yes', 'No', 'Yes']),
            'Dependents': pd.Categorical(['Yes', 'No', 'No', 'No', 'Yes']),
            'tenure': [12, 24, 0, 36, 48],
            'PhoneService': pd.Categorical(['Yes', 'No', 'Yes', 'No', 'Yes']),
            'MonthlyCharges': [29.85, 56.95, 53.85, 42.30, 70.70],
            'TotalCharges': [29.85, 1366.80, 0.0, 1522.80, 3393.60],  # Preenchido
            'Churn': pd.Categorical(['No', 'No', 'Yes', 'No', 'Yes'])
        }
        return pd.DataFrame(dados)

    def test_diagnosticar_problemas(self, dados_brutos_simulados):
        """
        Testa se o diagnóstico identifica corretamente os problemas nos dados.
        """
        resultado = diagnosticar_problemas(dados_brutos_simulados, detalhado=False)
        
        # Verificar se identificou os problemas
        assert resultado['total_valores_ausentes'] > 0
        assert len(resultado['colunas_com_ausentes']) > 0
        assert 'TotalCharges' in resultado['colunas_com_ausentes']
        assert 'Dependents' in resultado['colunas_com_ausentes']
        
        # Verificar dimensões
        assert resultado['dimensoes'] == dados_brutos_simulados.shape
        
        print("✅ diagnóstico_problemas: Identifica corretamente problemas nos dados")

    def test_corrigir_tipos_dados(self, dados_brutos_simulados):
        """
        Testa se a correção de tipos de dados funciona corretamente.
        """
        df_corrigido = corrigir_tipos_dados(dados_brutos_simulados)
        
        # Verificar tipos corrigidos
        assert pd.api.types.is_numeric_dtype(df_corrigido['tenure'])
        assert pd.api.types.is_numeric_dtype(df_corrigido['MonthlyCharges'])
        assert pd.api.types.is_float_dtype(df_corrigido['TotalCharges'])
        assert pd.api.types.is_categorical_dtype(df_corrigido['gender'])
        assert pd.api.types.is_categorical_dtype(df_corrigido['Churn'])
        
        # Verificar conversão de valores
        assert df_corrigido['tenure'].dtype in [np.int32, np.int64]
        assert df_corrigido['MonthlyCharges'].dtype in [np.float32, np.float64]
        
        print("✅ corrigir_tipos_dados: Converte tipos corretamente")

    def test_tratar_valores_ausentes(self, dados_brutos_simulados):
        """
        Testa se o tratamento de valores ausentes funciona corretamente.
        """
        # Primeiro corrigir tipos
        df_corrigido = corrigir_tipos_dados(dados_brutos_simulados)
        
        # Tratar valores ausentes
        df_sem_ausentes = tratar_valores_ausentes(df_corrigido)
        
        # Verificar se não há mais valores ausentes
        assert df_sem_ausentes.isnull().sum().sum() == 0
        
        # Verificar se valores foram preenchidos corretamente
        assert df_sem_ausentes['Dependents'].isnull().sum() == 0
        assert df_sem_ausentes['TotalCharges'].isnull().sum() == 0
        
        # Verificar se a mediana foi usada para TotalCharges
        mediana_esperada = df_corrigido['TotalCharges'].median()
        assert df_sem_ausentes['TotalCharges'].iloc[4] == mediana_esperada
        
        print("✅ tratar_valores_ausentes: Remove/preenche valores ausentes corretamente")

    def test_normalizar_categoricas(self, dados_brutos_simulados):
        """
        Testa se a normalização de valores categóricos funciona corretamente.
        """
        # Primeiro corrigir tipos
        df_corrigido = corrigir_tipos_dados(dados_brutos_simulados)
        
        # Normalizar valores categóricos
        df_normalizado = normalizar_categoricas(df_corrigido)
        
        # Verificar normalização
        valores_gender = set(df_normalizado['gender'].unique())
        assert valores_gender.issubset({'Male', 'Female'})
        
        valores_partner = set(df_normalizado['Partner'].unique())
        assert valores_partner.issubset({'Yes', 'No', 'Sim', 'Não'})
        
        # Verificar se valores inconsistentes foram corrigidos
        assert 'male' not in df_normalizado['gender'].unique()
        assert 'FEMALE' not in df_normalizado['gender'].unique()
        assert 'yes' not in df_normalizado['Partner'].unique()
        assert 'NO' not in df_normalizado['Partner'].unique()
        
        print("✅ normalizar_categoricas: Normaliza valores categóricos corretamente")

    def test_criar_novas_variaveis(self, dados_limpos_simulados):
        """
        Testa se a criação de novas variáveis funciona corretamente.
        """
        df_com_novas_variaveis = criar_novas_variaveis(dados_limpos_simulados)
        
        # Verificar se novas variáveis foram criadas
        assert 'TenureGroup' in df_com_novas_variaveis.columns
        assert 'TotalServicos' in df_com_novas_variaveis.columns
        assert 'TemFamilia' in df_com_novas_variaveis.columns
        
        # Verificar tipos das novas variáveis
        assert pd.api.types.is_categorical_dtype(df_com_novas_variaveis['TenureGroup'])
        assert pd.api.types.is_numeric_dtype(df_com_novas_variaveis['TotalServicos'])
        assert pd.api.types.is_numeric_dtype(df_com_novas_variaveis['TemFamilia'])
        
        # Verificar valores das novas variáveis
        assert df_com_novas_variaveis['TotalServicos'].min() >= 0
        assert df_com_novas_variaveis['TemFamilia'].min() >= 0
        assert df_com_novas_variaveis['TemFamilia'].max() <= 1
        
        print("✅ criar_novas_variaveis: Cria novas variáveis corretamente")

    def test_validar_qualidade_dados(self, dados_limpos_simulados):
        """
        Testa se a validação de qualidade detecta dados corretos.
        """
        qualidade_ok = validar_qualidade_dados(dados_limpos_simulados)
        
        # Dados limpos devem passar na validação
        assert qualidade_ok == True
        
        print("✅ validar_qualidade_dados: Valida dados corretos apropriadamente")

    def test_validar_qualidade_dados_com_problemas(self, dados_brutos_simulados):
        """
        Testa se a validação de qualidade detecta dados com problemas.
        """
        # Não corrigir problemas propositalmente
        qualidade_ok = validar_qualidade_dados(dados_brutos_simulados)
        
        # Dados brutos com problemas devem falhar na validação
        assert qualidade_ok == False
        
        print("✅ validar_qualidade_dados: Detecta dados com problemas apropriadamente")

    def test_pipeline_limpeza_completa(self, dados_brutos_simulados):
        """
        Testa se o pipeline completo de limpeza funciona corretamente.
        """
        df_limpo = pipeline_limpeza_completa(dados_brutos_simulados, verbose=False)
        
        # Verificar se todos os problemas foram resolvidos
        assert df_limpo.isnull().sum().sum() == 0
        
        # Verificar tipos corretos
        assert pd.api.types.is_numeric_dtype(df_limpo['tenure'])
        assert pd.api.types.is_numeric_dtype(df_limpo['MonthlyCharges'])
        assert pd.api.types.is_float_dtype(df_limpo['TotalCharges'])
        
        # Verificar valores categóricos normalizados
        valores_gender = set(df_limpo['gender'].unique())
        assert valores_gender.issubset({'Male', 'Female', 'Sim', 'Não'})
        
        # Verificar se novas variáveis foram criadas
        assert 'TenureGroup' in df_limpo.columns
        assert 'TotalServicos' in df_limpo.columns
        
        # Verificar qualidade final
        qualidade_final = validar_qualidade_dados(df_limpo)
        assert qualidade_final == True
        
        print("✅ pipeline_limpeza_completa: Executa fluxo completo corretamente")

    def test_pipeline_mantem_dimensoes_razoaveis(self, dados_brutos_simulados):
        """
        Testa se o pipeline mantém dimensões razoáveis dos dados.
        """
        df_limpo = pipeline_limpeza_completa(dados_brutos_simulados, verbose=False)
        
        # Verificar que não perdeu linhas significativamente
        assert len(df_limpo) >= len(dados_brutos_simulados) * 0.8  # Máximo 20% de perda
        
        # Verificar que ganhou colunas (novas variáveis)
        assert len(df_limpo.columns) >= len(dados_brutos_simulados.columns)
        
        print("✅ pipeline_limpeza_completa: Mantém dimensões razoáveis dos dados")

# Funções de teste individuais para facilitar execução específica
def test_diagnosticar_problemas():
    """Teste individual para diagnosticar_problemas"""
    dados = {
        'coluna_numerica': ['1', '2', None, '4'],
        'coluna_categorica': ['A', 'B', 'C', None]
    }
    df = pd.DataFrame(dados)
    resultado = diagnosticar_problemas(df, detalhado=False)
    assert resultado['total_valores_ausentes'] == 2
    print("✅ teste_diagnosticar_problemas: Passou")

def test_corrigir_tipos_dados():
    """Teste individual para corrigir_tipos_dados"""
    dados = {
        'numerica_str': ['1', '2', '3'],
        'categorica': ['A', 'B', 'A']
    }
    df = pd.DataFrame(dados)
    df_corrigido = corrigir_tipos_dados(df)
    assert pd.api.types.is_numeric_dtype(df_corrigido['numerica_str'])
    print("✅ teste_corrigir_tipos_dados: Passou")

def test_tratar_valores_ausentes():
    """Teste individual para tratar_valores_ausentes"""
    dados = {
        'coluna': [1, 2, None, 4]
    }
    df = pd.DataFrame(dados)
    df_tratado = tratar_valores_ausentes(df)
    assert df_tratado.isnull().sum().sum() == 0
    print("✅ teste_tratar_valores_ausentes: Passou")

def test_normalizar_categoricas():
    """Teste individual para normalizar_categoricas"""
    dados = {
        'categorica': ['SIM', 'não', 'Sim', 'NÃO']
    }
    df = pd.DataFrame(dados)
    df_normalizado = normalizar_categoricas(df)
    valores_unicos = set(df_normalizado['categorica'].unique())
    assert valores_unicos.issubset({'Sim', 'Não'})
    print("✅ teste_normalizar_categoricas: Passou")

def test_criar_novas_variaveis():
    """Teste individual para criar_novas_variaveis"""
    dados = {
        'tenure': [12, 24, 36],
        'Partner': ['Yes', 'No', 'Yes'],
        'Dependents': ['Yes', 'No', 'No']
    }
    df = pd.DataFrame(dados)
    df_novo = criar_novas_variaveis(df)
    assert 'TenureGroup' in df_novo.columns
    assert 'TemFamilia' in df_novo.columns
    print("✅ teste_criar_novas_variaveis: Passou")

def test_validar_qualidade_dados():
    """Teste individual para validar_qualidade_dados"""
    dados = {
        'numerica': [1, 2, 3],
        'categorica': pd.Categorical(['A', 'B', 'A'])
    }
    df = pd.DataFrame(dados)
    assert validar_qualidade_dados(df) == True
    print("✅ teste_validar_qualidade_dados: Passou")

def test_pipeline_limpeza_completa():
    """Teste individual para pipeline_limpeza_completa"""
    dados = {
        'coluna_numerica': ['1', '2', '3'],
        'coluna_categorica': ['A', 'B', 'A']
    }
    df = pd.DataFrame(dados)
    df_limpo = pipeline_limpeza_completa(df, verbose=False)
    assert validar_qualidade_dados(df_limpo) == True
    print("✅ teste_pipeline_limpeza_completa: Passou")

if __name__ == "__main__":
    # Executar testes individualmente
    test_diagnosticar_problemas()
    test_corrigir_tipos_dados()
    test_tratar_valores_ausentes()
    test_normalizar_categoricas()
    test_criar_novas_variaveis()
    test_validar_qualidade_dados()
    test_pipeline_limpeza_completa()
    
    print("\n🎉 Todos os testes de limpeza de dados passaram!")