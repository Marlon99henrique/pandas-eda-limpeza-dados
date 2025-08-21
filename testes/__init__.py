"""
Pacote de testes para o projeto Telco Customer Churn.

Este pacote contém testes unitários para validar as funções dos módulos:
- limpeza_dados: Testes para funções de limpeza e transformação
- validacao_dados: Testes para funções de validação
- visualizacao: Testes para funções de visualização
- utils: Testes para funções utilitárias

Os testes garantem a qualidade do código e previnem regressões.

Autor: Marlon Henrique
Data: 2025
Versão: 1.0.0
"""

# Version do pacote de testes
__version__ = "0.1.0"
__author__ = "Marlon Henrique"
__email__ = "marlon.99henrique@gmail.com"

# Importações para facilitar o acesso aos testes
from .test_limpeza_dados import (
    TestLimpezaDados,
    test_diagnosticar_problemas,
    test_corrigir_tipos_dados,
    test_tratar_valores_ausentes,
    test_normalizar_categoricas,
    test_criar_novas_variaveis,
    test_validar_qualidade_dados,
    test_pipeline_limpeza_completa
)

from .test_validacao_dados import (
    TestValidacaoDados,
    test_validar_estrutura_dataset,
    test_verificar_valores_ausentes,
    test_validar_tipos_dados,
    test_verificar_consistencia_categorica,
    test_gerar_relatorio_validacao
)

from .test_utils import (
    TestUtils,
    test_carregar_dados,
    test_salvar_dados,
    test_calcular_estatisticas_descritivas,
    test_gerar_resumo_dataset,
    test_verificar_duplicatas,
    test_criar_diretorios_projeto
)

from .test_visualizacao import (
    TestVisualizacao,
    test_plotar_distribuicoes_antes_depois,
    test_criar_heatmap_correlacao,
    test_plotar_valores_ausentes,
    test_visualizar_churn_por_categoria,
    test_criar_grafico_importancia_variaveis
)

# Lista do que será importado com "from testes import *"
__all__ = [
    # Testes de limpeza de dados
    'TestLimpezaDados',
    'test_diagnosticar_problemas',
    'test_corrigir_tipos_dados',
    'test_tratar_valores_ausentes',
    'test_normalizar_categoricas',
    'test_criar_novas_variaveis',
    'test_validar_qualidade_dados',
    'test_pipeline_limpeza_completa',
    
    # Testes de validação de dados
    'TestValidacaoDados',
    'test_validar_estrutura_dataset',
    'test_verificar_valores_ausentes',
    'test_validar_tipos_dados',
    'test_verificar_consistencia_categorica',
    'test_gerar_relatorio_validacao',
    
    # Testes de utilitários
    'TestUtils',
    'test_carregar_dados',
    'test_salvar_dados',
    'test_calcular_estatisticas_descritivas',
    'test_gerar_resumo_dataset',
    'test_verificar_duplicatas',
    'test_criar_diretorios_projeto',
    
    # Testes de visualização
    'TestVisualizacao',
    'test_plotar_distribuicoes_antes_depois',
    'test_criar_heatmap_correlacao',
    'test_plotar_valores_ausentes',
    'test_visualizar_churn_por_categoria',
    'test_criar_grafico_importancia_variaveis'
]

# Mensagem quando o pacote de testes é importado
print(f"✅ Pacote de testes versão {__version__} importado com sucesso!")
print("🧪 Módulos de teste disponíveis:")
print("   - test_limpeza_dados: Testes para funções de limpeza")
print("   - test_validacao_dados: Testes para funções de validação")
print("   - test_utils: Testes para funções utilitárias")
print("   - test_visualizacao: Testes para funções de visualização")
print("")
print("💡 Para executar todos os testes:")
print("   python -m pytest testes/ -v")
print("")
print("💡 Para executar testes específicos:")
print("   python -m pytest testes/test_limpeza_dados.py -v")