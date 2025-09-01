"""
Pacote src para o projeto de Análise e Limpeza do Telco Customer Churn.

Este pacote contém módulos para:
- Limpeza e transformação de dados
- Validação e qualidade de dados
- Visualizações e gráficos
- Utilitários e funções auxiliares

Módulos:
- limpeza_dados: Funções para limpeza e preparação de dados
- validacao_dados: Funções para validação e controle de qualidade
- visualizacao: Funções para criação de visualizações
- utils: Funções utilitárias e helpers

Exemplo de uso:
>>> from src import limpeza_dados, validacao_dados
>>> from src.visualizacao import criar_graficos
"""

# Version do pacote
__version__ = "0.1.0"
__author__ = "Marlon Henrique"
__email__ = "marlon.99henrique@gmail.com"

# Importações principais para facilitar o acesso
from .limpeza_dados import (
    diagnosticar_problemas,
    corrigir_tipos_dados,
    tratar_valores_ausentes,
    normalizar_categoricas,
    criar_novas_variaveis,
    validar_qualidade_dados,
    pipeline_limpeza_completa
)

from .validacao_dados import (
    validar_estrutura_dataset,
    verificar_valores_ausentes,
    validar_tipos_dados,
    verificar_consistencia_categorica,
    gerar_relatorio_validacao
)

from .visualizacao import (
    plotar_distribuicoes_antes_depois,
    criar_heatmap_correlacao,
    plotar_valores_ausentes,
    visualizar_churn_por_categoria,
    criar_grafico_importancia_variaveis
)

from .utils import (
    carregar_dados,
    salvar_dados,
    configurar_ambiente_visualizacao,
    calcular_estatisticas_descritivas,
    gerar_resumo_dataset
)

# Lista do que será importado com "from src import *"
__all__ = [
    # Funções de limpeza
    'diagnosticar_problemas',
    'corrigir_tipos_dados',
    'tratar_valores_ausentes',
    'normalizar_categoricas',
    'criar_novas_variaveis',
    'validar_qualidade_dados',
    'pipeline_limpeza_completa',
    
    # Funções de validação
    'validar_estrutura_dataset',
    'verificar_valores_ausentes',
    'validar_tipos_dados',
    'verificar_consistencia_categorica',
    'gerar_relatorio_validacao',
    
    # Funções de visualização
    'plotar_distribuicoes_antes_depois',
    'criar_heatmap_correlacao',
    'plotar_valores_ausentes',
    'visualizar_churn_por_categoria',
    'criar_grafico_importancia_variaveis',
    
    # Funções utilitárias
    'carregar_dados',
    'salvar_dados',
    'configurar_ambiente_visualizacao',
    'calcular_estatisticas_descritivas',
    'gerar_resumo_dataset'
]

