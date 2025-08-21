# 📋 Metodologia de Análise - Telco Customer Churn

## 🎯 Objetivo do Projeto
Demonstrar domínio completo da biblioteca Pandas através da limpeza e preparação do dataset Telco Customer Churn, transformando dados brutos em um conjunto pronto para análise.

## 🔄 Fluxo de Trabalho Aplicado

### 1. 📥 Carregamento e Diagnóstico Inicial
**Arquivos envolvidos:** `src/utils.py`, `src/limpeza_dados.py`
```python
# Funções principais utilizadas:
from src.utils import carregar_dados, mostrar_resumo
from src.limpeza_dados import diagnosticar_problemas

df = carregar_dados('dados/brutos/telco_churn.csv')
mostrar_resumo(df)  # Visão geral dos dados
diagnostico = diagnosticar_problemas(df)  # Identifica problemas