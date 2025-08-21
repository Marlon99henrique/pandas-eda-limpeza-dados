"""
⚙️ Utilitário para carregar configurações do arquivo YAML

Autor: Marlon Henrique
Data: 2025
"""

import yaml
from pathlib import Path

def carregar_configuracao(caminho=None):
    """
    Carrega configurações do arquivo YAML
    
    Args:
        caminho (str): Caminho para o arquivo de configuração
        
    Returns:
        dict: Dicionário com as configurações
    """
    if caminho is None:
        caminho = Path(__file__).parent / "parametros.yaml"
    
    try:
        with open(caminho, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠️ Arquivo de configuração não encontrado: {caminho}")
        return {}
    except Exception as e:
        print(f"❌ Erro ao carregar configuração: {e}")
        return {}

# Configuração global
config = carregar_configuracao()