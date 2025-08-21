#!/usr/bin/env python3
"""
📦 Setup.py para o projeto Telco Customer Churn

Este script permite instalar o projeto como um pacote Python,
facilitando a importação dos módulos e reprodução do ambiente.

Autor: Marlon Henrique
Email: marlon.99henrique@gmail.com
"""

from setuptools import setup, find_packages
from pathlib import Path
import codecs
import os

# Ler o conteúdo do README para long_description
readme_path = Path(__file__).parent / "README.md"
try:
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Análise e limpeza do dataset Telco Customer Churn usando Pandas"

# Ler requirements.txt para install_requires
requirements_path = Path(__file__).parent / "ambiente" / "requirements.txt"
try:
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f.readlines()
            if line.strip() and not line.startswith("#")
        ]
except FileNotFoundError:
    requirements = []

# Ler a versão do arquivo de configuração
def get_version():
    """Extrai a versão do arquivo de configuração YAML"""
    config_path = Path(__file__).parent / "config" / "parametros.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                if "versao:" in line:
                    return line.split(":")[1].strip().replace('"', '').replace("'", "")
    except FileNotFoundError:
        return "0.1.0"
    return "0.1.0"

# Configuração do setup
setup(
    # Informações básicas do projeto
    name="telco-churn-analysis",
    version=get_version(),
    author="Marlon Henrique",
    author_email="marlon.99henrique@gmail.com",
    description="Análise e limpeza do dataset Telco Customer Churn usando Pandas",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # URLs do projeto
    url="https://github.com/Marlon99henrique/pandas-eda-limpeza-dados",
    project_urls={
        "Bug Tracker": "https://github.com/Marlon99henrique/pandas-eda-limpeza-dados/issues",
        "Documentation": "https://github.com/Marlon99henrique/pandas-eda-limpeza-dados/blob/main/README.md",
        "Source Code": "https://github.com/Marlon99henrique/pandas-eda-limpeza-dados",
    },
    
    # Classificações
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Palavras-chave
    keywords=[
        "data-science",
        "pandas",
        "data-cleaning",
        "eda",
        "churn-analysis",
        "machine-learning",
        "data-analysis",
    ],
    
    # Pacotes e estrutura
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    
    # Dependências
    install_requires=requirements,
    python_requires=">=3.8",
    
    # Dependências extras
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "pre-commit>=3.0",
        ],
        "docs": [
            "sphinx>=7.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=2.0",
        ],
        "ml": [
            "scikit-learn>=1.2",
            "xgboost>=1.7",
            "lightgbm>=3.3",
            "catboost>=1.0",
        ],
        "viz": [
            "plotly>=5.14",
            "seaborn>=0.12",
            "matplotlib>=3.7",
            "missingno>=0.5",
        ],
    },
    
    # Scripts de console
    entry_points={
        "console_scripts": [
            "telco-analysis=src.cli:main",
            "telco-clean=src.limpeza_dados:pipeline_limpeza_completa_cli",
            "telco-validate=src.validacao_dados:validador_cli",
        ],
    },
    
    # Dados de pacote
    package_data={
        "": [
            "*.yaml",
            "*.yml",
            "*.json",
            "*.md",
        ],
    },
    
    # Metadados adicionais
    license="MIT",
    platforms=["any"],
    
    # Configurações de build
    zip_safe=False,
)

# Mensagem pós-instalação
def print_post_install_message():
    """Exibe mensagem útil após a instalação"""
    print("\n" + "="*60)
    print("🎉 Telco Churn Analysis instalado com sucesso!")
    print("="*60)
    print("\n📚 Como usar:")
    print("  Importe os módulos: from src import limpeza_dados, validacao_dados")
    print("  Execute análise: python -m notebooks.01_analise_telco")
    print("  Execute testes: python -m pytest testes/ -v")
    print("\n🔧 Comandos disponíveis:")
    print("  telco-analysis    - Interface CLI principal")
    print("  telco-clean       - Limpeza de dados via CLI")
    print("  telco-validate    - Validação de dados via CLI")
    print("\n📖 Documentação: https://github.com/Marlon99henrique/pandas-eda-limpeza-dados")
    print("="*60)

if __name__ == "__main__":
    print_post_install_message()