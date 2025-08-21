"""
Módulo utilitário para o projeto Telco Customer Churn.
Funções auxiliares para carregamento, salvamento, configuração e utilitários gerais.

Autor: Marlon Henrique
Data: 2025
Versão: 1.0.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import json
import yaml
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import pickle
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suprimir warnings
warnings.filterwarnings('ignore')

def configurar_ambiente_visualizacao(estilo: str = 'seaborn', 
                                   contexto: str = 'notebook', 
                                   palette: str = 'viridis') -> None:
    """
    Configura o ambiente de visualização com estilo e configurações padrão.
    
    Args:
        estilo (str): Estilo do matplotlib ('seaborn', 'ggplot', 'dark_background', etc.)
        contexto (str): Contexto do seaborn ('notebook', 'paper', 'talk', 'poster')
        palette (str): Palette de cores do seaborn
    """
    plt.style.use(estilo)
    sns.set_context(contexto)
    sns.set_palette(palette)
    
    # Configurações específicas do pandas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 50)
    pd.set_option('display.float_format', '{:.2f}'.format)
    
    print(f"✅ Ambiente configurado: estilo={estilo}, contexto={contexto}, palette={palette}")

def carregar_dados(caminho: Union[str, Path], 
                  formato: Optional[str] = None,
                  **kwargs) -> pd.DataFrame:
    """
    Carrega dados de vários formatos com tratamento de erros.
    
    Args:
        caminho (Union[str, Path]): Caminho para o arquivo
        formato (str, optional): Formato do arquivo ('csv', 'excel', 'parquet', 'feather', 'pickle')
        **kwargs: Argumentos adicionais para a função de carregamento
        
    Returns:
        pd.DataFrame: DataFrame com os dados carregados
        
    Raises:
        FileNotFoundError: Se o arquivo não for encontrado
        ValueError: Se o formato não for suportado
    """
    caminho = Path(caminho)
    
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    
    # Determinar formato automaticamente se não especificado
    if formato is None:
        formato = caminho.suffix.lower()[1:]  # Remove o ponto da extensão
    
    try:
        if formato == 'csv':
            df = pd.read_csv(caminho, **kwargs)
        elif formato in ['xlsx', 'xls']:
            df = pd.read_excel(caminho, **kwargs)
        elif formato == 'parquet':
            df = pd.read_parquet(caminho, **kwargs)
        elif formato == 'feather':
            df = pd.read_feather(caminho, **kwargs)
        elif formato in ['pkl', 'pickle']:
            with open(caminho, 'rb') as f:
                df = pickle.load(f)
        else:
            raise ValueError(f"Formato não suportado: {formato}")
        
        logger.info(f"✅ Dados carregados: {caminho.name} ({len(df)} linhas, {len(df.columns)} colunas)")
        return df
        
    except Exception as e:
        logger.error(f"❌ Erro ao carregar {caminho}: {e}")
        raise

def salvar_dados(df: pd.DataFrame, 
                caminho: Union[str, Path], 
                formato: Optional[str] = None,
                **kwargs) -> None:
    """
    Salva DataFrame em vários formatos com tratamento de erros.
    
    Args:
        df (pd.DataFrame): DataFrame para salvar
        caminho (Union[str, Path]): Caminho para salvar o arquivo
        formato (str, optional): Formato do arquivo ('csv', 'excel', 'parquet', 'feather', 'pickle')
        **kwargs: Argumentos adicionais para a função de salvamento
    """
    caminho = Path(caminho)
    
    # Criar diretório se não existir
    caminho.parent.mkdir(parents=True, exist_ok=True)
    
    # Determinar formato automaticamente se não especificado
    if formato is None:
        formato = caminho.suffix.lower()[1:]  # Remove o ponto da extensão
    
    try:
        if formato == 'csv':
            df.to_csv(caminho, index=False, **kwargs)
        elif formato in ['xlsx', 'xls']:
            df.to_excel(caminho, index=False, **kwargs)
        elif formato == 'parquet':
            df.to_parquet(caminho, index=False, **kwargs)
        elif formato == 'feather':
            df.to_feather(caminho, **kwargs)
        elif formato in ['pkl', 'pickle']:
            with open(caminho, 'wb') as f:
                pickle.dump(df, f)
        else:
            raise ValueError(f"Formato não suportado: {formato}")
        
        tamanho_mb = caminho.stat().st_size / (1024 * 1024)
        logger.info(f"💾 Dados salvos: {caminho.name} ({tamanho_mb:.2f} MB)")
        
    except Exception as e:
        logger.error(f"❌ Erro ao salvar {caminho}: {e}")
        raise

def calcular_estatisticas_descritivas(df: pd.DataFrame, 
                                     colunas_numericas: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Calcula estatísticas descritivas completas para o DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame para análise
        colunas_numericas (List[str], optional): Lista de colunas numéricas específicas
        
    Returns:
        Dict[str, Any]: Dicionário com estatísticas descritivas
    """
    if colunas_numericas is None:
        colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    
    estatisticas = {
        'geral': {
            'linhas': df.shape[0],
            'colunas': df.shape[1],
            'memoria_mb': df.memory_usage(deep=True).sum() / (1024**2),
            'colunas_numericas': len(colunas_numericas),
            'colunas_categoricas': len(df.select_dtypes(include=['category', 'object']).columns)
        },
        'ausentes': {
            'total_ausentes': df.isnull().sum().sum(),
            'colunas_com_ausentes': df.isnull().any().sum(),
            'percentual_ausentes': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        },
        'numericas': {}
    }
    
    # Estatísticas para cada coluna numérica
    for col in colunas_numericas:
        if col in df.columns:
            estatisticas['numericas'][col] = {
                'media': df[col].mean(),
                'mediana': df[col].median(),
                'desvio_padrao': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'q1': df[col].quantile(0.25),
                'q3': df[col].quantile(0.75),
                'ausentes': df[col].isnull().sum(),
                'zeros': (df[col] == 0).sum() if df[col].dtype != 'object' else 0
            }
    
    return estatisticas

def gerar_resumo_dataset(df: pd.DataFrame, titulo: str = "Resumo do Dataset") -> None:
    """
    Gera um resumo completo e formatado do dataset.
    
    Args:
        df (pd.DataFrame): DataFrame para análise
        titulo (str): Título do resumo
    """
    print(f"📊 {titulo}")
    print("=" * 60)
    
    # Informações básicas
    print(f"📦 Dimensões: {df.shape[0]} linhas × {df.shape[1]} colunas")
    print(f"💾 Memória: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Tipos de dados
    tipos = df.dtypes.value_counts()
    print("\n🎯 Tipos de dados:")
    for dtype, count in tipos.items():
        print(f"  {dtype}: {count} colunas")
    
    # Valores ausentes
    ausentes_total = df.isnull().sum().sum()
    ausentes_colunas = df.isnull().any().sum()
    print(f"\n❌ Valores ausentes: {ausentes_total} ({ausentes_colunas} colunas com ausentes)")
    
    if ausentes_colunas > 0:
        print("  Colunas com valores ausentes:")
        for col in df.columns[df.isnull().any()]:
            n_ausentes = df[col].isnull().sum()
            pct_ausentes = (n_ausentes / len(df)) * 100
            print(f"    {col}: {n_ausentes} ({pct_ausentes:.1f}%)")
    
    # Colunas categóricas
    cat_cols = df.select_dtypes(include=['category', 'object']).columns
    print(f"\n🎭 Colunas categóricas: {len(cat_cols)}")
    for col in cat_cols:
        n_unique = df[col].nunique()
        print(f"  {col}: {n_unique} valores únicos")
        if n_unique <= 10:
            print(f"    Valores: {list(df[col].unique())}")
    
    # Colunas numéricas
    num_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\n📈 Colunas numéricas: {len(num_cols)}")
    for col in num_cols:
        stats = df[col].describe()
        print(f"  {col}: {stats['mean']:.2f} ± {stats['std']:.2f} "
              f"[{stats['min']:.2f}-{stats['max']:.2f}]")

def verificar_duplicatas(df: pd.DataFrame, subset: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Verifica e analisa linhas duplicadas no DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame para verificação
        subset (List[str], optional): Subconjunto de colunas para verificar duplicatas
        
    Returns:
        Dict[str, Any]: Informações sobre duplicatas
    """
    duplicatas_completas = df.duplicated().sum()
    duplicatas_subset = df.duplicated(subset=subset).sum() if subset else 0
    
    resultado = {
        'duplicatas_completas': duplicatas_completas,
        'duplicatas_subset': duplicatas_subset,
        'percentual_completas': (duplicatas_completas / len(df)) * 100 if len(df) > 0 else 0,
        'linhas_unicas': len(df) - duplicatas_completas
    }
    
    if duplicatas_completas > 0:
        linhas_duplicadas = df[df.duplicated(keep=False)]
        resultado['exemplo_duplicatas'] = linhas_duplicadas.head(3).to_dict('records')
    
    return resultado

def dividir_dataset_temporal(df: pd.DataFrame, 
                           coluna_data: str, 
                           data_corte: str, 
                           formato_data: str = '%Y-%m-%d') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide o dataset em treino e teste baseado em data.
    
    Args:
        df (pd.DataFrame): DataFrame com dados temporais
        coluna_data (str): Nome da coluna de data
        data_corte (str): Data para divisão
        formato_data (str): Formato da data
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (treino, teste)
    """
    df[coluna_data] = pd.to_datetime(df[coluna_data], format=formato_data)
    data_corte = pd.to_datetime(data_corte)
    
    treino = df[df[coluna_data] < data_corte]
    teste = df[df[coluna_data] >= data_corte]
    
    logger.info(f"📅 Dataset dividido: {len(treino)} treino, {len(teste)} teste")
    logger.info(f"   Período treino: {treino[coluna_data].min()} até {treino[coluna_data].max()}")
    logger.info(f"   Período teste: {teste[coluna_data].min()} até {teste[coluna_data].max()}")
    
    return treino, teste

def carregar_configuracao(caminho: Union[str, Path] = '../config/parametros.yaml') -> Dict[str, Any]:
    """
    Carrega arquivo de configuração YAML.
    
    Args:
        caminho (Union[str, Path]): Caminho para o arquivo de configuração
        
    Returns:
        Dict[str, Any]: Dicionário com configurações
    """
    caminho = Path(caminho)
    
    if not caminho.exists():
        logger.warning(f"Arquivo de configuração não encontrado: {caminho}")
        return {}
    
    try:
        with open(caminho, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✅ Configuração carregada: {caminho.name}")
        return config
    except Exception as e:
        logger.error(f"❌ Erro ao carregar configuração: {e}")
        return {}

def salvar_configuracao(config: Dict[str, Any], 
                       caminho: Union[str, Path] = '../config/parametros.yaml') -> None:
    """
    Salva configuração em arquivo YAML.
    
    Args:
        config (Dict[str, Any]): Dicionário com configurações
        caminho (Union[str, Path]): Caminho para salvar o arquivo
    """
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(caminho, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"💾 Configuração salva: {caminho.name}")
    except Exception as e:
        logger.error(f"❌ Erro ao salvar configuração: {e}")
        raise

def criar_diretorios_projeto() -> None:
    """
    Cria a estrutura de diretórios do projeto se não existir.
    """
    diretorios = [
        '../dados/brutos',
        '../dados/processados',
        '../dados/externos',
        '../notebooks',
        '../src',
        '../testes',
        '../docs',
        '../relatorios/figuras',
        '../ambiente',
        '../config'
    ]
    
    for diretorio in diretorios:
        Path(diretorio).mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ Estrutura de diretórios criada/verificada")

def tempo_execucao(func):
    """
    Decorator para medir tempo de execução de funções.
    
    Args:
        func: Função a ser decorada
        
    Returns:
        Função decorada com medição de tempo
    """
    import time
    
    def wrapper(*args, **kwargs):
        inicio = time.time()
        resultado = func(*args, **kwargs)
        fim = time.time()
        tempo_decorrido = fim - inicio
        
        logger.info(f"⏱️  {func.__name__} executado em {tempo_decorrido:.2f} segundos")
        return resultado
    
    return wrapper

def amostrar_dataset(df: pd.DataFrame, 
                    tamanho: Union[int, float] = 0.1, 
                    estratificar: Optional[str] = None,
                    random_state: int = 42) -> pd.DataFrame:
    """
    Cria uma amostra do dataset, com opção de estratificação.
    
    Args:
        df (pd.DataFrame): DataFrame original
        tamanho (Union[int, float]): Tamanho da amostra (absoluto ou proporção)
        estratificar (str, optional): Coluna para estratificação
        random_state (int): Seed para reproducibilidade
        
    Returns:
        pd.DataFrame: Amostra do dataset
    """
    if isinstance(tamanho, float):
        tamanho = int(len(df) * tamanho)
    
    if estratificar and estratificar in df.columns:
        from sklearn.model_selection import train_test_split
        _, amostra = train_test_split(
            df, 
            train_size=tamanho, 
            stratify=df[estratificar],
            random_state=random_state
        )
    else:
        amostra = df.sample(n=min(tamanho, len(df)), random_state=random_state)
    
    logger.info(f"🔍 Amostra criada: {len(amostra)} linhas ({len(amostra)/len(df)*100:.1f}%)")
    return amostra

# Exemplo de uso
if __name__ == "__main__":
    print("🔧 Módulo utilitário - Telco Customer Churn")
    print("💡 Funções disponíveis:")
    print("   - configurar_ambiente_visualizacao()")
    print("   - carregar_dados() / salvar_dados()")
    print("   - calcular_estatisticas_descritivas()")
    print("   - gerar_resumo_dataset()")
    print("   - verificar_duplicatas()")
    print("   - carregar_configuracao() / salvar_configuracao()")
    print("   - criar_diretorios_projeto()")
    print("   - @tempo_execucao decorator")