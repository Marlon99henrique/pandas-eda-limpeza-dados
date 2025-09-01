
"""
Módulo utilitário para o projeto Telco Customer Churn.
Funções auxiliares para carregamento, salvamento, configuração e utilitários gerais.

Autor: Marlon Henrique
Data: 2025
Versão: 1.0.1
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
# Dica: configure o logging no seu script/notebook/CLI, por exemplo:
# import logging
# logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Suprimir warnings (opcional)
warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# Visualização / Ambiente
# ----------------------------------------------------------------------
def configurar_ambiente_visualizacao(
    estilo: str = "seaborn-v0_8",
    contexto: str = "notebook",
    palette: str = "viridis",
) -> None:
    """
    Configura o ambiente de visualização com estilo e configurações padrão.

    Args:
        estilo (str): Estilo do matplotlib (ex.: 'seaborn-v0_8', 'ggplot', 'dark_background').
        contexto (str): Contexto do seaborn ('notebook', 'paper', 'talk', 'poster').
        palette (str): Paleta de cores do seaborn (ex.: 'viridis', 'deep', 'muted').

    Exemplo:
        >>> configurar_ambiente_visualizacao(estilo='seaborn-v0_8', contexto='notebook', palette='deep')
    """
    try:
        plt.style.use(estilo)
    except Exception:  # fallback se o estilo não existir
        plt.style.use("default")
        logger.warning("Estilo '%s' não encontrado. Usando 'default'.", estilo)

    try:
        sns.set_context(contexto)
    except Exception:
        sns.set_context("notebook")
        logger.warning("Contexto '%s' inválido. Usando 'notebook'.", contexto)

    try:
        sns.set_palette(palette)
    except Exception:
        logger.warning("Paleta '%s' inválida. Mantendo paleta padrão.", palette)

    # Configurações específicas do pandas (apenas quando solicitado)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.float_format", "{:.2f}".format)

    logger.info("Ambiente configurado: estilo=%s | contexto=%s | palette=%s", estilo, contexto, palette)


# ----------------------------------------------------------------------
# I/O de dados
# ----------------------------------------------------------------------
def carregar_dados(
    caminho: Union[str, Path],
    formato: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Carrega dados de vários formatos com tratamento de erros.

    Args:
        caminho (Union[str, Path]): Caminho para o arquivo.
        formato (str, optional): Formato do arquivo ('csv', 'xlsx', 'xls', 'parquet', 'feather', 'pickle'/'pkl').
        **kwargs: Argumentos adicionais repassados para as funções de leitura do pandas.

    Returns:
        pd.DataFrame: DataFrame com os dados carregados.

    Raises:
        FileNotFoundError: Se o arquivo não for encontrado.
        ValueError: Se o formato não for suportado.

    Exemplo:
        >>> df = carregar_dados('dados/brutos/telco.csv')
        >>> df = carregar_dados('dados/telco.parquet', formato='parquet')
    """
    caminho = Path(caminho)

    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    if formato is None:
        formato = caminho.suffix.lower().lstrip(".")  # deduz da extensão

    try:
        if formato == "csv":
            df = pd.read_csv(caminho, **kwargs)
        elif formato in ["xlsx", "xls"]:
            df = pd.read_excel(caminho, **kwargs)
        elif formato == "parquet":
            df = pd.read_parquet(caminho, **kwargs)
        elif formato == "feather":
            df = pd.read_feather(caminho, **kwargs)
        elif formato in ["pkl", "pickle"]:
            with open(caminho, "rb") as f:
                df = pickle.load(f)
        else:
            raise ValueError(f"Formato não suportado: {formato}")

        logger.info("✅ Dados carregados: %s (%d linhas, %d colunas)", caminho.name, len(df), len(df.columns))
        return df

    except Exception as e:
        logger.error("❌ Erro ao carregar %s: %s", caminho, e)
        raise


def salvar_dados(
    df: pd.DataFrame,
    caminho: Union[str, Path],
    formato: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Salva DataFrame em vários formatos com tratamento de erros.

    Args:
        df (pd.DataFrame): DataFrame para salvar.
        caminho (Union[str, Path]): Caminho para salvar o arquivo.
        formato (str, optional): Formato do arquivo ('csv', 'xlsx', 'xls', 'parquet', 'feather', 'pickle'/'pkl').
        **kwargs: Argumentos adicionais repassados para as funções de escrita do pandas.

    Exemplo:
        >>> salvar_dados(df, 'dados/processados/telco_limpo.csv')
        >>> salvar_dados(df, 'dados/processed/telco.parquet', formato='parquet')
    """
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)  # garante diretório

    if formato is None:
        formato = caminho.suffix.lower().lstrip(".")

    try:
        if formato == "csv":
            df.to_csv(caminho, index=False, **kwargs)
        elif formato in ["xlsx", "xls"]:
            df.to_excel(caminho, index=False, **kwargs)
        elif formato == "parquet":
            df.to_parquet(caminho, index=False, **kwargs)
        elif formato == "feather":
            df.to_feather(caminho, **kwargs)
        elif formato in ["pkl", "pickle"]:
            with open(caminho, "wb") as f:
                pickle.dump(df, f)
        else:
            raise ValueError(f"Formato não suportado: {formato}")

        tamanho_mb = caminho.stat().st_size / (1024 * 1024)
        logger.info("💾 Dados salvos: %s (%.2f MB)", caminho.name, tamanho_mb)

    except Exception as e:
        logger.error("❌ Erro ao salvar %s: %s", caminho, e)
        raise


# ----------------------------------------------------------------------
# Exploração / Resumos
# ----------------------------------------------------------------------
def calcular_estatisticas_descritivas(
    df: pd.DataFrame,
    colunas_numericas: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Calcula estatísticas descritivas completas para o DataFrame.

    Args:
        df (pd.DataFrame): DataFrame para análise.
        colunas_numericas (List[str], optional): Lista de colunas numéricas específicas.

    Returns:
        Dict[str, Any]: Dicionário com estatísticas descritivas.

    Exemplo:
        >>> stats = calcular_estatisticas_descritivas(df)
        >>> stats['geral']['linhas'], stats['numericas']['MonthlyCharges']['media']
    """
    if colunas_numericas is None:
        colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

    estatisticas: Dict[str, Any] = {
        "geral": {
            "linhas": int(df.shape[0]),
            "colunas": int(df.shape[1]),
            "memoria_mb": float(df.memory_usage(deep=True).sum() / (1024**2)),
            "colunas_numericas": int(len(colunas_numericas)),
            "colunas_categoricas": int(len(df.select_dtypes(include=["category", "object"]).columns)),
        },
        "ausentes": {
            "total_ausentes": int(df.isnull().sum().sum()),
            "colunas_com_ausentes": int(df.isnull().any().sum()),
            "percentual_ausentes": float(
                (df.isnull().sum().sum() / max(1, (df.shape[0] * df.shape[1]))) * 100
            ),
        },
        "numericas": {},
    }

    for col in colunas_numericas:
        if col in df.columns:
            serie = df[col]
            estatisticas["numericas"][col] = {
                "media": float(serie.mean()),
                "mediana": float(serie.median()),
                "desvio_padrao": float(serie.std()),
                "min": float(serie.min()),
                "max": float(serie.max()),
                "q1": float(serie.quantile(0.25)),
                "q3": float(serie.quantile(0.75)),
                "ausentes": int(serie.isnull().sum()),
                "zeros": int((serie == 0).sum()) if pd.api.types.is_numeric_dtype(serie) else 0,
            }

    return estatisticas


def gerar_resumo_dataset(df: pd.DataFrame, titulo: str = "Resumo do Dataset") -> None:
    """
    Gera um resumo completo e formatado do dataset (via logs INFO/DEBUG).

    Args:
        df (pd.DataFrame): DataFrame para análise.
        titulo (str): Título do resumo.

    Exemplo:
        >>> import logging
        >>> logging.basicConfig(level=logging.INFO)
        >>> gerar_resumo_dataset(df, "Análise Exploratória")
    """
    logger.info("📊 %s", titulo)
    logger.info("📦 Dimensões: %d linhas × %d colunas", df.shape[0], df.shape[1])
    logger.info("💾 Memória: %.2f MB", df.memory_usage(deep=True).sum() / 1024**2)

    tipos = df.dtypes.value_counts()
    logger.info("🎯 Tipos de dados:")
    for dtype, count in tipos.items():
        logger.info("  %s: %d colunas", dtype, count)

    ausentes_total = int(df.isnull().sum().sum())
    ausentes_colunas = int(df.isnull().any().sum())
    logger.info("❌ Valores ausentes: %d (em %d colunas)", ausentes_total, ausentes_colunas)

    if ausentes_colunas > 0:
        cols = [c for c in df.columns if df[c].isnull().any()]
        for col in cols:
            n_aus = int(df[col].isnull().sum())
            pct = (n_aus / max(1, len(df))) * 100
            logger.debug("  - %s: %d (%.1f%%)", col, n_aus, pct)

    cat_cols = df.select_dtypes(include=["category", "object"]).columns
    logger.info("🎭 Colunas categóricas: %d", len(cat_cols))
    for col in cat_cols:
        n_unique = int(df[col].nunique(dropna=True))
        logger.debug("  %s: %d valores únicos", col, n_unique)
        if n_unique <= 10:
            logger.debug("    Valores: %s", list(pd.Series(df[col].unique()).dropna()))

    num_cols = df.select_dtypes(include=[np.number]).columns
    logger.info("📈 Colunas numéricas: %d", len(num_cols))
    for col in num_cols:
        stats = df[col].describe()
        logger.debug(
            "  %s: média=%.2f ± %.2f | faixa=[%.2f–%.2f]",
            col,
            float(stats["mean"]),
            float(stats["std"]),
            float(stats["min"]),
            float(stats["max"]),
        )


# ----------------------------------------------------------------------
# Verificações / Amostragem
# ----------------------------------------------------------------------
def verificar_duplicatas(df: pd.DataFrame, subset: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Verifica e analisa linhas duplicadas no DataFrame.

    Args:
        df (pd.DataFrame): DataFrame para verificação.
        subset (List[str], optional): Subconjunto de colunas para checar duplicatas.

    Returns:
        Dict[str, Any]: Informações sobre duplicatas.

    Exemplo:
        >>> info = verificar_duplicatas(df, subset=['customerID'])
        >>> info['duplicatas_completas'], info.get('exemplo_duplicatas')
    """
    duplicatas_completas = int(df.duplicated().sum())
    duplicatas_subset = int(df.duplicated(subset=subset).sum()) if subset else 0

    resultado: Dict[str, Any] = {
        "duplicatas_completas": duplicatas_completas,
        "duplicatas_subset": duplicatas_subset,
        "percentual_completas": (duplicatas_completas / max(1, len(df))) * 100,
        "linhas_unicas": int(len(df) - duplicatas_completas),
    }

    if duplicatas_completas > 0:
        linhas_duplicadas = df[df.duplicated(keep=False)]
        resultado["exemplo_duplicatas"] = linhas_duplicadas.head(3).to_dict("records")

    return resultado


def amostrar_dataset(
    df: pd.DataFrame,
    tamanho: Union[int, float] = 0.1,
    estratificar: Optional[str] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Cria uma amostra do dataset, com opção de estratificação.

    Args:
        df (pd.DataFrame): DataFrame original.
        tamanho (Union[int, float]): Tamanho da amostra (absoluto ou proporção).
        estratificar (str, optional): Nome da coluna para estratificação.
        random_state (int): Seed para reprodutibilidade.

    Returns:
        pd.DataFrame: Amostra do dataset.

    Exemplo:
        >>> df_sample = amostrar_dataset(df, tamanho=0.2, estratificar='Churn')
    """
    n_total = len(df)
    if isinstance(tamanho, float):
        tamanho = int(n_total * tamanho)

    if tamanho <= 0:
        raise ValueError("tamanho da amostra deve ser > 0")

    if estratificar and estratificar in df.columns:
        try:
            from sklearn.model_selection import train_test_split
        except Exception as e:
            raise ImportError("scikit-learn é necessário para amostragem estratificada.") from e

        _, amostra = train_test_split(
            df,
            train_size=min(tamanho, n_total),
            stratify=df[estratificar],
            random_state=random_state,
        )
    else:
        amostra = df.sample(n=min(tamanho, n_total), random_state=random_state)

    logger.info("🔍 Amostra criada: %d linhas (%.1f%%)", len(amostra), (len(amostra) / max(1, n_total)) * 100)
    return amostra


# ----------------------------------------------------------------------
# Datas / Split temporal
# ----------------------------------------------------------------------
def dividir_dataset_temporal(
    df: pd.DataFrame,
    coluna_data: str,
    data_corte: str,
    formato_data: str = "%Y-%m-%d",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide o dataset em treino e teste baseado em uma data de corte.

    Args:
        df (pd.DataFrame): DataFrame com dados temporais.
        coluna_data (str): Nome da coluna de data.
        data_corte (str): Data para divisão (ex.: '2022-01-01').
        formato_data (str): Formato da data (ex.: '%Y-%m-%d').

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (treino, teste).

    Exemplo:
        >>> treino, teste = dividir_dataset_temporal(df, 'data_ref', '2023-01-01')
    """
    if coluna_data not in df.columns:
        raise KeyError(f"Coluna de data '{coluna_data}' não encontrada no DataFrame.")

    df_local = df.copy()
    df_local[coluna_data] = pd.to_datetime(df_local[coluna_data], format=formato_data, errors="coerce")

    data_corte_ts = pd.to_datetime(data_corte, format=formato_data, errors="coerce")
    if pd.isna(data_corte_ts):
        raise ValueError(f"Data de corte inválida: {data_corte}")

    treino = df_local[df_local[coluna_data] < data_corte_ts]
    teste = df_local[df_local[coluna_data] >= data_corte_ts]

    logger.info("📅 Dataset dividido: %d treino, %d teste", len(treino), len(teste))
    if not treino.empty:
        logger.debug("   Período treino: %s → %s", treino[coluna_data].min(), treino[coluna_data].max())
    if not teste.empty:
        logger.debug("   Período teste: %s → %s", teste[coluna_data].min(), teste[coluna_data].max())

    return treino, teste


# ----------------------------------------------------------------------
# Configurações (YAML)
# ----------------------------------------------------------------------
def carregar_configuracao(caminho: Union[str, Path] = "../config/parametros.yaml") -> Dict[str, Any]:
    """
    Carrega arquivo de configuração YAML.

    Args:
        caminho (Union[str, Path]): Caminho para o arquivo de configuração.

    Returns:
        Dict[str, Any]: Dicionário com configurações (vazio se não existir).

    Exemplo:
        >>> cfg = carregar_configuracao('../config/parametros.yaml')
        >>> cfg.get('geral', {}).get('versao')
    """
    caminho = Path(caminho)

    if not caminho.exists():
        logger.warning("Arquivo de configuração não encontrado: %s", caminho)
        return {}

    try:
        with open(caminho, "r") as f:
            config = yaml.safe_load(f)
        logger.info("✅ Configuração carregada: %s", caminho.name)
        return config or {}
    except Exception as e:
        logger.error("❌ Erro ao carregar configuração: %s", e)
        return {}


def salvar_configuracao(
    config: Dict[str, Any],
    caminho: Union[str, Path] = "../config/parametros.yaml",
) -> None:
    """
    Salva configuração em arquivo YAML.

    Args:
        config (Dict[str, Any]): Dicionário com configurações.
        caminho (Union[str, Path]): Caminho para salvar o arquivo.

    Exemplo:
        >>> salvar_configuracao({'geral': {'versao': '0.1.0'}}, '../config/parametros.yaml')
    """
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(caminho, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        logger.info("💾 Configuração salva: %s", caminho.name)
    except Exception as e:
        logger.error("❌ Erro ao salvar configuração: %s", e)
        raise


# ----------------------------------------------------------------------
# Estrutura do projeto
# ----------------------------------------------------------------------
def criar_diretorios_projeto() -> None:
    """
    Cria a estrutura de diretórios do projeto (se não existir).

    Exemplo:
        >>> criar_diretorios_projeto()
    """
    diretorios = [
        "../dados/brutos",
        "../dados/processados",
        "../dados/externos",
        "../notebooks",
        "../src",
        "../testes",
        "../docs",
        "../relatorios/figuras",
        "../ambiente",
        "../config",
    ]

    for d in diretorios:
        Path(d).mkdir(parents=True, exist_ok=True)

    logger.info("✅ Estrutura de diretórios criada/verificada")


# ----------------------------------------------------------------------
# Decorators
# ----------------------------------------------------------------------
def tempo_execucao(func):
    """
    Decorator para medir tempo de execução de funções.

    Args:
        func: Função a ser decorada.

    Returns:
        Função decorada com medição de tempo (log INFO).

    Exemplo:
        >>> @tempo_execucao
        ... def tarefa():
        ...     pass
    """
    import time

    def wrapper(*args, **kwargs):
        inicio = time.time()
        resultado = func(*args, **kwargs)
        fim = time.time()
        logger.info("⏱️  %s executado em %.2f segundos", func.__name__, fim - inicio)
        return resultado

    return wrapper


# ----------------------------------------------------------------------
# Execução direta (debug manual)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("🔧 Módulo utilitário - Telco Customer Churn")
    logger.info("Funções principais:")
    logger.info(" - configurar_ambiente_visualizacao()")
    logger.info(" - carregar_dados() / salvar_dados()")
    logger.info(" - calcular_estatisticas_descritivas()")
    logger.info(" - gerar_resumo_dataset()")
    logger.info(" - verificar_duplicatas()")
    logger.info(" - amostrar_dataset()")
    logger.info(" - dividir_dataset_temporal()")
    logger.info(" - carregar_configuracao() / salvar_configuracao()")
    logger.info(" - criar_diretorios_projeto()")
    logger.info(" - @tempo_execucao decorator")
