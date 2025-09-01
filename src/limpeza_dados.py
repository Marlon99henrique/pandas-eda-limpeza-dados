"""
Módulo de limpeza e transformação de dados para o projeto Telco Customer Churn.
Demonstra domínio avançado do Pandas com técnicas profissionais de limpeza.

Funcionalidades:
- Diagnóstico completo de qualidade de dados
- Correção de tipos de dados inadequados
- Tratamento estratégico de valores ausentes
- Normalização de valores categóricos
- Criação de features derivadas
- Validação final da qualidade
- Pipeline completo de limpeza

Autor: Marlon Henrique
Data: 2025
Versão: 1.0.1
"""

from __future__ import annotations

import logging
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import pandas as pd

# Logger do módulo (configurado externamente pela CLI/notebooks)
logger = logging.getLogger(__name__)


def diagnosticar_problemas(df: pd.DataFrame, detalhado: bool = True) -> Dict[str, Any]:
    """
    Realiza diagnóstico completo da qualidade dos dados com relatório detalhado.

    Args:
        df (pd.DataFrame): DataFrame com dados brutos
        detalhado (bool): Se True, registra análise detalhada de cada coluna

    Returns:
        Dict[str, Any]: Dicionário com métricas de qualidade
    """
    logger.info("🔍 Iniciando diagnóstico de qualidade de dados")

    resultado = {
        "dimensoes": df.shape,
        "memoria_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
        "colunas_com_ausentes": [],
        "total_valores_ausentes": 0,
        "tipos_incorretos": [],
        "inconsistencias_categoricas": [],
    }

    logger.info("📊 Dimensões: %d linhas x %d colunas", df.shape[0], df.shape[1])
    logger.info("📦 Consumo de memória: %.2f MB", resultado["memoria_mb"])

    # Tipos potencialmente incorretos: object com poucos valores únicos
    for col, dtype in df.dtypes.items():
        if dtype == "object" and df[col].nunique(dropna=True) < 10:
            resultado["tipos_incorretos"].append(col)

    # Valores ausentes
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if not missing_cols.empty:
        resultado["colunas_com_ausentes"] = missing_cols.index.tolist()
        resultado["total_valores_ausentes"] = int(missing_cols.sum())
        logger.info("❌ Colunas com ausentes: %s", ", ".join(resultado["colunas_com_ausentes"]))
        logger.debug("Detalhe ausentes:\n%s", missing_cols.to_string())
    else:
        logger.info("✅ Nenhum valor ausente encontrado")

    # Inconsistências categóricas simples (ex.: presença de espaços)
    if detalhado:
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            unique_vals = df[col].dropna().astype(str).unique()
            if any(" " in x for x in unique_vals):
                resultado["inconsistencias_categoricas"].append(col)
                logger.debug("⚠️  Possível inconsistência (espaços) em: %s", col)

    # Estatísticas descritivas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        logger.debug("📈 Resumo numérico:\n%s", df[numeric_cols].describe().round(2).to_string())
    else:
        logger.debug("ℹ️ Nenhuma variável numérica encontrada")

    logger.info(
        "📋 Resumo diagnóstico | ausentes cols: %d | total ausentes: %d | tipos incorretos: %d | inconsistências categóricas: %d",
        len(resultado["colunas_com_ausentes"]),
        resultado["total_valores_ausentes"],
        len(resultado["tipos_incorretos"]),
        len(resultado["inconsistencias_categoricas"]),
    )
    return resultado


def corrigir_tipos_dados(df: pd.DataFrame, config_tipos: Optional[Dict] = None) -> pd.DataFrame:
    """
    Corrige tipos de dados inadequados no DataFrame com mapeamento configurável.

    Args:
        df (pd.DataFrame): DataFrame com dados brutos
        config_tipos (Dict, optional): Dicionário com mapeamento de tipos

    Returns:
        pd.DataFrame: DataFrame com tipos corrigidos
    """
    logger.info("🛠️ Corrigindo tipos de dados")
    df_clean = df.copy()

    if config_tipos is None:
        # Configuração padrão para o dataset Telco
        config_tipos = {
            "SeniorCitizen": "category",
            "Partner": "category",
            "Dependents": "category",
            "tenure": "int32",
            "PhoneService": "category",
            "MultipleLines": "category",
            "InternetService": "category",
            "OnlineSecurity": "category",
            "OnlineBackup": "category",
            "DeviceProtection": "category",
            "TechSupport": "category",
            "StreamingTV": "category",
            "StreamingMovies": "category",
            "Contract": "category",
            "PaperlessBilling": "category",
            "PaymentMethod": "category",
            "Churn": "category",
            "TotalCharges": "float32",
            "MonthlyCharges": "float32",
        }

    conversoes_realizadas = 0
    for col, target_dtype in config_tipos.items():
        if col not in df_clean.columns:
            continue
        original_dtype = str(df_clean[col].dtype)
        try:
            if target_dtype == "category":
                df_clean[col] = df_clean[col].astype("category")
            elif target_dtype.startswith(("int", "float")):
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").astype(target_dtype)
            else:
                df_clean[col] = df_clean[col].astype(target_dtype)
            conversoes_realizadas += 1
            logger.debug("✅ %s: %s → %s", col, original_dtype, str(df_clean[col].dtype))
        except Exception as e:
            logger.warning("❌ Erro ao converter %s (%s → %s): %s", col, original_dtype, target_dtype, e)

    memoria_antes = float(df.memory_usage(deep=True).sum() / 1024**2)
    memoria_depois = float(df_clean.memory_usage(deep=True).sum() / 1024**2)
    economia = memoria_antes - memoria_depois
    logger.info("💾 Economia de memória: %.2f MB (%.2f → %.2f MB)", economia, memoria_antes, memoria_depois)
    logger.info("🔢 Total de conversões realizadas: %d", conversoes_realizadas)

    return df_clean


def tratar_valores_ausentes(df: pd.DataFrame, estrategias: Optional[Dict] = None) -> pd.DataFrame:
    """
    Trata valores ausentes usando estratégias avançadas e configuráveis.

    Args:
        df (pd.DataFrame): DataFrame com tipos corrigidos
        estrategias (Dict, optional): Estratégias específicas por coluna

    Returns:
        pd.DataFrame: DataFrame sem (ou com menos) valores ausentes
    """
    logger.info("🔧 Tratando valores ausentes")
    df_clean = df.copy()

    ausentes_antes = int(df_clean.isnull().sum().sum())
    if ausentes_antes == 0:
        logger.info("✅ Nenhum valor ausente encontrado para tratamento")
        return df_clean

    # Estratégias padrão para o dataset Telco
    if estrategias is None:
        estrategias = {
            "TotalCharges": {"estrategia": "mediana", "params": {}},
            "Dependents": {"estrategia": "moda", "params": {}},
            "PhoneService": {"estrategia": "moda", "params": {}},
            "InternetService": {"estrategia": "valor_constante", "params": {"valor": "No"}},
            "default": {"estrategia": "remover", "params": {}},
        }

    logger.info("🔍 Valores ausentes antes do tratamento: %d", ausentes_antes)

    linhas_removidas_total = 0
    for col in list(df_clean.columns):
        if not df_clean[col].isnull().any():
            continue

        n_ausentes = int(df_clean[col].isnull().sum())
        estrategia_col = estrategias.get(col, estrategias["default"])
        estrategia = estrategia_col.get("estrategia", "remover")
        params = estrategia_col.get("params", {})

        logger.debug("📦 %s: %d ausentes | Estratégia: %s", col, n_ausentes, estrategia)

        try:
            if estrategia == "mediana":
                fill_value = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(fill_value)
                logger.debug("   ✅ Preenchido com mediana: %s", fill_value)

            elif estrategia == "moda":
                fill_value = df_clean[col].mode().iloc[0]
                df_clean[col] = df_clean[col].fillna(fill_value)
                logger.debug("   ✅ Preenchido com moda: %s", fill_value)

            elif estrategia == "valor_constante":
                fill_value = params.get("valor", 0)
                df_clean[col] = df_clean[col].fillna(fill_value)
                logger.debug("   ✅ Preenchido com valor constante: %s", fill_value)

            elif estrategia == "interpolar":
                df_clean[col] = df_clean[col].interpolate(**params)
                logger.debug("   ✅ Interpolação aplicada")

            elif estrategia == "remover":
                prev_rows = df_clean.shape[0]
                df_clean = df_clean.dropna(subset=[col])
                removed = prev_rows - df_clean.shape[0]
                linhas_removidas_total += removed
                logger.debug("   ✅ Removidas %d linhas com valores ausentes", removed)

            else:
                logger.warning("   ⚠️ Estratégia '%s' não reconhecida para %s", estrategia, col)

        except Exception as e:
            logger.warning("   ❌ Erro ao tratar %s: %s", col, e)

    ausentes_depois = int(df_clean.isnull().sum().sum())
    logger.info(
        "🎯 Ausentes após tratamento: %d | Linhas removidas no processo: %d",
        ausentes_depois,
        linhas_removidas_total,
    )

    if ausentes_depois == 0:
        logger.info("✅ Todos os valores ausentes foram tratados com sucesso!")
    else:
        logger.warning("⚠️ Ainda existem %d valores ausentes", ausentes_depois)

    return df_clean


def normalizar_categoricas(df: pd.DataFrame, mapeamentos: Optional[Dict] = None) -> pd.DataFrame:
    """
    Normaliza valores categóricos e remove inconsistências com mapeamentos configuráveis.

    Args:
        df (pd.DataFrame): DataFrame com valores ausentes tratados
        mapeamentos (Dict, optional): Dicionário com mapeamentos personalizados

    Returns:
        pd.DataFrame: DataFrame com categorias normalizadas
    """
    logger.info("🎭 Normalizando valores categóricos")
    df_clean = df.copy()

    # Mapeamentos padrão: padronizando para PT-BR
    if mapeamentos is None:
        mapeamentos = {
            "Partner": {"Yes": "Sim", "No": "Não", "Y": "Sim", "N": "Não"},
            "Dependents": {"Yes": "Sim", "No": "Não", "Y": "Sim", "N": "Não"},
            "Churn": {"Yes": "Sim", "No": "Não", "Churned": "Sim", "Stayed": "Não"},
            "PhoneService": {"Yes": "Sim", "No": "Não"},
            "PaperlessBilling": {"Yes": "Sim", "No": "Não"},
            "MultipleLines": {"No phone service": "Sem serviço telefônico"},
            "OnlineSecurity": {"No internet service": "Sem serviço de internet"},
            "OnlineBackup": {"No internet service": "Sem serviço de internet"},
            "DeviceProtection": {"No internet service": "Sem serviço de internet"},
            "TechSupport": {"No internet service": "Sem serviço de internet"},
            "StreamingTV": {"No internet service": "Sem serviço de internet"},
            "StreamingMovies": {"No internet service": "Sem serviço de internet"},
        }

    normalizacoes_realizadas = 0

    for col, mapping in mapeamentos.items():
        if col not in df_clean.columns:
            continue

        valores_antes = int(df_clean[col].nunique(dropna=True))
        df_clean[col] = df_clean[col].replace(mapping)

        # Atualizar categorias se já for dtype 'category'
        if pd.api.types.is_categorical_dtype(df_clean[col]):
            novas_categorias = pd.Series(df_clean[col].unique()).dropna().tolist()
            df_clean[col] = df_clean[col].cat.set_categories(novas_categorias)

        valores_depois = int(df_clean[col].nunique(dropna=True))
        logger.debug("✅ %s: %d → %d valores únicos", col, valores_antes, valores_depois)
        normalizacoes_realizadas += 1

    logger.info("🔢 Normalizações realizadas: %d", normalizacoes_realizadas)
    return df_clean


def criar_novas_variaveis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria novas features derivadas para enriquecer a análise.

    Args:
        df (pd.DataFrame): DataFrame com dados limpos

    Returns:
        pd.DataFrame: DataFrame com novas variáveis
    """
    logger.info("✨ Criando novas variáveis")
    df_clean = df.copy()
    novas_variaveis: List[str] = []

    # 1) Categorização de tenure (tempo como cliente)
    if "tenure" in df_clean.columns:
        bins = [0, 12, 24, 36, 48, 60, 72, np.inf]
        labels = ["0–12m", "13–24m", "25–36m", "37–48m", "49–60m", "61–72m", "72m+"]
        df_clean["TenureGroup"] = pd.cut(
            df_clean["tenure"], bins=bins, labels=labels, right=True, include_lowest=True
        )
        df_clean["TenureGroup"] = df_clean["TenureGroup"].astype("category")
        novas_variaveis.append("TenureGroup")
        logger.debug("✅ Criada variável TenureGroup")

    # 2) Total de serviços contratados
    servicos = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    servicos_presentes = [col for col in servicos if col in df_clean.columns]
    if servicos_presentes:
        # True (tem serviço) se não for 'No' ou 'Sem serviço de internet'
        mask = pd.DataFrame(index=df_clean.index)
        for servico in servicos_presentes:
            mask[servico] = ~df_clean[servico].isin(["No", "Sem serviço de internet", "No internet service"])
        df_clean["TotalServicos"] = mask.sum(axis=1).astype(int)
        novas_variaveis.append("TotalServicos")
        logger.debug("✅ Criada variável TotalServicos (média: %.2f)", float(df_clean["TotalServicos"].mean()))

    # 3) Cliente com dependentes e parceiro
    if {"Partner", "Dependents"}.issubset(df_clean.columns):
        df_clean["TemFamilia"] = ((df_clean["Partner"] == "Sim") | (df_clean["Dependents"] == "Sim")).astype(int)
        novas_variaveis.append("TemFamilia")
        logger.debug("✅ Criada variável TemFamilia")

    # 4) Tipo de cliente baseado no contrato e tempo
    if {"Contract", "tenure"}.issubset(df_clean.columns):
        conditions = [
            (df_clean["Contract"] == "Month-to-month"),
            (df_clean["Contract"] == "One year") & (df_clean["tenure"] < 12),
            (df_clean["Contract"] == "One year") & (df_clean["tenure"] >= 12),
            (df_clean["Contract"] == "Two year") & (df_clean["tenure"] < 24),
            (df_clean["Contract"] == "Two year") & (df_clean["tenure"] >= 24),
        ]
        choices = ["Novo_Mensal", "Novo_Anual", "Estavel_Anual", "Novo_Bianual", "Estavel_Bianual"]
        df_clean["TipoCliente"] = np.select(conditions, choices, default="Outro")
        df_clean["TipoCliente"] = df_clean["TipoCliente"].astype("category")
        novas_variaveis.append("TipoCliente")
        logger.debug("✅ Criada variável TipoCliente")

    # 5) Valor médio mensal por serviço
    if {"MonthlyCharges", "TotalServicos"}.issubset(df_clean.columns):
        df_clean["CustoPorServico"] = np.where(
            df_clean["TotalServicos"] > 0, df_clean["MonthlyCharges"] / df_clean["TotalServicos"], 0.0
        )
        novas_variaveis.append("CustoPorServico")
        logger.debug("✅ Criada variável CustoPorServico")

    logger.info("🎯 Total de novas variáveis criadas: %d (%s)", len(novas_variaveis), ", ".join(novas_variaveis))
    return df_clean


def validar_qualidade_dados(df: pd.DataFrame) -> bool:
    """
    Realiza validação final da qualidade dos dados após limpeza.

    Args:
        df (pd.DataFrame): DataFrame processado

    Returns:
        bool: True se a qualidade for satisfatória
    """
    logger.info("✅ Validação final da qualidade")
    qualidade_ok = True
    testes: List[Tuple[str, bool, Any]] = []

    # 1) Ausentes totais
    missing_total = int(df.isnull().sum().sum())
    testes.append(("Valores ausentes", missing_total == 0, missing_total))

    # 2) Tipos essenciais
    tipos_essenciais = {
        "Churn": "category",
        "SeniorCitizen": "category",
        "Contract": "category",
        "TotalCharges": "float32",
        "MonthlyCharges": "float32",
    }
    for col, expected_type in tipos_essenciais.items():
        if col in df.columns:
            actual_type = str(df[col].dtype)
            tipo_correto = expected_type in actual_type
            testes.append((f"Tipo {col}", tipo_correto, actual_type))
            qualidade_ok &= tipo_correto

    # 3) Consistência categórica básica
    if "Churn" in df.columns:
        allowed = {"Sim", "Não"}
        unique_values = set(pd.Series(df["Churn"]).dropna().unique())
        is_subset = unique_values.issubset(allowed)
        testes.append(("Valores Churn", is_subset, list(unique_values)))
        qualidade_ok &= is_subset

    if "Partner" in df.columns:
        allowed = {"Sim", "Não"}
        unique_values = set(pd.Series(df["Partner"]).dropna().unique())
        is_subset = unique_values.issubset(allowed)
        testes.append(("Valores Partner", is_subset, list(unique_values)))
        qualidade_ok &= is_subset

    if "Dependents" in df.columns:
        allowed = {"Sim", "Não"}
        unique_values = set(pd.Series(df["Dependents"]).dropna().unique())
        is_subset = unique_values.issubset(allowed)
        testes.append(("Valores Dependents", is_subset, list(unique_values)))
        qualidade_ok &= is_subset

    # 4) Faixa numérica simples
    if "MonthlyCharges" in df.columns:
        dentro_faixa = (df["MonthlyCharges"] >= 0).all()
        rng = f"{df['MonthlyCharges'].min():.2f}-{df['MonthlyCharges'].max():.2f}"
        testes.append(("Faixa MonthlyCharges", dentro_faixa, rng))
        qualidade_ok &= bool(dentro_faixa)

    # Log dos testes
    logger.info("📋 Resultados dos testes:")
    for teste, ok, info in testes:
        status = "OK" if ok else "FALHA"
        logger.info("  %s: %s (%s)", teste, status, info)

    # Estatísticas finais
    logger.info(
        "📊 Estatísticas finais | Dimensões: %d x %d | Categóricas: %d | Numéricas: %d",
        df.shape[0],
        df.shape[1],
        len(df.select_dtypes(include="category").columns),
        len(df.select_dtypes(include=np.number).columns),
    )

    if qualidade_ok:
        logger.info("🎉 Qualidade dos dados validada com sucesso! Dataset pronto para análise/modelagem.")
    else:
        logger.warning("⚠️ ALERTA: Foram identificados problemas na qualidade dos dados.")

    return qualidade_ok


def pipeline_limpeza_completa(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Executa o pipeline completo de limpeza de dados.

    Args:
        df (pd.DataFrame): DataFrame com dados brutos
        verbose (bool): Se True, registra detalhes de cada etapa

    Returns:
        pd.DataFrame: DataFrame limpo e processado
    """
    # Controla verbosidade local do logger do módulo
    old_level = logger.level
    if verbose:
        logger.setLevel(logging.INFO)
    try:
        logger.info("🚀 Iniciando pipeline completo de limpeza")
        logger.info("📥 Input: %d linhas, %d colunas", df.shape[0], df.shape[1])

        df_clean = (
            df.pipe(corrigir_tipos_dados)
            .pipe(tratar_valores_ausentes)
            .pipe(normalizar_categoricas)
            .pipe(criar_novas_variaveis)
        )

        qualidade_ok = validar_qualidade_dados(df_clean)
        logger.info("📤 Output: %d linhas, %d colunas", df_clean.shape[0], df_clean.shape[1])
        logger.info("🏁 Pipeline concluído: %s", "SUCESSO" if qualidade_ok else "COM AVISOS")

        return df_clean
    finally:
        logger.setLevel(old_level)


# Execução direta (apenas para debug local)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("🔧 Módulo de limpeza de dados - Telco Customer Churn")
    logger.info("Use as funções individualmente ou pipeline_limpeza_completa(df, verbose=True)")
