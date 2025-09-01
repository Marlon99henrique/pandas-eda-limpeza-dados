"""
Módulo de visualização e plotagem para o projeto Telco Customer Churn.
Funções para criação de gráficos, análise exploratória e visualização de dados.

Autor: Marlon Henrique
Data: 2025
Versão: 1.0.1
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats

import plotly.express as px
import plotly.graph_objects as go

# Logger do módulo (configure no app/notebook/CLI)
logger = logging.getLogger(__name__)

# Suprimir warnings visuais (opcional)
warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# Estilo / Configuração
# ----------------------------------------------------------------------
def configurar_estilo_graficos(
    contexto: str = "notebook",
    estilo: str = "whitegrid",
    palette: str = "viridis",
) -> None:
    """
    Configura o estilo dos gráficos de forma consistente.

    Args:
        contexto (str): Contexto do seaborn ('notebook', 'paper', 'talk', 'poster')
        estilo (str): Estilo do seaborn ('whitegrid', 'darkgrid', 'white', 'dark')
        palette (str): Paleta de cores do seaborn

    Exemplo:
        >>> configurar_estilo_graficos(contexto='notebook', estilo='whitegrid', palette='viridis')
    """
    try:
        sns.set_context(contexto)
    except Exception:
        sns.set_context("notebook")
        logger.warning("Contexto '%s' inválido. Usando 'notebook'.", contexto)

    try:
        sns.set_style(estilo)
    except Exception:
        sns.set_style("whitegrid")
        logger.warning("Estilo '%s' inválido. Usando 'whitegrid'.", estilo)

    try:
        sns.set_palette(palette)
    except Exception:
        logger.warning("Paleta '%s' inválida. Mantendo paleta padrão.", palette)

    # Tamanhos padrão (podem ser sobrescritos nos gráficos específicos)
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12

    logger.info("Estilo configurado: contexto=%s | estilo=%s | palette=%s", contexto, estilo, palette)


# ----------------------------------------------------------------------
# Gráficos Matplotlib / Seaborn
# ----------------------------------------------------------------------
def plotar_distribuicoes_antes_depois(
    df_antes: pd.DataFrame,
    df_depois: pd.DataFrame,
    colunas_numericas: List[str],
    titulo: str = "Comparação Antes/Depois da Limpeza",
) -> plt.Figure:
    """
    Plota comparação de distribuições antes e depois da limpeza (KDE).

    Args:
        df_antes (pd.DataFrame): DataFrame antes da limpeza
        df_depois (pd.DataFrame): DataFrame depois da limpeza
        colunas_numericas (List[str]): Colunas numéricas para comparar
        titulo (str): Título do gráfico

    Returns:
        plt.Figure: Figura matplotlib
    """
    if not colunas_numericas:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "Nenhuma coluna numérica fornecida", ha="center", va="center")
        ax.set_axis_off()
        return fig

    n_cols = min(3, len(colunas_numericas))
    n_rows = (len(colunas_numericas) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]

    for i, coluna in enumerate(colunas_numericas):
        if i >= len(axes):
            break
        if coluna not in df_antes.columns or coluna not in df_depois.columns:
            axes[i].set_axis_off()
            continue
        # somente dados numéricos
        s1 = pd.to_numeric(df_antes[coluna], errors="coerce").dropna()
        s2 = pd.to_numeric(df_depois[coluna], errors="coerce").dropna()
        if s1.empty and s2.empty:
            axes[i].set_axis_off()
            continue

        sns.kdeplot(s1, ax=axes[i], label="Antes", fill=True, alpha=0.5)
        sns.kdeplot(s2, ax=axes[i], label="Depois", fill=True, alpha=0.5)

        axes[i].set_title(f"Distribuição de {coluna}")
        axes[i].set_xlabel(coluna)
        axes[i].set_ylabel("Densidade")
        axes[i].legend()

        mu1, sigma1 = (s1.mean(), s1.std()) if not s1.empty else (np.nan, np.nan)
        mu2, sigma2 = (s2.mean(), s2.std()) if not s2.empty else (np.nan, np.nan)
        stats_text = f"Antes: μ={mu1:.2f}, σ={sigma1:.2f}\nDepois: μ={mu2:.2f}, σ={sigma2:.2f}"
        axes[i].text(
            0.05,
            0.95,
            stats_text,
            transform=axes[i].transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Apagar eixos sobrando
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(titulo, fontsize=16, fontweight="bold")
    plt.tight_layout()
    return fig


def criar_heatmap_correlacao(
    df: pd.DataFrame,
    metodo: str = "pearson",
    annot: bool = True,
    mask_superior: bool = True,
    titulo: str = "Mapa de Correlação",
) -> plt.Figure:
    """
    Cria heatmap de correlação para variáveis numéricas.

    Args:
        df (pd.DataFrame): DataFrame com dados
        metodo (str): Método ('pearson', 'spearman', 'kendall')
        annot (bool): Mostrar valores de correlação
        mask_superior (bool): Mascara a triangular superior
        titulo (str): Título do gráfico

    Returns:
        plt.Figure: Figura matplotlib
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "Correlação requer ao menos 2 colunas numéricas", ha="center", va="center")
        ax.set_axis_off()
        return fig

    corr_matrix = df[numeric_cols].corr(method=metodo)

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) if mask_superior else None

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=annot,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(titulo, fontsize=16, fontweight="bold", pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig


def plotar_valores_ausentes(df: pd.DataFrame, titulo: str = "Análise de Valores Ausentes") -> plt.Figure:
    """
    Visualiza a distribuição de valores ausentes no dataset.

    Args:
        df (pd.DataFrame): DataFrame para análise
        titulo (str): Título do gráfico

    Returns:
        plt.Figure: Figura matplotlib
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, "Nenhum valor ausente encontrado", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        return fig

    missing_pct = (missing / max(1, len(df))) * 100

    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])

    # Barras
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(missing.index, missing, alpha=0.8)
    ax1.set_title("Valores Ausentes por Coluna", fontweight="bold")
    ax1.set_ylabel("Quantidade")
    ax1.tick_params(axis="x", rotation=45)
    for bar, pct in zip(bars, missing_pct):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 0.1, f"{int(h)}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

    # Pizza
    ax2 = fig.add_subplot(gs[0, 1])
    wedges, texts, autotexts = ax2.pie(missing_pct, labels=missing.index, autopct="%1.1f%%", startangle=90)
    ax2.set_title("Distribuição Percentual", fontweight="bold")

    # Matriz de ausentes
    ax3 = fig.add_subplot(gs[1, :])
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax3)
    ax3.set_title("Matriz de Valores Ausentes", fontweight="bold")
    ax3.set_xlabel("Colunas")
    ax3.set_ylabel("Linhas")
    ax3.tick_params(axis="x", rotation=45)

    plt.suptitle(titulo, fontsize=16, fontweight="bold")
    plt.tight_layout()
    return fig


def _to_binary_target(s: pd.Series) -> pd.Series:
    """
    Converte target categórico/booleano em binário {0,1}.

    Regras:
      - Trata 'Sim', 'Yes', True como 1
      - Trata 'Não', 'No', False como 0
      - Se já é numérico, força para 0/1

    Retorna:
        pd.Series (0/1) alinhada ao índice original
    """
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)

    if pd.api.types.is_numeric_dtype(s):
        return (s > 0).astype(int)

    m = (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({"sim": 1, "yes": 1, "true": 1, "não": 0, "nao": 0, "no": 0, "false": 0})
    )
    return m.fillna(0).astype(int)


def visualizar_churn_por_categoria(
    df: pd.DataFrame,
    colunas_categoricas: List[str],
    target: str = "Churn",
    max_categories: int = 10,
    titulo: str = "Churn por Categoria",
) -> plt.Figure:
    """
    Visualiza a taxa de churn por diferentes categorias.

    Args:
        df (pd.DataFrame): DataFrame com dados
        colunas_categoricas (List[str]): Lista de colunas categóricas
        target (str): Coluna target (aceita 'Sim/Não', 'Yes/No', bool ou 0/1)
        max_categories (int): Número máximo de categorias por coluna
        titulo (str): Título do gráfico

    Returns:
        plt.Figure: Figura matplotlib
    """
    if target not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, f"Target '{target}' não encontrado", ha="center", va="center")
        ax.set_axis_off()
        return fig

    y = _to_binary_target(df[target])

    n_cols = min(3, len(colunas_categoricas))
    n_rows = (len(colunas_categoricas) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]

    for i, coluna in enumerate(colunas_categoricas):
        if i >= len(axes):
            break
        ax = axes[i]
        if coluna not in df.columns:
            ax.set_axis_off()
            continue

        churn_rate = y.groupby(df[coluna]).mean().sort_values(ascending=False)
        churn_rate = churn_rate.head(max_categories)

        bars = ax.bar(range(len(churn_rate)), churn_rate.values * 100)
        ax.set_title(f"Taxa de Churn por {coluna}")
        ax.set_xlabel(coluna)
        ax.set_ylabel("Taxa de Churn (%)")
        ax.set_xticks(range(len(churn_rate)))
        ax.set_xticklabels(churn_rate.index, rotation=45, ha="right")

        for j, val in enumerate(churn_rate.values):
            ax.text(j, val * 100 + 1, f"{val*100:.1f}%", ha="center", va="bottom", fontweight="bold")

        media_geral = float(y.mean() * 100)
        ax.axhline(y=media_geral, color="red", linestyle="--", label=f"Média Geral: {media_geral:.1f}%")
        ax.legend()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(titulo, fontsize=16, fontweight="bold")
    plt.tight_layout()
    return fig


def criar_grafico_importancia_variaveis(
    df: pd.DataFrame,
    target: str = "Churn",
    metodo: str = "mutual_info",
    top_n: int = 15,
    titulo: str = "Importância das Variáveis",
) -> plt.Figure:
    """
    Cria gráfico de importância das variáveis para o target.

    Args:
        df (pd.DataFrame): DataFrame com dados
        target (str): Coluna target ('Sim/Não', 'Yes/No', bool ou 0/1)
        metodo (str): 'mutual_info', 'chi2', 'correlation'
        top_n (int): Top variáveis
        titulo (str): Título do gráfico

    Returns:
        plt.Figure: Figura matplotlib
    """
    from sklearn.feature_selection import mutual_info_classif, chi2
    from sklearn.preprocessing import LabelEncoder

    if target not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, f"Target '{target}' não encontrado", ha="center", va="center")
        ax.set_axis_off()
        return fig

    X = df.drop(columns=[target])
    y = _to_binary_target(df[target])

    # codifica categóricas
    X_encoded = X.copy()
    le = LabelEncoder()
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X_encoded[col] = le.fit_transform(X[col].astype(str))

    if metodo == "mutual_info":
        importancia = mutual_info_classif(X_encoded, y, random_state=42)
    elif metodo == "chi2":
        # chi2 requer não-negativos: normaliza mín->0
        X_nonneg = X_encoded.copy()
        for c in X_nonneg.columns:
            if pd.api.types.is_numeric_dtype(X_nonneg[c]):
                minv = X_nonneg[c].min()
                if pd.notna(minv) and minv < 0:
                    X_nonneg[c] = X_nonneg[c] - minv
        importancia = chi2(X_nonneg, y)[0]
    else:  # correlation
        numeric_cols = X_encoded.select_dtypes(include=[np.number]).columns
        serie = X_encoded[numeric_cols].corrwith(y).abs()
        full = pd.Series(0.0, index=X_encoded.columns)
        full.loc[serie.index] = serie.values
        importancia = full.values

    feature_importance = (
        pd.DataFrame({"feature": X_encoded.columns, "importance": importancia})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(feature_importance)), feature_importance["importance"])
    ax.set_yticks(range(len(feature_importance)))
    ax.set_yticklabels(feature_importance["feature"])
    ax.set_xlabel("Importância")
    ax.set_title(titulo, fontsize=16, fontweight="bold")

    for i, (_, row) in enumerate(feature_importance.iterrows()):
        ax.text(row["importance"] * 1.01, i, f'{row["importance"]:.3f}', va="center", fontweight="bold")

    plt.gca().invert_yaxis()
    plt.tight_layout()
    return fig


def plotar_boxplots_numericos(
    df: pd.DataFrame,
    colunas_numericas: List[str],
    target: str = "Churn",
    titulo: str = "Distribuição por Target",
) -> plt.Figure:
    """
    Cria boxplots para variáveis numéricas agrupadas por target.

    Args:
        df (pd.DataFrame): DataFrame com dados
        colunas_numericas (List[str]): Colunas numéricas
        target (str): Coluna target (aceita 'Sim/Não', 'Yes/No', bool ou 0/1)
        titulo (str): Título do gráfico

    Returns:
        plt.Figure: Figura matplotlib
    """
    if target not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, f"Target '{target}' não encontrado", ha="center", va="center")
        ax.set_axis_off()
        return fig

    target_cat = df[target]
    # Garanta rótulos legíveis
    if pd.api.types.is_bool_dtype(target_cat) or pd.api.types.is_numeric_dtype(target_cat):
        target_cat = _to_binary_target(target_cat).map({0: "Não", 1: "Sim"})

    n_cols = min(3, len(colunas_numericas))
    n_rows = (len(colunas_numericas) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]

    for i, coluna in enumerate(colunas_numericas):
        if i >= len(axes):
            break
        ax = axes[i]
        if coluna not in df.columns:
            ax.set_axis_off()
            continue

        # Boxplot
        sns.boxplot(x=target_cat, y=df[coluna], ax=ax)
        ax.set_title(f"{coluna} por {target}")
        ax.set_xlabel(target)
        ax.set_ylabel(coluna)

        # Teste estatístico (somente 2 grupos e com dados)
        groups = [df.loc[target_cat == g, coluna].dropna() for g in target_cat.unique()]
        groups = [g for g in groups if not g.empty]
        if len(groups) == 2:
            try:
                stat, p_value = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                ax.text(
                    0.5,
                    0.95,
                    f"p-value: {p_value:.3f}",
                    transform=ax.transAxes,
                    ha="center",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                )
            except Exception:
                pass

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(titulo, fontsize=16, fontweight="bold")
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Plotly (interativo)
# ----------------------------------------------------------------------
def criar_grafico_interativo(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
    titulo: str = "Gráfico Interativo",
) -> go.Figure:
    """
    Cria gráfico interativo usando Plotly.

    Args:
        df (pd.DataFrame): DataFrame com dados
        x_col (str): Coluna eixo x
        y_col (str): Coluna eixo y
        color_col (str, optional): Coluna de cor
        size_col (str, optional): Coluna para tamanho
        titulo (str): Título

    Returns:
        go.Figure: Figura Plotly
    """
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        title=titulo,
        hover_data=df.columns,
        template="plotly_white",
    )
    fig.update_layout(title_font_size=20, title_x=0.5, width=1000, height=600)
    return fig


# ----------------------------------------------------------------------
# Salvamento
# ----------------------------------------------------------------------
def salvar_grafico(
    fig: Union[plt.Figure, go.Figure],
    caminho: Union[str, Path],
    formato: str = "png",
    dpi: int = 300,
    **kwargs,
) -> None:
    """
    Salva gráfico em diferentes formatos.

    Args:
        fig: Figura matplotlib ou plotly
        caminho (str|Path): Caminho para salvar
        formato (str): 'png', 'jpg', 'svg', 'pdf', 'html' (para Plotly)
        dpi (int): Resolução (Matplotlib)
        **kwargs: Argumentos adicionais
    """
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    try:
        if isinstance(fig, plt.Figure):
            fig.savefig(caminho, format=formato, dpi=dpi, bbox_inches="tight", **kwargs)
            logger.info("Gráfico salvo: %s", caminho)
        elif isinstance(fig, go.Figure):
            if formato == "html" or str(caminho).lower().endswith(".html"):
                fig.write_html(caminho, **kwargs)
            else:
                # Para imagens estáticas Plotly, é preciso ter 'kaleido' instalado
                try:
                    fig.write_image(caminho, format=formato, **kwargs)
                except ValueError as e:
                    logger.error(
                        "Falhou ao exportar imagem Plotly. Instale 'kaleido' (pip install -U kaleido). Erro: %s",
                        e,
                    )
                    raise
            logger.info("Gráfico interativo salvo: %s", caminho)
        else:
            raise TypeError("Tipo de figura não suportado.")
    except Exception as e:
        logger.error("Erro ao salvar gráfico em %s: %s", caminho, e)
        raise


# ----------------------------------------------------------------------
# Execução direta (demonstração)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s: %(message)s")
    print("🎨 Módulo de visualização - Telco Customer Churn\n")
    print("💡 Funções disponíveis:")
    print("   - configurar_estilo_graficos()")
    print("   - plotar_distribuicoes_antes_depois()")
    print("   - criar_heatmap_correlacao()")
    print("   - plotar_valores_ausentes()")
    print("   - visualizar_churn_por_categoria()")
    print("   - criar_grafico_importancia_variaveis()")
    print("   - plotar_boxplots_numericos()")
    print("   - criar_grafico_interativo()")
    print("   - salvar_grafico()")
