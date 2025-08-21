"""
Módulo de visualização e plotagem para o projeto Telco Customer Churn.
Funções para criação de gráficos, análise exploratória e visualização de dados.

Autor: Marlon Henrique
Data: 2025
Versão: 1.0.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple, Union
import warnings
from matplotlib.gridspec import GridSpec
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuração e estilo
plt.style.use('default')
sns.set_palette("viridis")
sns.set_context("notebook")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

warnings.filterwarnings('ignore')

def configurar_estilo_graficos(contexto: str = 'notebook', 
                             estilo: str = 'whitegrid',
                             palette: str = 'viridis') -> None:
    """
    Configura o estilo dos gráficos de forma consistente.
    
    Args:
        contexto (str): Contexto do seaborn ('notebook', 'paper', 'talk', 'poster')
        estilo (str): Estilo do seaborn ('whitegrid', 'darkgrid', 'white', 'dark')
        palette (str): Palette de cores do seaborn
    """
    sns.set_context(contexto)
    sns.set_style(estilo)
    sns.set_palette(palette)
    
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    
    print(f"✅ Estilo configurado: contexto={contexto}, estilo={estilo}, palette={palette}")

def plotar_distribuicoes_antes_depois(df_antes: pd.DataFrame, 
                                    df_depois: pd.DataFrame,
                                    colunas_numericas: List[str],
                                    titulo: str = "Comparação Antes/Depois da Limpeza") -> plt.Figure:
    """
    Plota comparação de distribuições antes e depois da limpeza.
    
    Args:
        df_antes (pd.DataFrame): DataFrame antes da limpeza
        df_depois (pd.DataFrame): DataFrame depois da limpeza
        colunas_numericas (List[str]): Lista de colunas numéricas para comparar
        titulo (str): Título do gráfico
        
    Returns:
        plt.Figure: Figura matplotlib
    """
    n_cols = min(3, len(colunas_numericas))
    n_rows = (len(colunas_numericas) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, coluna in enumerate(colunas_numericas):
        if i < len(axes):
            # Plotar distribuições
            sns.kdeplot(df_antes[coluna].dropna(), ax=axes[i], label='Antes', fill=True, alpha=0.5)
            sns.kdeplot(df_depois[coluna].dropna(), ax=axes[i], label='Depois', fill=True, alpha=0.5)
            
            axes[i].set_title(f'Distribuição de {coluna}')
            axes[i].set_xlabel(coluna)
            axes[i].set_ylabel('Densidade')
            axes[i].legend()
            
            # Adicionar estatísticas
            stats_text = f'Antes: μ={df_antes[coluna].mean():.2f}, σ={df_antes[coluna].std():.2f}\n'
            stats_text += f'Depois: μ={df_depois[coluna].mean():.2f}, σ={df_depois[coluna].std():.2f}'
            axes[i].text(0.05, 0.95, stats_text, transform=axes[i].transAxes, 
                       fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Remover eixos vazios
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(titulo, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def criar_heatmap_correlacao(df: pd.DataFrame, 
                           metodo: str = 'pearson',
                           annot: bool = True,
                           mask_superior: bool = True,
                           titulo: str = "Mapa de Correlação") -> plt.Figure:
    """
    Cria heatmap de correlação para variáveis numéricas.
    
    Args:
        df (pd.DataFrame): DataFrame com dados
        metodo (str): Método de correlação ('pearson', 'spearman', 'kendall')
        annot (bool): Se True, mostra valores de correlação
        mask_superior (bool): Se True, mascara a triangular superior
        titulo (str): Título do gráfico
        
    Returns:
        plt.Figure: Figura matplotlib
    """
    # Selecionar apenas colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr(method=metodo)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Criar máscara para triangular superior
    mask = None
    if mask_superior:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Plotar heatmap
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=annot,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                square=True,
                cbar_kws={"shrink": .8},
                ax=ax)
    
    ax.set_title(titulo, fontsize=16, fontweight='bold', pad=20)
    
    # Rotacionar labels do eixo x
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    return fig

def plotar_valores_ausentes(df: pd.DataFrame, 
                          titulo: str = "Análise de Valores Ausentes") -> plt.Figure:
    """
    Visualiza a distribuição de valores ausentes no dataset.
    
    Args:
        df (pd.DataFrame): DataFrame para análise
        titulo (str): Título do gráfico
        
    Returns:
        plt.Figure: Figura matplotlib
    """
    # Calcular valores ausentes
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    missing_pct = (missing / len(df)) * 100
    
    if missing.empty:
        print("✅ Nenhum valor ausente encontrado")
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, 'Nenhum valor ausente encontrado', 
               ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Criar figura com subplots
    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])
    
    # Gráfico 1: Valores ausentes por coluna (barplot)
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(missing.index, missing, color='skyblue', alpha=0.7)
    ax1.set_title('Valores Ausentes por Coluna', fontweight='bold')
    ax1.set_ylabel('Quantidade de Valores Ausentes')
    ax1.tick_params(axis='x', rotation=45)
    
    # Adicionar valores nas barras
    for bar, pct in zip(bars, missing_pct):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    
    # Gráfico 2: Percentual de valores ausentes (pie chart)
    ax2 = fig.add_subplot(gs[0, 1])
    colors = plt.cm.Set3(np.linspace(0, 1, len(missing)))
    wedges, texts, autotexts = ax2.pie(missing_pct, labels=missing.index, 
                                      autopct='%1.1f%%', colors=colors,
                                      startangle=90)
    ax2.set_title('Distribuição Percentual', fontweight='bold')
    
    # Gráfico 3: Matriz de valores ausentes
    ax3 = fig.add_subplot(gs[1, :])
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax3)
    ax3.set_title('Matriz de Valores Ausentes', fontweight='bold')
    ax3.set_xlabel('Colunas')
    ax3.set_ylabel('Linhas')
    ax3.tick_params(axis='x', rotation=45)
    
    plt.suptitle(titulo, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def visualizar_churn_por_categoria(df: pd.DataFrame,
                                 colunas_categoricas: List[str],
                                 target: str = 'Churn',
                                 max_categories: int = 10,
                                 titulo: str = "Churn por Categoria") -> plt.Figure:
    """
    Visualiza a taxa de churn por diferentes categorias.
    
    Args:
        df (pd.DataFrame): DataFrame com dados
        colunas_categoricas (List[str]): Lista de colunas categóricas
        target (str): Coluna target (default: 'Churn')
        max_categories (int): Número máximo de categorias por coluna
        titulo (str): Título do gráfico
        
    Returns:
        plt.Figure: Figura matplotlib
    """
    n_cols = min(3, len(colunas_categoricas))
    n_rows = (len(colunas_categoricas) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, coluna in enumerate(colunas_categoricas):
        if i < len(axes) and coluna in df.columns:
            # Calcular taxas de churn
            churn_rate = df.groupby(coluna)[target].mean().sort_values(ascending=False)
            
            # Limitar número de categorias
            if len(churn_rate) > max_categories:
                churn_rate = churn_rate.head(max_categories)
            
            # Plotar gráfico de barras
            bars = axes[i].bar(range(len(churn_rate)), churn_rate.values * 100, 
                             color=plt.cm.viridis(np.linspace(0, 1, len(churn_rate))))
            
            axes[i].set_title(f'Taxa de Churn por {coluna}')
            axes[i].set_xlabel(coluna)
            axes[i].set_ylabel('Taxa de Churn (%)')
            axes[i].set_xticks(range(len(churn_rate)))
            axes[i].set_xticklabels(churn_rate.index, rotation=45, ha='right')
            
            # Adicionar valores nas barras
            for j, (idx, value) in enumerate(churn_rate.items()):
                axes[i].text(j, value * 100 + 1, f'{value*100:.1f}%', 
                           ha='center', va='bottom', fontweight='bold')
            
            # Adicionar linha de média geral
            media_geral = df[target].mean() * 100
            axes[i].axhline(y=media_geral, color='red', linestyle='--', 
                          label=f'Média Geral: {media_geral:.1f}%')
            axes[i].legend()
    
    # Remover eixos vazios
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(titulo, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def criar_grafico_importancia_variaveis(df: pd.DataFrame,
                                      target: str = 'Churn',
                                      metodo: str = 'mutual_info',
                                      top_n: int = 15,
                                      titulo: str = "Importância das Variáveis") -> plt.Figure:
    """
    Cria gráfico de importância das variáveis para o target.
    
    Args:
        df (pd.DataFrame): DataFrame com dados
        target (str): Coluna target
        metodo (str): Método de cálculo ('mutual_info', 'chi2', 'correlation')
        top_n (int): Número top de variáveis para mostrar
        titulo (str): Título do gráfico
        
    Returns:
        plt.Figure: Figura matplotlib
    """
    from sklearn.feature_selection import mutual_info_classif, chi2
    from sklearn.preprocessing import LabelEncoder
    
    # Preparar dados
    X = df.drop(columns=[target] if target in df.columns else [])
    y = df[target] if target in df.columns else pd.Series(np.zeros(len(df)))
    
    # Codificar variáveis categóricas
    X_encoded = X.copy()
    le = LabelEncoder()
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X_encoded[col] = le.fit_transform(X[col].astype(str))
    
    # Codificar target se for categórico
    if y.dtype == 'object' or y.dtype.name == 'category':
        y_encoded = le.fit_transform(y)
    else:
        y_encoded = y
    
    # Calcular importância
    if metodo == 'mutual_info':
        importancia = mutual_info_classif(X_encoded, y_encoded, random_state=42)
    elif metodo == 'chi2':
        importancia = chi2(X_encoded, y_encoded)[0]
    else:  # correlation
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        importancia = X[numeric_cols].corrwith(y_encoded).abs().values
        # Preencher com zeros para colunas não numéricas
        full_importancia = np.zeros(len(X.columns))
        for i, col in enumerate(X.columns):
            if col in numeric_cols:
                full_importancia[i] = importancia[list(numeric_cols).index(col)]
        importancia = full_importancia
    
    # Criar DataFrame com resultados
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importancia
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plotar gráfico
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(feature_importance)), 
                  feature_importance['importance'],
                  color=plt.cm.viridis(np.linspace(0, 1, len(feature_importance))))
    
    ax.set_yticks(range(len(feature_importance)))
    ax.set_yticklabels(feature_importance['feature'])
    ax.set_xlabel('Importância')
    ax.set_title(titulo, fontsize=16, fontweight='bold')
    
    # Adicionar valores nas barras
    for i, (idx, row) in enumerate(feature_importance.iterrows()):
        ax.text(row['importance'] * 1.01, i, f'{row["importance"]:.3f}', 
               va='center', fontweight='bold')
    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    return fig

def plotar_boxplots_numericos(df: pd.DataFrame,
                            colunas_numericas: List[str],
                            target: str = 'Churn',
                            titulo: str = "Distribuição por Target") -> plt.Figure:
    """
    Cria boxplots para variáveis numéricas agrupadas por target.
    
    Args:
        df (pd.DataFrame): DataFrame com dados
        colunas_numericas (List[str]): Lista de colunas numéricas
        target (str): Coluna target para agrupamento
        titulo (str): Título do gráfico
        
    Returns:
        plt.Figure: Figura matplotlib
    """
    n_cols = min(3, len(colunas_numericas))
    n_rows = (len(colunas_numericas) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, coluna in enumerate(colunas_numericas):
        if i < len(axes) and coluna in df.columns:
            # Plotar boxplot
            sns.boxplot(data=df, x=target, y=coluna, ax=axes[i])
            axes[i].set_title(f'{coluna} por {target}')
            axes[i].set_xlabel(target)
            axes[i].set_ylabel(coluna)
            
            # Adicionar teste estatístico
            groups = [df[df[target] == grupo][coluna].dropna() for grupo in df[target].unique()]
            if len(groups) == 2 and len(groups[0]) > 0 and len(groups[1]) > 0:
                stat, p_value = stats.ttest_ind(groups[0], groups[1])
                axes[i].text(0.5, 0.95, f'p-value: {p_value:.3f}', 
                           transform=axes[i].transAxes, ha='center', 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Remover eixos vazios
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(titulo, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def criar_grafico_interativo(df: pd.DataFrame,
                           x_col: str,
                           y_col: str,
                           color_col: Optional[str] = None,
                           size_col: Optional[str] = None,
                           titulo: str = "Gráfico Interativo") -> go.Figure:
    """
    Cria gráfico interativo usando Plotly.
    
    Args:
        df (pd.DataFrame): DataFrame com dados
        x_col (str): Coluna para eixo x
        y_col (str): Coluna para eixo y
        color_col (str, optional): Coluna para colorir pontos
        size_col (str, optional): Coluna para tamanho dos pontos
        titulo (str): Título do gráfico
        
    Returns:
        go.Figure: Figura Plotly
    """
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col,
                    title=titulo, hover_data=df.columns,
                    template='plotly_white')
    
    fig.update_layout(
        title_font_size=20,
        title_x=0.5,
        width=1000,
        height=600
    )
    
    return fig

def salvar_grafico(fig: Union[plt.Figure, go.Figure],
                  caminho: str,
                  formato: str = 'png',
                  dpi: int = 300,
                  **kwargs) -> None:
    """
    Salva gráfico em diferentes formatos.
    
    Args:
        fig: Figura matplotlib ou plotly
        caminho (str): Caminho para salvar
        formato (str): Formato do arquivo ('png', 'jpg', 'svg', 'pdf', 'html')
        dpi (int): Resolução para imagens
        **kwargs: Argumentos adicionais
    """
    import os
    from pathlib import Path
    
    # Criar diretório se não existir
    Path(os.path.dirname(caminho)).mkdir(parents=True, exist_ok=True)
    
    try:
        if isinstance(fig, plt.Figure):
            fig.savefig(caminho, format=formato, dpi=dpi, bbox_inches='tight', **kwargs)
            print(f"💾 Gráfico salvo: {caminho}")
        elif isinstance(fig, go.Figure):
            if formato == 'html':
                fig.write_html(caminho, **kwargs)
            else:
                fig.write_image(caminho, format=formato, **kwargs)
            print(f"💾 Gráfico interativo salvo: {caminho}")
    except Exception as e:
        print(f"❌ Erro ao salvar gráfico: {e}")

# Exemplo de uso
if __name__ == "__main__":
    print("🎨 Módulo de visualização - Telco Customer Churn")
    print()
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