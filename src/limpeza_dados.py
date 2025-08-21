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
Versão: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Configuração de display para melhor visualização
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 50)

def diagnosticar_problemas(df: pd.DataFrame, detalhado: bool = True) -> Dict[str, Any]:
    """
    Realiza diagnóstico completo da qualidade dos dados com relatório detalhado.
    
    Args:
        df (pd.DataFrame): DataFrame com dados brutos
        detalhado (bool): Se True, mostra análise detalhada de cada coluna
        
    Returns:
        Dict[str, Any]: Dicionário com métricas de qualidade
    """
    print("🔍 INICIANDO DIAGNÓSTICO DE QUALIDADE DE DADOS")
    print("=" * 60)
    
    resultado = {
        'dimensoes': df.shape,
        'memoria_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'colunas_com_ausentes': [],
        'total_valores_ausentes': 0,
        'tipos_incorretos': [],
        'inconsistencias_categoricas': []
    }
    
    # Informações básicas
    print(f"📊 Dimensões do dataset: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print(f"📦 Consumo de memória: {resultado['memoria_mb']:.2f} MB")
    
    # Tipos de dados
    print("\n🎯 TIPOS DE DADOS ORIGINAIS:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")
        # Identificar tipos potencialmente incorretos
        if df[col].dtype == 'object' and df[col].nunique() < 10:
            resultado['tipos_incorretos'].append(col)
    
    # Valores ausentes
    print("\n❌ VALORES AUSENTES:")
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({
        'Valores_Ausentes': missing,
        'Percentual': missing_pct.round(2)
    })
    
    missing_cols = missing_df[missing_df['Valores_Ausentes'] > 0]
    if not missing_cols.empty:
        print(missing_cols)
        resultado['colunas_com_ausentes'] = missing_cols.index.tolist()
        resultado['total_valores_ausentes'] = missing.sum()
    else:
        print("  ✅ Nenhum valor ausente encontrado")
    
    # Análise de colunas categóricas
    if detalhado:
        print("\n🎭 ANÁLISE DETALHADA DE COLUNAS CATEGÓRICAS:")
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            unique_vals = df[col].unique()
            print(f"\n  {col} ({len(unique_vals)} valores únicos):")
            print(f"    Valores: {unique_vals}")
            
            # Verificar inconsistências comuns
            if any(' ' in str(x) for x in unique_vals if pd.notna(x)):
                resultado['inconsistencias_categoricas'].append(col)
                print(f"    ⚠️  Possível inconsistência: valores com espaços")
    
    # Estatísticas para colunas numéricas
    print("\n📈 ESTATÍSTICAS DESCRITIVAS PARA VARIÁVEIS NUMÉRICAS:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        print(df[numeric_cols].describe().round(2))
    else:
        print("  ℹ️  Nenhuma variável numérica encontrada")
    
    # Resumo do diagnóstico
    print("\n📋 RESUMO DO DIAGNÓSTICO:")
    print(f"  • Colunas com valores ausentes: {len(resultado['colunas_com_ausentes'])}")
    print(f"  • Total de valores ausentes: {resultado['total_valores_ausentes']}")
    print(f"  • Colunas com tipos potencialmente incorretos: {len(resultado['tipos_incorretos'])}")
    print(f"  • Possíveis inconsistências categóricas: {len(resultado['inconsistencias_categoricas'])}")
    
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
    print("\n🛠️ CORRIGINDO TIPOS DE DADOS")
    print("=" * 50)
    
    df_clean = df.copy()
    
    # Configuração padrão de tipos para o dataset Telco
    if config_tipos is None:
        config_tipos = {
            'SeniorCitizen': 'category',
            'Partner': 'category',
            'Dependents': 'category',
            'tenure': 'int32',
            'PhoneService': 'category',
            'MultipleLines': 'category',
            'InternetService': 'category',
            'OnlineSecurity': 'category',
            'OnlineBackup': 'category',
            'DeviceProtection': 'category',
            'TechSupport': 'category',
            'StreamingTV': 'category',
            'StreamingMovies': 'category',
            'Contract': 'category',
            'PaperlessBilling': 'category',
            'PaymentMethod': 'category',
            'Churn': 'category',
            'TotalCharges': 'float32',
            'MonthlyCharges': 'float32'
        }
    
    conversoes_realizadas = 0
    
    for col, target_dtype in config_tipos.items():
        if col in df_clean.columns:
            original_dtype = str(df_clean[col].dtype)
            
            try:
                if target_dtype == 'category':
                    df_clean[col] = df_clean[col].astype('category')
                elif 'int' in target_dtype:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').astype(target_dtype)
                elif 'float' in target_dtype:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').astype(target_dtype)
                else:
                    df_clean[col] = df_clean[col].astype(target_dtype)
                
                novo_dtype = str(df_clean[col].dtype)
                print(f"✅ {col}: {original_dtype} → {novo_dtype}")
                conversoes_realizadas += 1
                
            except Exception as e:
                print(f"❌ Erro ao converter {col}: {e}")
    
    # Economia de memória
    memoria_antes = df.memory_usage(deep=True).sum() / 1024**2
    memoria_depois = df_clean.memory_usage(deep=True).sum() / 1024**2
    economia = memoria_antes - memoria_depois
    
    print(f"\n💾 Economia de memória: {economia:.2f} MB ({memoria_antes:.2f} → {memoria_depois:.2f} MB)")
    print(f"🔢 Total de conversões realizadas: {conversoes_realizadas}")
    
    return df_clean

def tratar_valores_ausentes(df: pd.DataFrame, estrategias: Optional[Dict] = None) -> pd.DataFrame:
    """
    Trata valores ausentes usando estratégias avançadas e configuráveis.
    
    Args:
        df (pd.DataFrame): DataFrame com tipos corrigidos
        estrategias (Dict, optional): Estratégias específicas por coluna
        
    Returns:
        pd.DataFrame: DataFrame sem valores ausentes
    """
    print("\n🔧 TRATANDO VALORES AUSENTES")
    print("=" * 50)
    
    df_clean = df.copy()
    ausentes_antes = df_clean.isnull().sum().sum()
    
    if ausentes_antes == 0:
        print("✅ Nenhum valor ausente encontrado para tratamento")
        return df_clean
    
    # Estratégias padrão para o dataset Telco
    if estrategias is None:
        estrategias = {
            'TotalCharges': {'estrategia': 'mediana', 'params': {}},
            'Dependents': {'estrategia': 'moda', 'params': {}},
            'PhoneService': {'estrategia': 'moda', 'params': {}},
            'InternetService': {'estrategia': 'valor_constante', 'params': {'valor': 'No'}},
            'default': {'estrategia': 'remover', 'params': {}}
        }
    
    print(f"🔍 Valores ausentes antes do tratamento: {ausentes_antes}")
    
    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            n_ausentes = df_clean[col].isnull().sum()
            
            # Obter estratégia para esta coluna ou usar padrão
            estrategia_col = estrategias.get(col, estrategias['default'])
            estrategia = estrategia_col['estrategia']
            params = estrategia_col['params']
            
            print(f"\n📦 {col}: {n_ausentes} valores ausentes")
            print(f"   Estratégia: {estrategia}")
            
            try:
                if estrategia == 'mediana':
                    fill_value = df_clean[col].median()
                    df_clean[col].fillna(fill_value, inplace=True)
                    print(f"   ✅ Preenchido com mediana: {fill_value:.2f}")
                
                elif estrategia == 'moda':
                    fill_value = df_clean[col].mode()[0]
                    df_clean[col].fillna(fill_value, inplace=True)
                    print(f"   ✅ Preenchido com moda: {fill_value}")
                
                elif estrategia == 'valor_constante':
                    fill_value = params.get('valor', 0)
                    df_clean[col].fillna(fill_value, inplace=True)
                    print(f"   ✅ Preenchido com valor constante: {fill_value}")
                
                elif estrategia == 'remover':
                    df_clean = df_clean.dropna(subset=[col])
                    print(f"   ✅ Removidas {n_ausentes} linhas com valores ausentes")
                
                elif estrategia == 'interpolar':
                    df_clean[col] = df_clean[col].interpolate(**params)
                    print(f"   ✅ Interpolação aplicada")
                
                else:
                    print(f"   ⚠️  Estratégia '{estrategia}' não reconhecida")
            
            except Exception as e:
                print(f"   ❌ Erro ao tratar {col}: {e}")
    
    ausentes_depois = df_clean.isnull().sum().sum()
    print(f"\n🎯 Valores ausentes após tratamento: {ausentes_depois}")
    
    if ausentes_depois == 0:
        print("✅ Todos os valores ausentes foram tratados com sucesso!")
    else:
        print(f"⚠️  Ainda existem {ausentes_depois} valores ausentes")
    
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
    print("\n🎭 NORMALIZANDO VALORES CATEGÓRICOS")
    print("=" * 50)
    
    df_clean = df.copy()
    
    # Mapeamentos padrão para o dataset Telco
    if mapeamentos is None:
        mapeamentos = {
            'Partner': {'Yes': 'Sim', 'No': 'Não', 'Y': 'Sim', 'N': 'Não', 'Yes': 'Sim', 'No': 'Não'},
            'Dependents': {'Yes': 'Sim', 'No': 'Não', 'Y': 'Sim', 'N': 'Não'},
            'Churn': {'Yes': 'Sim', 'No': 'Não', 'Churned': 'Sim', 'Stayed': 'Não'},
            'PhoneService': {'Yes': 'Sim', 'No': 'Não'},
            'PaperlessBilling': {'Yes': 'Sim', 'No': 'Não'},
            'MultipleLines': {'No phone service': 'Sem serviço telefônico'},
            'OnlineSecurity': {'No internet service': 'Sem serviço de internet'},
            'OnlineBackup': {'No internet service': 'Sem serviço de internet'},
            'DeviceProtection': {'No internet service': 'Sem serviço de internet'},
            'TechSupport': {'No internet service': 'Sem serviço de internet'},
            'StreamingTV': {'No internet service': 'Sem serviço de internet'},
            'StreamingMovies': {'No internet service': 'Sem serviço de internet'}
        }
    
    normalizacoes_realizadas = 0
    
    for col, mapping in mapeamentos.items():
        if col in df_clean.columns:
            valores_antes = df_clean[col].nunique()
            
            # Aplicar mapeamento
            df_clean[col] = df_clean[col].replace(mapping)
            
            # Para colunas categóricas, atualizar categorias
            if hasattr(df_clean[col], 'cat'):
                novas_categorias = list(df_clean[col].unique())
                df_clean[col] = df_clean[col].cat.set_categories(novas_categorias)
            
            valores_depois = df_clean[col].nunique()
            
            print(f"✅ {col}: {valores_antes} → {valores_depois} valores únicos")
            normalizacoes_realizadas += 1
    
    print(f"\n🔢 Normalizações realizadas: {normalizacoes_realizadas}")
    
    return df_clean

def criar_novas_variaveis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria novas features derivadas para enriquecer a análise.
    
    Args:
        df (pd.DataFrame): DataFrame com dados limpos
        
    Returns:
        pd.DataFrame: DataFrame com novas variáveis
    """
    print("\n✨ CRIANDO NOVAS VARIÁVEIS")
    print("=" * 50)
    
    df_clean = df.copy()
    novas_variaveis = []
    
    # 1. Categorização de tenure (tempo como cliente)
    if 'tenure' in df_clean.columns:
        bins = [0, 12, 24, 36, 48, 60, 72, np.inf]
        labels = ['0-1ano', '1-2anos', '2-3anos', '3-4anos', '4-5anos', '5-6anos', '6+anos']
        
        df_clean['TenureGroup'] = pd.cut(df_clean['tenure'], bins=bins, labels=labels)
        df_clean['TenureGroup'] = df_clean['TenureGroup'].astype('category')
        novas_variaveis.append('TenureGroup')
        print("✅ Criada variável TenureGroup")
    
    # 2. Total de serviços contratados
    servicos = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    servicos_presentes = [col for col in servicos if col in df_clean.columns]
    
    if servicos_presentes:
        df_clean['TotalServicos'] = 0
        for servico in servicos_presentes:
            # Considerar que tem o serviço se não for 'No' ou 'No internet service'
            tem_servico = (~df_clean[servico].isin(['No', 'Sem serviço de internet', 'No internet service'])).astype(int)
            df_clean['TotalServicos'] += tem_servico
        
        novas_variaveis.append('TotalServicos')
        print(f"✅ Criada variável TotalServicos (média: {df_clean['TotalServicos'].mean():.2f})")
    
    # 3. Cliente com dependentes e parceiro
    if 'Partner' in df_clean.columns and 'Dependents' in df_clean.columns:
        df_clean['TemFamilia'] = ((df_clean['Partner'] == 'Sim') | (df_clean['Dependents'] == 'Sim')).astype(int)
        novas_variaveis.append('TemFamilia')
        print("✅ Criada variável TemFamilia")
    
    # 4. Tipo de cliente baseado no contrato e tempo
    if 'Contract' in df_clean.columns and 'tenure' in df_clean.columns:
        conditions = [
            (df_clean['Contract'] == 'Month-to-month'),
            (df_clean['Contract'] == 'One year') & (df_clean['tenure'] < 12),
            (df_clean['Contract'] == 'One year') & (df_clean['tenure'] >= 12),
            (df_clean['Contract'] == 'Two year') & (df_clean['tenure'] < 24),
            (df_clean['Contract'] == 'Two year') & (df_clean['tenure'] >= 24)
        ]
        choices = ['Novo_Mensal', 'Novo_Anual', 'Estavel_Anual', 'Novo_Bianual', 'Estavel_Bianual']
        
        df_clean['TipoCliente'] = np.select(conditions, choices, default='Outro')
        df_clean['TipoCliente'] = df_clean['TipoCliente'].astype('category')
        novas_variaveis.append('TipoCliente')
        print("✅ Criada variável TipoCliente")
    
    # 5. Valor médio mensal por serviço
    if 'MonthlyCharges' in df_clean.columns and 'TotalServicos' in df_clean.columns:
        df_clean['CustoPorServico'] = np.where(
            df_clean['TotalServicos'] > 0,
            df_clean['MonthlyCharges'] / df_clean['TotalServicos'],
            0
        )
        novas_variaveis.append('CustoPorServico')
        print("✅ Criada variável CustoPorServico")
    
    print(f"\n🎯 Total de novas variáveis criadas: {len(novas_variaveis)}")
    print(f"📊 Variáveis: {', '.join(novas_variaveis)}")
    
    return df_clean

def validar_qualidade_dados(df: pd.DataFrame) -> bool:
    """
    Realiza validação final da qualidade dos dados após limpeza.
    
    Args:
        df (pd.DataFrame): DataFrame processado
        
    Returns:
        bool: True se a qualidade for satisfatória
    """
    print("\n✅ VALIDAÇÃO FINAL DA QUALIDADE")
    print("=" * 50)
    
    qualidade_ok = True
    testes = []
    
    # 1. Verificar valores ausentes
    missing_total = df.isnull().sum().sum()
    testes.append(('Valores ausentes', missing_total == 0, missing_total))
    
    # 2. Verificar tipos de dados essenciais
    tipos_essenciais = {
        'Churn': 'category',
        'SeniorCitizen': 'category',
        'Contract': 'category',
        'TotalCharges': 'float32',
        'MonthlyCharges': 'float32'
    }
    
    for col, expected_type in tipos_essenciais.items():
        if col in df.columns:
            actual_type = str(df[col].dtype)
            tipo_correto = expected_type in actual_type
            testes.append((f'Tipo {col}', tipo_correto, actual_type))
            if not tipo_correto:
                qualidade_ok = False
    
    # 3. Verificar consistência de valores categóricos
    categorias_consistentes = {
        'Churn': ['Sim', 'Não'],
        'Partner': ['Sim', 'Não'],
        'Dependents': ['Sim', 'Não']
    }
    
    for col, expected_values in categorias_consistentes.items():
        if col in df.columns:
            unique_values = set(df[col].unique())
            is_subset = unique_values.issubset(set(expected_values))
            testes.append((f'Valores {col}', is_subset, list(unique_values)))
            if not is_subset:
                qualidade_ok = False
    
    # 4. Verificar faixas de valores numéricos
    if 'MonthlyCharges' in df.columns:
        dentro_faixa = (df['MonthlyCharges'] >= 0).all()
        testes.append(('Faixa MonthlyCharges', dentro_faixa, 
                      f"{df['MonthlyCharges'].min():.2f}-{df['MonthlyCharges'].max():.2f}"))
    
    # Exibir resultados dos testes
    print("📋 RESULTADOS DOS TESTES:")
    for teste, resultado, info in testes:
        status = "✅" if resultado else "❌"
        print(f"  {status} {teste}: {resultado} ({info})")
    
    # Estatísticas finais
    print(f"\n📊 ESTATÍSTICAS FINAIS:")
    print(f"  • Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print(f"  • Memória: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"  • Variáveis categóricas: {len(df.select_dtypes(include='category').columns)}")
    print(f"  • Variáveis numéricas: {len(df.select_dtypes(include=np.number).columns)}")
    
    if qualidade_ok:
        print("\n🎉 QUALIDADE DOS DADOS VALIDADA COM SUCESSO!")
        print("📁 Dataset pronto para análise e modelagem")
    else:
        print("\n⚠️  ALERTA: Foram identificados problemas na qualidade dos dados")
    
    return qualidade_ok

def pipeline_limpeza_completa(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Executa o pipeline completo de limpeza de dados.
    
    Args:
        df (pd.DataFrame): DataFrame com dados brutos
        verbose (bool): Se True, mostra detalhes de cada etapa
        
    Returns:
        pd.DataFrame: DataFrame limpo e processado
    """
    if verbose:
        print("🚀 INICIANDO PIPELINE COMPLETO DE LIMPEZA")
        print("=" * 60)
        print(f"📥 Input: {df.shape[0]} linhas, {df.shape[1]} colunas")
    
    # Executar pipeline
    df_clean = (df
                .pipe(corrigir_tipos_dados)
                .pipe(tratar_valores_ausentes)
                .pipe(normalizar_categoricas)
                .pipe(criar_novas_variaveis))
    
    # Validação final
    qualidade_ok = validar_qualidade_dados(df_clean)
    
    if verbose:
        print(f"\n📤 Output: {df_clean.shape[0]} linhas, {df_clean.shape[1]} colunas")
        if qualidade_ok:
            print("🎉 PIPELINE CONCLUÍDO COM SUCESSO!")
        else:
            print("⚠️  PIPELINE CONCLUÍDO COM AVISOS!")
    
    return df_clean

# Exemplo de uso
if __name__ == "__main__":
    print("🔧 Módulo de limpeza de dados - Telco Customer Churn")
    print("💡 Use as funções individualmente ou pipeline_limpeza_completa()")
    print("📚 Documentação disponível nos docstrings de cada função")