"""
Módulo de validação e controle de qualidade de dados para o projeto Telco Customer Churn.
Funções para validação de estrutura, consistência e qualidade dos dados.

Autor: Marlon Henrique
Data: 2025
Versão: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
from dataclasses import dataclass
from enum import Enum
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suprimir warnings
warnings.filterwarnings('ignore')

class TipoValidacao(Enum):
    """Tipos de validação disponíveis."""
    ESTRUTURA = "estrutura"
    TIPOS = "tipos"
    AUSENTES = "ausentes"
    CATEGORIAS = "categorias"
    RANGE = "range"
    CONSISTENCIA = "consistencia"
    DUPLICATAS = "duplicatas"

@dataclass
class ResultadoValidacao:
    """Classe para armazenar resultados de validação."""
    sucesso: bool
    mensagem: str
    detalhes: Dict[str, Any]
    tipo: TipoValidacao

class ValidadorDados:
    """Classe principal para validação de dados."""
    
    def __init__(self, df: pd.DataFrame, nome_dataset: str = "dataset"):
        """
        Inicializa o validador com um DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame para validação
            nome_dataset (str): Nome do dataset para logging
        """
        self.df = df.copy()
        self.nome_dataset = nome_dataset
        self.resultados: List[ResultadoValidacao] = []
    
    def validar_estrutura(self, colunas_esperadas: List[str], 
                         min_linhas: int = 1) -> ResultadoValidacao:
        """
        Valida a estrutura básica do dataset.
        
        Args:
            colunas_esperadas (List[str]): Lista de colunas esperadas
            min_linhas (int): Número mínimo de linhas esperadas
            
        Returns:
            ResultadoValidacao: Resultado da validação
        """
        detalhes = {
            'colunas_esperadas': colunas_esperadas,
            'colunas_encontradas': self.df.columns.tolist(),
            'linhas_encontradas': len(self.df)
        }
        
        # Verificar colunas
        colunas_faltantes = set(colunas_esperadas) - set(self.df.columns)
        colunas_extra = set(self.df.columns) - set(colunas_esperadas)
        
        # Verificar número de linhas
        linhas_ok = len(self.df) >= min_linhas
        
        sucesso = not colunas_faltantes and linhas_ok
        mensagem = ""
        
        if not sucesso:
            if colunas_faltantes:
                mensagem += f"Colunas faltantes: {list(colunas_faltantes)}. "
            if not linhas_ok:
                mensagem += f"Mínimo de {min_linhas} linhas esperado, encontradas {len(self.df)}. "
            if colunas_extra:
                mensagem += f"Colunas extras: {list(colunas_extra)}"
        else:
            mensagem = "Estrutura do dataset válida"
            if colunas_extra:
                mensagem += f" (colunas extras: {list(colunas_extra)})"
        
        resultado = ResultadoValidacao(
            sucesso=sucesso,
            mensagem=mensagem,
            detalhes=detalhes,
            tipo=TipoValidacao.ESTRUTURA
        )
        
        self.resultados.append(resultado)
        return resultado
    
    def validar_tipos(self, mapeamento_tipos: Dict[str, str]) -> ResultadoValidacao:
        """
        Valida os tipos de dados das colunas.
        
        Args:
            mapeamento_tipos (Dict[str, str]): Mapeamento coluna → tipo esperado
            
        Returns:
            ResultadoValidacao: Resultado da validação
        """
        detalhes = {'tipos_incorretos': {}}
        tipos_incorretos = False
        
        for coluna, tipo_esperado in mapeamento_tipos.items():
            if coluna not in self.df.columns:
                continue
                
            tipo_atual = str(self.df[coluna].dtype)
            tipo_correto = tipo_esperado.lower() in tipo_atual.lower()
            
            if not tipo_correto:
                tipos_incorretos = True
                detalhes['tipos_incorretos'][coluna] = {
                    'esperado': tipo_esperado,
                    'encontrado': tipo_atual
                }
        
        sucesso = not tipos_incorretos
        mensagem = "Tipos de dados válidos" if sucesso else "Tipos de dados incorretos encontrados"
        
        resultado = ResultadoValidacao(
            sucesso=sucesso,
            mensagem=mensagem,
            detalhes=detalhes,
            tipo=TipoValidacao.TIPOS
        )
        
        self.resultados.append(resultado)
        return resultado
    
    def verificar_valores_ausentes(self, colunas: Optional[List[str]] = None, 
                                 limite_percentual: float = 5.0) -> ResultadoValidacao:
        """
        Verifica valores ausentes no dataset.
        
        Args:
            colunas (List[str], optional): Colunas específicas para verificar
            limite_percentual (float): Limite percentual para alerta
            
        Returns:
            ResultadoValidacao: Resultado da verificação
        """
        if colunas is None:
            colunas = self.df.columns.tolist()
        
        colunas_verificar = [col for col in colunas if col in self.df.columns]
        
        detalhes = {'ausentes_por_coluna': {}, 'total_ausentes': 0}
        tem_ausentes = False
        tem_excesso_ausentes = False
        
        for coluna in colunas_verificar:
            n_ausentes = self.df[coluna].isnull().sum()
            pct_ausentes = (n_ausentes / len(self.df)) * 100 if len(self.df) > 0 else 0
            
            detalhes['ausentes_por_coluna'][coluna] = {
                'quantidade': n_ausentes,
                'percentual': pct_ausentes,
                'excede_limite': pct_ausentes > limite_percentual
            }
            
            detalhes['total_ausentes'] += n_ausentes
            
            if n_ausentes > 0:
                tem_ausentes = True
            if pct_ausentes > limite_percentual:
                tem_excesso_ausentes = True
        
        sucesso = not tem_ausentes
        mensagem = "Nenhum valor ausente encontrado" if sucesso else "Valores ausentes encontrados"
        
        if tem_excesso_ausentes:
            mensagem += f" (alguns excedem {limite_percentual}%)"
        
        resultado = ResultadoValidacao(
            sucesso=sucesso,
            mensagem=mensagem,
            detalhes=detalhes,
            tipo=TipoValidacao.AUSENTES
        )
        
        self.resultados.append(resultado)
        return resultado
    
    def verificar_consistencia_categorica(self, coluna: str, 
                                        valores_esperados: List[str],
                                        ignorar_nulos: bool = True) -> ResultadoValidacao:
        """
        Verifica consistência de valores em coluna categórica.
        
        Args:
            coluna (str): Nome da coluna
            valores_esperados (List[str]): Valores esperados
            ignorar_nulos (bool): Se True, ignora valores nulos na verificação
            
        Returns:
            ResultadoValidacao: Resultado da verificação
        """
        if coluna not in self.df.columns:
            return ResultadoValidacao(
                sucesso=False,
                mensagem=f"Coluna '{coluna}' não encontrada",
                detalhes={},
                tipo=TipoValidacao.CATEGORIAS
            )
        
        if ignorar_nulos:
            valores_atuais = self.df[coluna].dropna().unique()
        else:
            valores_atuais = self.df[coluna].unique()
        
        valores_inesperados = set(valores_atuais) - set(valores_esperados)
        
        detalhes = {
            'coluna': coluna,
            'valores_esperados': valores_esperados,
            'valores_encontrados': list(valores_atuais),
            'valores_inesperados': list(valores_inesperados),
            'total_valores_inesperados': len(valores_inesperados)
        }
        
        sucesso = len(valores_inesperados) == 0
        mensagem = f"Valores categóricos consistentes em '{coluna}'"
        
        if not sucesso:
            mensagem = f"Valores inesperados em '{coluna}': {list(valores_inesperados)}"
        
        resultado = ResultadoValidacao(
            sucesso=sucesso,
            mensagem=mensagem,
            detalhes=detalhes,
            tipo=TipoValidacao.CATEGORIAS
        )
        
        self.resultados.append(resultado)
        return resultado
    
    def validar_range_numerico(self, coluna: str, 
                             min_val: Optional[float] = None,
                             max_val: Optional[float] = None) -> ResultadoValidacao:
        """
        Valida se valores numéricos estão dentro do range esperado.
        
        Args:
            coluna (str): Nome da coluna
            min_val (float): Valor mínimo permitido
            max_val (float): Valor máximo permitido
            
        Returns:
            ResultadoValidacao: Resultado da validação
        """
        if coluna not in self.df.columns:
            return ResultadoValidacao(
                sucesso=False,
                mensagem=f"Coluna '{coluna}' não encontrada",
                detalhes={},
                tipo=TipoValidacao.RANGE
            )
        
        if not pd.api.types.is_numeric_dtype(self.df[coluna]):
            return ResultadoValidacao(
                sucesso=False,
                mensagem=f"Coluna '{coluna}' não é numérica",
                detalhes={},
                tipo=TipoValidacao.RANGE
            )
        
        # Filtrar valores não nulos
        valores = self.df[coluna].dropna()
        fora_range = pd.Series(False, index=valores.index)
        
        if min_val is not None:
            fora_range = fora_range | (valores < min_val)
        if max_val is not None:
            fora_range = fora_range | (valores > max_val)
        
        n_fora_range = fora_range.sum()
        pct_fora_range = (n_fora_range / len(valores)) * 100 if len(valores) > 0 else 0
        
        detalhes = {
            'coluna': coluna,
            'min_esperado': min_val,
            'max_esperado': max_val,
            'min_encontrado': valores.min() if len(valores) > 0 else None,
            'max_encontrado': valores.max() if len(valores) > 0 else None,
            'valores_fora_range': n_fora_range,
            'percentual_fora_range': pct_fora_range
        }
        
        sucesso = n_fora_range == 0
        mensagem = f"Valores de '{coluna}' dentro do range esperado"
        
        if not sucesso:
            mensagem = f"{n_fora_range} valores fora do range em '{coluna}'"
        
        resultado = ResultadoValidacao(
            sucesso=sucesso,
            mensagem=mensagem,
            detalhes=detalhes,
            tipo=TipoValidacao.RANGE
        )
        
        self.resultados.append(resultado)
        return resultado
    
    def verificar_duplicatas(self, subset: Optional[List[str]] = None) -> ResultadoValidacao:
        """
        Verifica linhas duplicadas no dataset.
        
        Args:
            subset (List[str], optional): Subconjunto de colunas para verificação
            
        Returns:
            ResultadoValidacao: Resultado da verificação
        """
        if subset is None:
            duplicatas = self.df.duplicated().sum()
            detalhes = {'tipo': 'completas', 'duplicatas': duplicatas}
        else:
            subset_existente = [col for col in subset if col in self.df.columns]
            duplicatas = self.df.duplicated(subset=subset_existente).sum()
            detalhes = {'tipo': 'parciais', 'duplicatas': duplicatas, 'subset': subset_existente}
        
        sucesso = duplicatas == 0
        mensagem = "Nenhuma duplicata encontrada" if sucesso else f"{duplicatas} duplicatas encontradas"
        
        resultado = ResultadoValidacao(
            sucesso=sucesso,
            mensagem=mensagem,
            detalhes=detalhes,
            tipo=TipoValidacao.DUPLICATAS
        )
        
        self.resultados.append(resultado)
        return resultado
    
    def validar_consistencia_cruzada(self, regras: List[Dict[str, Any]]) -> ResultadoValidacao:
        """
        Valida regras de consistência cruzada entre colunas.
        
        Args:
            regras (List[Dict]): Lista de regras de validação
            
        Returns:
            ResultadoValidacao: Resultado da validação
        """
        violacoes = []
        detalhes = {'regras_violadas': []}
        
        for i, regra in enumerate(regras):
            condicao = regra.get('condicao')
            mensagem_erro = regra.get('mensagem', f'Violacao regra {i+1}')
            
            try:
                # Aplicar condição
                mask = self.df.eval(condicao) if isinstance(condicao, str) else condicao(self.df)
                n_violacoes = (~mask).sum()
                
                if n_violacoes > 0:
                    violacoes.append({
                        'regra': i + 1,
                        'condicao': condicao,
                        'violacoes': n_violacoes,
                        'mensagem': mensagem_erro
                    })
                    
            except Exception as e:
                violacoes.append({
                    'regra': i + 1,
                    'condicao': condicao,
                    'erro': str(e),
                    'mensagem': 'Erro na execução da regra'
                })
        
        detalhes['regras_violadas'] = violacoes
        sucesso = len(violacoes) == 0
        mensagem = "Todas as regras de consistência atendidas" if sucesso else "Regras de consistência violadas"
        
        resultado = ResultadoValidacao(
            sucesso=sucesso,
            mensagem=mensagem,
            detalhes=detalhes,
            tipo=TipoValidacao.CONSISTENCIA
        )
        
        self.resultados.append(resultado)
        return resultado
    
    def gerar_relatorio_validacao(self) -> Dict[str, Any]:
        """
        Gera relatório completo de validação.
        
        Returns:
            Dict[str, Any]: Relatório detalhado
        """
        total_validacoes = len(self.resultados)
        validacoes_sucesso = sum(1 for r in self.resultados if r.sucesso)
        taxa_sucesso = (validacoes_sucesso / total_validacoes * 100) if total_validacoes > 0 else 0
        
        relatorio = {
            'dataset': self.nome_dataset,
            'timestamp': pd.Timestamp.now().isoformat(),
            'estatisticas_gerais': {
                'total_validacoes': total_validacoes,
                'validacoes_sucesso': validacoes_sucesso,
                'validacoes_falha': total_validacoes - validacoes_sucesso,
                'taxa_sucesso': taxa_sucesso
            },
            'resultados_por_tipo': {},
            'resumo': [],
            'detalhes': []
        }
        
        # Agrupar resultados por tipo
        for resultado in self.resultados:
            tipo = resultado.tipo.value
            if tipo not in relatorio['resultados_por_tipo']:
                relatorio['resultados_por_tipo'][tipo] = {
                    'total': 0,
                    'sucesso': 0,
                    'falha': 0
                }
            
            relatorio['resultados_por_tipo'][tipo]['total'] += 1
            if resultado.sucesso:
                relatorio['resultados_por_tipo'][tipo]['sucesso'] += 1
            else:
                relatorio['resultados_por_tipo'][tipo]['falha'] += 1
            
            relatorio['resumo'].append({
                'tipo': tipo,
                'sucesso': resultado.sucesso,
                'mensagem': resultado.mensagem
            })
            
            relatorio['detalhes'].append({
                'tipo': tipo,
                'sucesso': resultado.sucesso,
                'mensagem': resultado.mensagem,
                'detalhes': resultado.detalhes
            })
        
        return relatorio
    
    def imprimir_relatorio(self) -> None:
        """Imprime relatório de validação formatado."""
        relatorio = self.gerar_relatorio_validacao()
        
        print(f"📋 RELATÓRIO DE VALIDAÇÃO - {self.nome_dataset}")
        print("=" * 60)
        print(f"📅 {relatorio['timestamp']}")
        print(f"📊 Estatísticas: {relatorio['estatisticas_gerais']['taxa_sucesso']:.1f}% de sucesso")
        print()
        
        # Resultados por tipo
        for tipo, stats in relatorio['resultados_por_tipo'].items():
            status = "✅" if stats['falha'] == 0 else "❌"
            print(f"{status} {tipo.upper()}: {stats['sucesso']}/{stats['total']}")
        
        print()
        print("📝 DETALHES:")
        for resultado in relatorio['resumo']:
            status = "✅" if resultado['sucesso'] else "❌"
            print(f"  {status} {resultado['tipo']}: {resultado['mensagem']}")
        
        # Mostrar detalhes de falhas
        falhas = [r for r in relatorio['detalhes'] if not r['sucesso']]
        if falhas:
            print()
            print("⚠️  FALHAS DETALHADAS:")
            for falha in falhas:
                print(f"  • {falha['tipo']}: {falha['mensagem']}")

# Funções de conveniência
def validar_estrutura_dataset(df: pd.DataFrame, colunas_esperadas: List[str]) -> bool:
    """Validação rápida de estrutura do dataset."""
    validador = ValidadorDados(df)
    resultado = validador.validar_estrutura(colunas_esperadas)
    return resultado.sucesso

def verificar_valores_ausentes(df: pd.DataFrame, limite_percentual: float = 5.0) -> Dict[str, Any]:
    """Verificação rápida de valores ausentes."""
    validador = ValidadorDados(df)
    resultado = validador.verificar_valores_ausentes(limite_percentual=limite_percentual)
    return resultado.detalhes

def validar_tipos_dados(df: pd.DataFrame, mapeamento_tipos: Dict[str, str]) -> bool:
    """Validação rápida de tipos de dados."""
    validador = ValidadorDados(df)
    resultado = validador.validar_tipos(mapeamento_tipos)
    return resultado.sucesso

def verificar_consistencia_categorica(df: pd.DataFrame, coluna: str, 
                                    valores_esperados: List[str]) -> bool:
    """Verificação rápida de consistência categórica."""
    validador = ValidadorDados(df)
    resultado = validador.verificar_consistencia_categorica(coluna, valores_esperados)
    return resultado.sucesso

def gerar_relatorio_validacao(df: pd.DataFrame, 
                            regras_validacao: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """
    Gera relatório completo de validação para o dataset.
    
    Args:
        df (pd.DataFrame): DataFrame para validação
        regras_validacao (List[Dict], optional): Regras de consistência cruzada
        
    Returns:
        Dict[str, Any]: Relatório de validação
    """
    validador = ValidadorDados(df)
    
    # Executar validações básicas
    validador.verificar_valores_ausentes()
    validador.verificar_duplicatas()
    
    if regras_validacao:
        validador.validar_consistencia_cruzada(regras_validacao)
    
    return validador.gerar_relatorio_validacao()

# Exemplo de uso
if __name__ == "__main__":
    print("🔍 Módulo de validação de dados - Telco Customer Churn")
    print()
    print("💡 Exemplo de uso:")
    print("""
# Validação completa
validador = ValidadorDados(df, 'Telco Customer Churn')
validador.validar_estrutura(['customerID', 'Churn', 'tenure'])
validador.verificar_valores_ausentes()
validador.verificar_duplicatas()

# Gerar relatório
relatorio = validador.gerar_relatorio_validacao()
validador.imprimir_relatorio()
    """)