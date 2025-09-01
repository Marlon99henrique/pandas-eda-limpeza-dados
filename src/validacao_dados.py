"""
Módulo de validação e controle de qualidade de dados para o projeto Telco Customer Churn.
Funções para validação de estrutura, consistência e qualidade dos dados.

Autor: Marlon Henrique
Data: 2025
Versão: 1.0.1
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Logger do módulo (configure no app/notebook/CLI)
logger = logging.getLogger(__name__)

# Suprimir warnings (opcional)
warnings.filterwarnings("ignore")


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

    # ------------------------------------------------------------------ #
    # Estrutura
    # ------------------------------------------------------------------ #
    def validar_estrutura(self, colunas_esperadas: List[str], min_linhas: int = 1) -> ResultadoValidacao:
        """
        Valida a estrutura básica do dataset.

        Args:
            colunas_esperadas (List[str]): Lista de colunas esperadas
            min_linhas (int): Número mínimo de linhas esperadas

        Returns:
            ResultadoValidacao: Resultado da validação
        """
        detalhes = {
            "colunas_esperadas": list(colunas_esperadas),
            "colunas_encontradas": self.df.columns.tolist(),
            "linhas_encontradas": int(len(self.df)),
        }

        colunas_faltantes = list(set(colunas_esperadas) - set(self.df.columns))
        colunas_extra = list(set(self.df.columns) - set(colunas_esperadas))
        linhas_ok = len(self.df) >= min_linhas

        sucesso = (len(colunas_faltantes) == 0) and linhas_ok
        if sucesso:
            mensagem = "Estrutura do dataset válida"
            if colunas_extra:
                mensagem += f" (colunas extras: {sorted(colunas_extra)})"
        else:
            partes = []
            if colunas_faltantes:
                partes.append(f"Colunas faltantes: {sorted(colunas_faltantes)}")
            if not linhas_ok:
                partes.append(f"Mínimo de {min_linhas} linhas esperado, encontradas {len(self.df)}")
            if colunas_extra:
                partes.append(f"Colunas extras: {sorted(colunas_extra)}")
            mensagem = "; ".join(partes)

        resultado = ResultadoValidacao(
            sucesso=sucesso,
            mensagem=mensagem,
            detalhes=detalhes,
            tipo=TipoValidacao.ESTRUTURA,
        )
        self.resultados.append(resultado)
        logger.info("Validação de estrutura: %s", "OK" if sucesso else "FALHA")
        return resultado

    # ------------------------------------------------------------------ #
    # Tipos
    # ------------------------------------------------------------------ #
    def validar_tipos(self, mapeamento_tipos: Dict[str, str]) -> ResultadoValidacao:
        """
        Valida os tipos de dados das colunas.

        Args:
            mapeamento_tipos (Dict[str, str]): Mapeamento coluna → tipo esperado (ex.: 'category', 'float32')

        Returns:
            ResultadoValidacao: Resultado da validação
        """
        detalhes = {"tipos_incorretos": {}}
        tipos_incorretos = False

        for coluna, tipo_esperado in mapeamento_tipos.items():
            if coluna not in self.df.columns:
                continue
            tipo_atual = str(self.df[coluna].dtype)
            # checagem flexível: 'float32' ∈ 'float32', 'category' ∈ 'category'
            tipo_correto = tipo_esperado.lower() in tipo_atual.lower()
            if not tipo_correto:
                tipos_incorretos = True
                detalhes["tipos_incorretos"][coluna] = {"esperado": tipo_esperado, "encontrado": tipo_atual}

        sucesso = not tipos_incorretos
        mensagem = "Tipos de dados válidos" if sucesso else "Tipos de dados incorretos encontrados"

        resultado = ResultadoValidacao(
            sucesso=sucesso,
            mensagem=mensagem,
            detalhes=detalhes,
            tipo=TipoValidacao.TIPOS,
        )
        self.resultados.append(resultado)
        logger.info("Validação de tipos: %s", "OK" if sucesso else "FALHA")
        return resultado

    # ------------------------------------------------------------------ #
    # Ausentes
    # ------------------------------------------------------------------ #
    def verificar_valores_ausentes(
        self,
        colunas: Optional[List[str]] = None,
        limite_percentual: float = 5.0,
    ) -> ResultadoValidacao:
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

        colunas_verificar = [c for c in colunas if c in self.df.columns]

        detalhes = {"ausentes_por_coluna": {}, "total_ausentes": 0}
        tem_ausentes = False
        tem_excesso_ausentes = False

        n_rows = max(1, len(self.df))
        for coluna in colunas_verificar:
            n_ausentes = int(self.df[coluna].isnull().sum())
            pct_ausentes = (n_ausentes / n_rows) * 100
            detalhes["ausentes_por_coluna"][coluna] = {
                "quantidade": n_ausentes,
                "percentual": pct_ausentes,
                "excede_limite": pct_ausentes > limite_percentual,
            }
            detalhes["total_ausentes"] += n_ausentes
            tem_ausentes |= n_ausentes > 0
            tem_excesso_ausentes |= pct_ausentes > limite_percentual

        sucesso = not tem_ausentes
        mensagem = "Nenhum valor ausente encontrado" if sucesso else "Valores ausentes encontrados"
        if tem_excesso_ausentes:
            mensagem += f" (alguns excedem {limite_percentual}%)"

        resultado = ResultadoValidacao(
            sucesso=sucesso,
            mensagem=mensagem,
            detalhes=detalhes,
            tipo=TipoValidacao.AUSENTES,
        )
        self.resultados.append(resultado)
        logger.info("Verificação de ausentes: %s", "OK" if sucesso else "AVISO")
        return resultado

    # ------------------------------------------------------------------ #
    # Categorias
    # ------------------------------------------------------------------ #
    def verificar_consistencia_categorica(
        self,
        coluna: str,
        valores_esperados: List[str],
        ignorar_nulos: bool = True,
    ) -> ResultadoValidacao:
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
                tipo=TipoValidacao.CATEGORIAS,
            )

        serie = self.df[coluna]
        valores_atuais = serie.dropna().unique() if ignorar_nulos else serie.unique()
        valores_inesperados = list(set(valores_atuais) - set(valores_esperados))

        detalhes = {
            "coluna": coluna,
            "valores_esperados": list(valores_esperados),
            "valores_encontrados": list(valores_atuais),
            "valores_inesperados": valores_inesperados,
            "total_valores_inesperados": len(valores_inesperados),
        }

        sucesso = len(valores_inesperados) == 0
        mensagem = (
            f"Valores categóricos consistentes em '{coluna}'"
            if sucesso
            else f"Valores inesperados em '{coluna}': {valores_inesperados}"
        )

        resultado = ResultadoValidacao(
            sucesso=sucesso,
            mensagem=mensagem,
            detalhes=detalhes,
            tipo=TipoValidacao.CATEGORIAS,
        )
        self.resultados.append(resultado)
        logger.info("Consistência categórica (%s): %s", coluna, "OK" if sucesso else "FALHA")
        return resultado

    # ------------------------------------------------------------------ #
    # Faixas numéricas
    # ------------------------------------------------------------------ #
    def validar_range_numerico(
        self,
        coluna: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> ResultadoValidacao:
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
                tipo=TipoValidacao.RANGE,
            )

        if not pd.api.types.is_numeric_dtype(self.df[coluna]):
            return ResultadoValidacao(
                sucesso=False,
                mensagem=f"Coluna '{coluna}' não é numérica",
                detalhes={},
                tipo=TipoValidacao.RANGE,
            )

        valores = self.df[coluna].dropna()
        if valores.empty:
            detalhes = {
                "coluna": coluna,
                "min_esperado": min_val,
                "max_esperado": max_val,
                "min_encontrado": None,
                "max_encontrado": None,
                "valores_fora_range": 0,
                "percentual_fora_range": 0.0,
            }
            resultado_vazio = ResultadoValidacao(
                sucesso=True,
                mensagem=f"Sem valores não nulos em '{coluna}'",
                detalhes=detalhes,
                tipo=TipoValidacao.RANGE,
            )
            self.resultados.append(resultado_vazio)
            logger.info("Range numérico (%s): OK (sem valores não nulos)", coluna)
            return resultado_vazio

        fora_range = pd.Series(False, index=valores.index)
        if min_val is not None:
            fora_range |= valores < min_val
        if max_val is not None:
            fora_range |= valores > max_val

        n_fora_range = int(fora_range.sum())
        pct_fora_range = float((n_fora_range / len(valores)) * 100) if len(valores) > 0 else 0.0

        detalhes = {
            "coluna": coluna,
            "min_esperado": min_val,
            "max_esperado": max_val,
            "min_encontrado": float(valores.min()) if len(valores) else None,
            "max_encontrado": float(valores.max()) if len(valores) else None,
            "valores_fora_range": n_fora_range,
            "percentual_fora_range": pct_fora_range,
        }

        sucesso = n_fora_range == 0
        mensagem = (
            f"Valores de '{coluna}' dentro do range esperado"
            if sucesso
            else f"{n_fora_range} valores fora do range em '{coluna}'"
        )

        resultado = ResultadoValidacao(
            sucesso=sucesso,
            mensagem=mensagem,
            detalhes=detalhes,
            tipo=TipoValidacao.RANGE,
        )
        self.resultados.append(resultado)
        logger.info("Range numérico (%s): %s", coluna, "OK" if sucesso else "FALHA")
        return resultado

    # ------------------------------------------------------------------ #
    # Duplicatas
    # ------------------------------------------------------------------ #
    def verificar_duplicatas(self, subset: Optional[List[str]] = None) -> ResultadoValidacao:
        """
        Verifica linhas duplicadas no dataset.

        Args:
            subset (List[str], optional): Subconjunto de colunas para verificação

        Returns:
            ResultadoValidacao: Resultado da verificação
        """
        if subset is None:
            duplicatas = int(self.df.duplicated().sum())
            detalhes = {"tipo": "completas", "duplicatas": duplicatas}
        else:
            subset_existente = [col for col in subset if col in self.df.columns]
            duplicatas = int(self.df.duplicated(subset=subset_existente).sum())
            detalhes = {"tipo": "parciais", "duplicatas": duplicatas, "subset": subset_existente}

        sucesso = duplicatas == 0
        mensagem = "Nenhuma duplicata encontrada" if sucesso else f"{duplicatas} duplicatas encontradas"

        resultado = ResultadoValidacao(
            sucesso=sucesso,
            mensagem=mensagem,
            detalhes=detalhes,
            tipo=TipoValidacao.DUPLICATAS,
        )
        self.resultados.append(resultado)
        logger.info("Verificação de duplicatas: %s", "OK" if sucesso else "FALHA")
        return resultado

    # ------------------------------------------------------------------ #
    # Consistência cruzada
    # ------------------------------------------------------------------ #
    def validar_consistencia_cruzada(self, regras: List[Dict[str, Any]]) -> ResultadoValidacao:
        """
        Valida regras de consistência cruzada entre colunas.

        Args:
            regras (List[Dict]): Lista de regras de validação.
                Cada regra pode ter:
                - 'condicao': str (usando DataFrame.eval) ou callable(df) -> Series[bool] ou bool
                - 'mensagem': mensagem a exibir em caso de violação

        Returns:
            ResultadoValidacao: Resultado da validação
        """
        violacoes: List[Dict[str, Any]] = []
        for i, regra in enumerate(regras, start=1):
            cond = regra.get("condicao")
            msg = regra.get("mensagem", f"Violação da regra {i}")

            try:
                if isinstance(cond, str):
                    mask = self.df.eval(cond)  # Series[bool]
                else:
                    mask = cond(self.df)  # Series[bool] ou bool

                if isinstance(mask, (pd.Series, np.ndarray, list)):
                    mask = pd.Series(mask, index=self.df.index).fillna(False)
                    n_viol = int((~mask).sum())
                else:
                    # Se for bool único, False => tudo viola; True => nada viola
                    n_viol = int(len(self.df)) if (mask is False) else 0

                if n_viol > 0:
                    violacoes.append({"regra": i, "condicao": cond, "violacoes": n_viol, "mensagem": msg})

            except Exception as e:
                violacoes.append({"regra": i, "condicao": cond, "erro": str(e), "mensagem": "Erro na regra"})

        sucesso = len(violacoes) == 0
        mensagem = "Todas as regras de consistência atendidas" if sucesso else "Regras de consistência violadas"
        detalhes = {"regras_violadas": violacoes}

        resultado = ResultadoValidacao(
            sucesso=sucesso,
            mensagem=mensagem,
            detalhes=detalhes,
            tipo=TipoValidacao.CONSISTENCIA,
        )
        self.resultados.append(resultado)
        logger.info("Consistência cruzada: %s", "OK" if sucesso else "FALHA")
        return resultado

    # ------------------------------------------------------------------ #
    # Relatórios
    # ------------------------------------------------------------------ #
    def gerar_relatorio_validacao(self) -> Dict[str, Any]:
        """
        Gera relatório completo de validação.

        Returns:
            Dict[str, Any]: Relatório detalhado
        """
        total_validacoes = len(self.resultados)
        validacoes_sucesso = sum(1 for r in self.resultados if r.sucesso)
        taxa_sucesso = (validacoes_sucesso / total_validacoes * 100) if total_validacoes > 0 else 0.0

        relatorio: Dict[str, Any] = {
            "dataset": self.nome_dataset,
            "timestamp": pd.Timestamp.now().isoformat(),
            "estatisticas_gerais": {
                "total_validacoes": total_validacoes,
                "validacoes_sucesso": validacoes_sucesso,
                "validacoes_falha": total_validacoes - validacoes_sucesso,
                "taxa_sucesso": taxa_sucesso,
            },
            "resultados_por_tipo": {},
            "resumo": [],
            "detalhes": [],
        }

        for resultado in self.resultados:
            tipo = resultado.tipo.value
            if tipo not in relatorio["resultados_por_tipo"]:
                relatorio["resultados_por_tipo"][tipo] = {"total": 0, "sucesso": 0, "falha": 0}

            relatorio["resultados_por_tipo"][tipo]["total"] += 1
            if resultado.sucesso:
                relatorio["resultados_por_tipo"][tipo]["sucesso"] += 1
            else:
                relatorio["resultados_por_tipo"][tipo]["falha"] += 1

            relatorio["resumo"].append(
                {"tipo": tipo, "sucesso": resultado.sucesso, "mensagem": resultado.mensagem}
            )
            relatorio["detalhes"].append(
                {
                    "tipo": tipo,
                    "sucesso": resultado.sucesso,
                    "mensagem": resultado.mensagem,
                    "detalhes": resultado.detalhes,
                }
            )

        return relatorio

    def imprimir_relatorio(self) -> None:
        """Imprime relatório de validação formatado (saída legível no console)."""
        relatorio = self.gerar_relatorio_validacao()

        print(f"📋 RELATÓRIO DE VALIDAÇÃO - {self.nome_dataset}")
        print("=" * 60)
        print(f"📅 {relatorio['timestamp']}")
        print(f"📊 Estatísticas: {relatorio['estatisticas_gerais']['taxa_sucesso']:.1f}% de sucesso")
        print()

        # Resultados por tipo
        for tipo, stats in relatorio["resultados_por_tipo"].items():
            status = "✅" if stats["falha"] == 0 else "❌"
            print(f"{status} {tipo.upper()}: {stats['sucesso']}/{stats['total']}")

        print()
        print("📝 DETALHES:")
        for resultado in relatorio["resumo"]:
            status = "✅" if resultado["sucesso"] else "❌"
            print(f"  {status} {resultado['tipo']}: {resultado['mensagem']}")

        # Mostrar detalhes de falhas
        falhas = [r for r in relatorio["detalhes"] if not r["sucesso"]]
        if falhas:
            print()
            print("⚠️  FALHAS DETALHADAS:")
            for falha in falhas:
                print(f"  • {falha['tipo']}: {falha['mensagem']}")


# ---------------------------------------------------------------------- #
# Funções de conveniência (atalhos rápidos)
# ---------------------------------------------------------------------- #
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


def verificar_consistencia_categorica(df: pd.DataFrame, coluna: str, valores_esperados: List[str]) -> bool:
    """Verificação rápida de consistência categórica."""
    validador = ValidadorDados(df)
    resultado = validador.verificar_consistencia_categorica(coluna, valores_esperados)
    return resultado.sucesso


def gerar_relatorio_validacao(
    df: pd.DataFrame,
    regras_validacao: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Gera relatório completo de validação para o dataset.

    Args:
        df (pd.DataFrame): DataFrame para validação
        regras_validacao (List[Dict], optional): Regras de consistência cruzada

    Returns:
        Dict[str, Any]: Relatório de validação
    """
    validador = ValidadorDados(df)
    # Validações básicas
    validador.verificar_valores_ausentes()
    validador.verificar_duplicatas()
    # Regras cruzadas
    if regras_validacao:
        validador.validar_consistencia_cruzada(regras_validacao)
    return validador.gerar_relatorio_validacao()


# ---------------------------------------------------------------------- #
# Execução direta (demonstração rápida)
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s: %(message)s")
    print("🔍 Módulo de validação de dados - Telco Customer Churn\n")
    print("💡 Exemplo de uso:")
    print(
        """
# Validação completa
validador = ValidadorDados(df, 'Telco Customer Churn')
validador.validar_estrutura(['customerID', 'Churn', 'tenure'])
validador.verificar_valores_ausentes()
validador.verificar_duplicatas()

# Gerar relatório
relatorio = validador.gerar_relatorio_validacao()
validador.imprimir_relatorio()
"""
    )
