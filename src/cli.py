"""
CLI (Interface de Linha de Comando) para o projeto Telco Customer Churn.

Comandos:
  - clean     : limpeza/preparo do dataset
  - validate  : validação de qualidade do dataset (opcional: exporta JSON)
  - explore   : análise exploratória básica e (opcional) geração de figura
  - version   : mostra a versão do pacote

Autor: Marlon Henrique
Ano: 2025
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------
# Imports dos módulos do pacote (versão PT-BR)
# ---------------------------------------------
from src.limpeza_dados import pipeline_limpeza_completa
from src.validacao_dados import (
    ValidadorDados,
    gerar_relatorio_validacao as gerar_relatorio_validacao_dict,
    validar_estrutura_dataset,
    validar_tipos_dados,
    verificar_valores_ausentes,
    verificar_consistencia_categorica,
)
from src.visualizacao import (
    criar_heatmap_correlacao,
    salvar_grafico,
)
from src.utils import (
    carregar_dados,
    salvar_dados,
    gerar_resumo_dataset,
    configurar_ambiente_visualizacao,
)

# versão do pacote
try:
    from src import __version__ as PKG_VERSION
except Exception:
    PKG_VERSION = "0.1.0"

# ---------------------------------------------
# Logging
# ---------------------------------------------
logger = logging.getLogger("telco_cli")


def _configure_logging(verbosity: int) -> None:
    """
    Configura o nível de logging:
      0 -> WARNING (padrão)
      1 -> INFO
      2+ -> DEBUG
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )
    logger.debug("Logging configurado. Nível: %s", logging.getLevelName(level))


# ---------------------------------------------
# Helpers
# ---------------------------------------------
def _ensure_parent_dir(path: Path) -> None:
    """Garante que o diretório pai exista antes de salvar o arquivo."""
    dummy = path if path.suffix else (path / "dummy")
    dummy.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------
# Comandos
# ---------------------------------------------
def cmd_clean(args: argparse.Namespace) -> int:
    logger.info("🧹 Iniciando limpeza de dados...")
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_path = output_dir / (args.output_name or "telco_limpo.csv")

    if not input_path.exists():
        logger.error("Arquivo de entrada não encontrado: %s", input_path)
        return 2

    df = carregar_dados(input_path)
    df_clean = pipeline_limpeza_completa(df, verbose=(args.verbose > 0))
    _ensure_parent_dir(output_path)
    salvar_dados(df_clean, output_path)
    logger.info("✅ Dados limpos salvos em: %s", output_path)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    logger.info("🔎 Iniciando validação de dados...")
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Arquivo de dados não encontrado: %s", data_path)
        return 2

    df = carregar_dados(data_path)

    # Validações básicas (exemplos de uso dos atalhos)
    ok_schema = validar_estrutura_dataset(df, df.columns.tolist())  # aqui você pode passar uma lista esperada fixa
    ok_dtypes = validar_tipos_dados(
        df,
        {
            # mapeamento opcional (coluna -> tipo esperado)
            # 'Churn': 'category',
            # 'MonthlyCharges': 'float32',
        },
    )
    _ = verificar_valores_ausentes(df, limite_percentual=5.0)
    # exemplo de consistência categórica (opcional)
    # _ = verificar_consistencia_categorica(df, 'Churn', ['Sim', 'Não'])

    # Relatório consolidado
    validador = ValidadorDados(df, "Telco Customer Churn")
    validador.verificar_valores_ausentes()
    validador.verificar_duplicatas()
    relatorio = validador.gerar_relatorio_validacao()

    # Se o usuário passou --report, salva JSON
    if args.report:
        out = Path(args.report)
        _ensure_parent_dir(out)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(relatorio, f, ensure_ascii=False, indent=2)
        logger.info("📝 Relatório de validação salvo em: %s", out)

    logger.info("✅ Validação concluída. Schema: %s | Dtypes: %s", ok_schema, ok_dtypes)
    return 0


def cmd_explore(args: argparse.Namespace) -> int:
    logger.info("📊 Iniciando análise exploratória...")
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Arquivo de dados não encontrado: %s", data_path)
        return 2

    df = carregar_dados(data_path)
    configurar_ambiente_visualizacao()
    gerar_resumo_dataset(df, "Análise Exploratória")

    if args.report:
        # Gera heatmap de correlação e salva no caminho informado
        fig = criar_heatmap_correlacao(df)
        out = Path(args.report)
        _ensure_parent_dir(out)
        # deduz formato pelo sufixo (ex.: .png)
        ext = out.suffix.lower().lstrip(".") or "png"
        salvar_grafico(fig, out, formato=ext)
        logger.info("📈 Figura gerada em: %s", out)

    return 0


def cmd_version(_: argparse.Namespace) -> int:
    print(f"📦 Versão: {PKG_VERSION}")
    print("🚀 Projeto: Telco EDA")
    return 0


# ---------------------------------------------
# Main / Argparse
# ---------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Telco Customer Churn - Interface de Linha de Comando",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemplos:\n"
            "  telco-eda clean    --input dados/brutos/telco.csv --output dados/processados/\n"
            "  telco-eda validate --data dados/processados/telco_limpo.csv --report relatorios/validacao.json\n"
            "  telco-eda explore  --data dados/processados/telco_limpo.csv "
            "--report relatorios/figuras/correlacao.png\n"
        ),
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Aumenta a verbosidade do log (-v, -vv).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # clean
    p_clean = sub.add_parser("clean", help="Limpeza/preparo do dataset")
    p_clean.add_argument("--input", "-i", required=True, help="Arquivo de entrada (CSV/Parquet/...)")
    p_clean.add_argument("--output", "-o", required=True, help="Diretório de saída")
    p_clean.add_argument(
        "--output-name",
        "-n",
        default=None,
        help="Nome do arquivo de saída (padrão: telco_limpo.csv)",
    )
    p_clean.set_defaults(func=cmd_clean)

    # validate
    p_val = sub.add_parser("validate", help="Validação de qualidade do dataset")
    p_val.add_argument("--data", "-d", required=True, help="Arquivo de dados para validar")
    p_val.add_argument(
        "--report",
        help="Se definido, salva o relatório em JSON (ex.: relatorios/validacao.json)",
    )
    p_val.set_defaults(func=cmd_validate)

    # explore
    p_exp = sub.add_parser("explore", help="Análise exploratória")
    p_exp.add_argument("--data", "-d", required=True, help="Arquivo de dados para explorar")
    p_exp.add_argument(
        "--report",
        "-r",
        help="Se definido, salva figura (ex.: relatorios/figuras/correlacao.png)",
    )
    p_exp.set_defaults(func=cmd_explore)

    # version
    p_ver = sub.add_parser("version", help="Mostrar versão do projeto")
    p_ver.set_defaults(func=cmd_version)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)

    try:
        rc = args.func(args)
        sys.exit(rc)
    except SystemExit:
        raise
    except Exception as e:
        logger.exception("❌ Erro inesperado: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
