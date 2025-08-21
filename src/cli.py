"""
🎯 Interface de linha de comando (CLI) para o projeto Telco Customer Churn

Fornece comandos convenientes para executar as funcionalidades do projeto
diretamente do terminal.

Autor: Marlon Henrique
Data: 2025
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Adicionar o diretório src ao path
sys.path.append(str(Path(__file__).parent.parent))

def main():
    """Função principal da CLI"""
    parser = argparse.ArgumentParser(
        description="Telco Customer Churn Analysis - CLI Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  telco-analysis clean --input dados/brutos/telco.csv --output dados/processados/
  telco-analysis validate --data dados/processados/telco_limpo.csv
  telco-analysis explore --data dados/processados/telco_limpo.csv --report
        """
    )
    
    # Subcomandos
    subparsers = parser.add_subparsers(dest="command", help="Comando a ser executado")
    
    # Subcomando: clean
    parser_clean = subparsers.add_parser("clean", help="Limpeza de dados")
    parser_clean.add_argument("--input", "-i", required=True, help="Arquivo de entrada")
    parser_clean.add_argument("--output", "-o", required=True, help="Diretório de saída")
    parser_clean.add_argument("--config", "-c", help="Arquivo de configuração")
    
    # Subcomando: validate
    parser_validate = subparsers.add_parser("validate", help="Validação de dados")
    parser_validate.add_argument("--data", "-d", required=True, help="Arquivo de dados")
    parser_validate.add_argument("--config", "-c", help="Arquivo de configuração")
    
    # Subcomando: explore
    parser_explore = subparsers.add_parser("explore", help="Análise exploratória")
    parser_explore.add_argument("--data", "-d", required=True, help="Arquivo de dados")
    parser_explore.add_argument("--report", "-r", action="store_true", help="Gerar relatório")
    
    # Subcomando: version
    subparsers.add_parser("version", help="Mostrar versão do projeto")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "clean":
            from src.limpeza_dados import pipeline_limpeza_completa
            from src.utils import carregar_dados, salvar_dados
            
            print("🧹 Iniciando limpeza de dados...")
            df = carregar_dados(args.input)
            df_limpo = pipeline_limpeza_completa(df, verbose=True)
            
            output_path = Path(args.output) / "telco_limpo.csv"
            salvar_dados(df_limpo, output_path)
            print(f"✅ Dados limpos salvos em: {output_path}")
            
        elif args.command == "validate":
            from src.validacao_dados import ValidadorDados
            from src.utils import carregar_dados
            
            print("✅ Iniciando validação de dados...")
            df = carregar_dados(args.data)
            validador = ValidadorDados(df, "Dataset Validado")
            validador.verificar_valores_ausentes()
            validador.verificar_duplicatas()
            validador.imprimir_relatorio()
            
        elif args.command == "explore":
            from src.utils import carregar_dados, gerar_resumo_dataset
            from src.visualizacao import criar_heatmap_correlacao, salvar_grafico
            
            print("📊 Iniciando análise exploratória...")
            df = carregar_dados(args.data)
            gerar_resumo_dataset(df, "Análise Exploratória")
            
            if args.report:
                fig = criar_heatmap_correlacao(df)
                salvar_grafico(fig, "relatorios/figuras/correlacao.png")
                print("📈 Relatório gerado em: relatorios/figuras/correlacao.png")
                
        elif args.command == "version":
            from config.parametros import carregar_configuracao
            config = carregar_configuracao()
            print(f"📦 Versão: {config['geral']['versao']}")
            print(f"🚀 Projeto: {config['geral']['nome_projeto']}")
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()