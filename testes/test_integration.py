"""
Teste de integração do pipeline Telco EDA:
- Gera um CSV sintético no esquema Telco
- Executa o pipeline de limpeza
- Roda validações básicas
- Gera e salva uma figura (heatmap de correlação)

Autor: Marlon Henrique
Ano: 2025
"""

from pathlib import Path
import pandas as pd
import numpy as np

from src.utils import carregar_dados, salvar_dados
from src.limpeza_dados import pipeline_limpeza_completa
from src.validacao_dados import ValidadorDados
from src.visualizacao import criar_heatmap_correlacao, salvar_grafico


def _criar_dataset_sintetico(caminho: Path) -> None:
    """Cria um CSV mínimo com colunas típicas do Telco."""
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        "customerID": [f"ID{i:05d}" for i in range(n)],
        "SeniorCitizen": np.random.randint(0, 2, size=n),
        "Partner": np.random.choice(["Yes", "No"], size=n),
        "Dependents": np.random.choice(["Yes", "No"], size=n),
        "tenure": np.random.randint(0, 72, size=n),
        "PhoneService": np.random.choice(["Yes", "No"], size=n),
        "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], size=n),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], size=n),
        "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], size=n),
        "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], size=n),
        "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], size=n),
        "TechSupport": np.random.choice(["Yes", "No", "No internet service"], size=n),
        "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], size=n),
        "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], size=n),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n),
        "PaperlessBilling": np.random.choice(["Yes", "No"], size=n),
        "PaymentMethod": np.random.choice(
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n
        ),
        "MonthlyCharges": np.round(np.random.uniform(20, 120, size=n), 2),
        # Intencionalmente introduzimos alguns NaNs para testar a limpeza
        "TotalCharges": pd.Series(np.round(np.random.uniform(20, 8000, size=n), 2)).mask(
            np.random.rand(n) < 0.05
        ),
        "Churn": np.random.choice(["Yes", "No"], size=n),
    })

    # Introduz um NaN extra em categórica para testar tratamento
    df.loc[0, "Partner"] = np.nan

    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=False)


def test_pipeline_integracao(tmp_path):
    # 1) Dataset sintético
    bruto = tmp_path / "dados" / "telco_bruto.csv"
    _criar_dataset_sintetico(bruto)

    # 2) Carregar e limpar
    df_raw = carregar_dados(bruto)
    assert not df_raw.empty

    df_clean = pipeline_limpeza_completa(df_raw, verbose=False)
    assert not df_clean.empty
    # Deve ter removido/ajustado NaNs relevantes
    assert df_clean.isnull().sum().sum() == 0

    # Features criadas no seu pipeline
    for col in ["TenureGroup", "TotalServicos", "TipoCliente", "CustoPorServico"]:
        assert col in df_clean.columns

    # 3) Validar
    val = ValidadorDados(df_clean, "Telco (limpo)")
    r1 = val.verificar_valores_ausentes()
    r2 = val.verificar_duplicatas()
    # Após limpeza, o ideal é zero ausentes
    assert r1.sucesso is True
    # Não exigimos zero duplicatas no sintético, mas normalmente é True
    assert isinstance(r2.sucesso, bool)

    relatorio = val.gerar_relatorio_validacao()
    assert "estatisticas_gerais" in relatorio
    assert relatorio["estatisticas_gerais"]["total_validacoes"] >= 2

    # 4) Visualização (heatmap) + salvar
    fig = criar_heatmap_correlacao(df_clean)
    saida_fig = tmp_path / "relatorios" / "figuras" / "correlacao.png"
    salvar_grafico(fig, saida_fig, formato="png")
    assert saida_fig.exists()

    # 5) Persistir dataset limpo
    processado = tmp_path / "dados" / "telco_limpo.csv"
    salvar_dados(df_clean, processado)
    assert processado.exists()
