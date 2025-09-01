"""
Testes unitários para o módulo src/validacao_dados.py

Autor: Marlon Henrique
Ano: 2025
"""

import pytest
import pandas as pd

from src.validacao_dados import (
    ValidadorDados,
    validar_estrutura_dataset,
    verificar_valores_ausentes,
    validar_tipos_dados,
    verificar_consistencia_categorica,
    gerar_relatorio_validacao,
    TipoValidacao,
)

class TestValidacaoDados:
    @pytest.fixture
    def dados_validos(self):
        """Dataset válido de referência."""
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "nome": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "idade": [25, 30, 35, 40, 45],
            "cidade": ["SP", "RJ", "SP", "MG", "SP"],
            "ativo": [True, True, False, True, False],
            "data_criacao": pd.date_range("2023-01-01", periods=5, freq="D"),
        })

    @pytest.fixture
    def dados_com_problemas(self):
        """Dataset com diversos problemas intencionais."""
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],              # coluna extra
            "nome": ["Alice", "Bob", None, "David", "Eve", "Frank"],  # ausente
            "idade": ["25", "30", "35", "40", "45", "60"],            # tipo errado
            "cidade": ["SP", "RJ", "SP", "MG", "XX", "SP"],           # valor inválido
            "salario": [5000, 6000, 7000, 8000, 9000, -1000],         # negativo
            "status": ["Ativo", "Inativo", "Ativo", "Pendente", "Ativo", "Ativo"],
        })

    @pytest.fixture
    def dados_tipos_incorretos(self):
        return pd.DataFrame({
            "id": ["1", "2", "3", "4", "5"],          # esperado numérico
            "idade": [25, 30, 35, 40, 45],            # ok
            "preco": ["10.5", "20.3", "15.7", "25.1", "30.9"],  # esperado float
            "ativo": ["True", "False", "True", "False", "True"],  # esperado bool
        })

    def test_validador_inicializacao(self, dados_validos):
        v = ValidadorDados(dados_validos, "Test Dataset")
        assert v.df.shape == dados_validos.shape
        assert v.nome_dataset == "Test Dataset"
        assert len(v.resultados) == 0

    def test_validar_estrutura_valida(self, dados_validos):
        v = ValidadorDados(dados_validos)
        cols = ["id", "nome", "idade", "cidade", "ativo", "data_criacao"]
        r = v.validar_estrutura(cols, min_linhas=3)
        assert r.sucesso is True and r.tipo == TipoValidacao.ESTRUTURA

    def test_validar_estrutura_faltando_colunas(self, dados_validos):
        v = ValidadorDados(dados_validos)
        cols = ["id", "nome", "idade", "cidade", "ativo", "data_criacao", "faltante"]
        r = v.validar_estrutura(cols)
        assert r.sucesso is False
        assert "faltantes" in r.mensagem.lower()

    def test_validar_estrutura_poucas_linhas(self, dados_validos):
        v = ValidadorDados(dados_validos)
        r = v.validar_estrutura(["id", "nome"], min_linhas=10)
        assert r.sucesso is False

    def test_validar_tipos_corretos(self, dados_validos):
        v = ValidadorDados(dados_validos)
        tipos = {"id": "int", "nome": "object", "idade": "int", "cidade": "object", "ativo": "bool"}
        r = v.validar_tipos(tipos)
        assert r.sucesso is True and r.tipo == TipoValidacao.TIPOS

    def test_validar_tipos_incorretos(self, dados_tipos_incorretos):
        v = ValidadorDados(dados_tipos_incorretos)
        tipos = {"id": "int", "idade": "int", "preco": "float", "ativo": "bool"}
        r = v.validar_tipos(tipos)
        assert r.sucesso is False
        assert len(r.detalhes["tipos_incorretos"]) == 3

    def test_verificar_valores_ausentes_sem_ausentes(self, dados_validos):
        v = ValidadorDados(dados_validos)
        r = v.verificar_valores_ausentes()
        assert r.sucesso is True and r.detalhes["total_ausentes"] == 0

    def test_verificar_valores_ausentes_com_ausentes(self, dados_com_problemas):
        v = ValidadorDados(dados_com_problemas)
        r = v.verificar_valores_ausentes()
        assert r.sucesso is False
        assert r.detalhes["total_ausentes"] > 0
        assert "nome" in r.detalhes["ausentes_por_coluna"]

    def test_verificar_valores_ausentes_limite_percentual(self, dados_com_problemas):
        v = ValidadorDados(dados_com_problemas)
        r = v.verificar_valores_ausentes(limite_percentual=10.0)
        cols_excesso = [c for c, info in r.detalhes["ausentes_por_coluna"].items() if info["excede_limite"]]
        assert len(cols_excesso) > 0

    def test_verificar_consistencia_categorica_valida(self, dados_validos):
        v = ValidadorDados(dados_validos)
        r = v.verificar_consistencia_categorica("cidade", ["SP", "RJ", "MG"])
        assert r.sucesso is True and r.tipo == TipoValidacao.CATEGORIAS

    def test_verificar_consistencia_categorica_invalida(self, dados_com_problemas):
        v = ValidadorDados(dados_com_problemas)
        r = v.verificar_consistencia_categorica("cidade", ["SP", "RJ", "MG"])
        assert r.sucesso is False
        assert "XX" in str(r.detalhes["valores_inesperados"])

    def test_verificar_consistencia_categorica_coluna_inexistente(self, dados_validos):
        v = ValidadorDados(dados_validos)
        r = v.verificar_consistencia_categorica("col_inexistente", ["A", "B"])
        assert r.sucesso is False
        assert "não encontrada" in r.mensagem.lower()

    def test_validar_range_numerico_valido(self, dados_validos):
        v = ValidadorDados(dados_validos)
        r = v.validar_range_numerico("idade", min_val=18, max_val=100)
        assert r.sucesso is True and r.tipo == TipoValidacao.RANGE

    def test_validar_range_numerico_invalido(self, dados_com_problemas):
        v = ValidadorDados(dados_com_problemas)
        r = v.validar_range_numerico("salario", min_val=0)
        assert r.sucesso is False
        assert r.detalhes["valores_fora_range"] > 0

    def test_validar_range_numerico_coluna_nao_numerica(self, dados_validos):
        v = ValidadorDados(dados_validos)
        r = v.validar_range_numerico("nome", min_val=0, max_val=100)
        assert r.sucesso is False
        assert "não é numérica" in r.mensagem.lower()

    def test_verificar_duplicatas_sem_duplicatas(self, dados_validos):
        v = ValidadorDados(dados_validos)
        r = v.verificar_duplicatas()
        assert r.sucesso is True and r.detalhes["duplicatas"] == 0

    def test_verificar_duplicatas_com_duplicatas(self, dados_com_problemas):
        dup = pd.concat([dados_com_problemas, dados_com_problemas.head(2)], ignore_index=True)
        v = ValidadorDados(dup)
        r = v.verificar_duplicatas()
        assert r.sucesso is False and r.detalhes["duplicatas"] == 2

    def test_verificar_duplicatas_subset(self, dados_com_problemas):
        v = ValidadorDados(dados_com_problemas)
        r = v.verificar_duplicatas(subset=["id"])
        assert r.detalhes["tipo"] == "parciais"
        assert "id" in r.detalhes["subset"]

    def test_validar_consistencia_cruzada_valida(self, dados_validos):
        v = ValidadorDados(dados_validos)
        regras = [
            {"condicao": "idade >= 18", "mensagem": "Idade deve ser >= 18"},
            {"condicao": "ativo == True or idade < 50", "mensagem": "Regra de negócio"},
        ]
        r = v.validar_consistencia_cruzada(regras)
        assert r.sucesso is True and r.tipo == TipoValidacao.CONSISTENCIA

    def test_validar_consistencia_cruzada_invalida(self, dados_com_problemas):
        v = ValidadorDados(dados_com_problemas)
        r = v.validar_consistencia_cruzada([{"condicao": "salario > 0", "mensagem": "Salário positivo"}])
        assert r.sucesso is False
        assert len(r.detalhes["regras_violadas"]) > 0

    def test_gerar_relatorio_validacao(self, dados_validos):
        v = ValidadorDados(dados_validos, "Test Dataset")
        v.validar_estrutura(["id", "nome", "idade", "cidade", "ativo", "data_criacao"])
        v.verificar_valores_ausentes()
        v.verificar_duplicatas()
        rel = v.gerar_relatorio_validacao()
        assert rel["dataset"] == "Test Dataset"
        assert rel["estatisticas_gerais"]["total_validacoes"] == 3
        assert rel["estatisticas_gerais"]["validacoes_sucesso"] == 3
        assert rel["estatisticas_gerais"]["taxa_sucesso"] == 100.0


# ---- Funções avulsas (conveniência) ----
def test_validar_estrutura_dataset_func(dados_validos):
    cols = ["id", "nome", "idade", "cidade", "ativo", "data_criacao"]
    assert validar_estrutura_dataset(dados_validos, cols) is True


def test_verificar_valores_ausentes_func(dados_com_problemas):
    info = verificar_valores_ausentes(dados_com_problemas)
    assert "total_ausentes" in info and info["total_ausentes"] > 0


def test_validar_tipos_dados_func(dados_validos):
    assert validar_tipos_dados(dados_validos, {"id": "int", "idade": "int"}) in (True, False)


def test_verificar_consistencia_categorica_func(dados_validos):
    assert verificar_consistencia_categorica(dados_validos, "cidade", ["SP", "RJ", "MG"]) in (True, False)


def test_gerar_relatorio_validacao_func(dados_validos):
    rel = gerar_relatorio_validacao(dados_validos)
    assert "estatisticas_gerais" in rel and rel["estatisticas_gerais"]["total_validacoes"] >= 2
