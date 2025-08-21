# 📊 Domínio do Pandas: Análise e Limpeza do Telco Customer Churn

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Fazendo-brightgreen)

## 🎯 Objetivo do Projeto

Projeto demonstra **domínio avançado da biblioteca Pandas** para limpeza e preparação de dados, utilizando o **dataset Telco Customer Churn** como estudo de caso. Através de técnicas profissionais, transformo dados brutos com problemas comuns (valores ausentes, inconsistências, tipos incorretos) em um conjunto limpo e pronto para análise.



---
## 🔍 Sobre o Dataset

**Telco Customer Churn** - Dataset público do Kaggle que simula desafios reais de qualidade de dados:

- **Fonte**: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Registros**: 7,043 clientes
- **Variáveis**: 21 características demográficas e de serviços
- **Problemas**: Valores ausentes, tipos incorretos, inconsistências categóricas

---
## 🛠️ Habilidades Demonstradas

### Técnicas com Pandas
- ✅ Diagnóstico completo de qualidade de dados
- ✅ Estratégias avançadas para missing values
- ✅ Correção de tipos de dados inadequados
- ✅ Normalização de valores categóricos
- ✅ Criação de features derivadas
- ✅ Análise de correlações e padrões

### Fluxo de Trabalho Profissional
```bash
# Abordagem profissional de limpeza de dados
df.pipe(diagnosticar_problemas)
  .pipe(corrigir_tipos_dados)
  .pipe(tratar_valores_ausentes)
  .pipe(normalizar_categoricas)
  .pipe(criar_novas_variaveis)
  .pipe(validar_qualidade_dados)

```
---
## 📊 Resultados da Limpeza

### 🔴 Antes
- 11 colunas com tipos incorretos  
- 3 colunas com valores ausentes  
- 5 colunas com inconsistências categóricas  
- Dados numéricos armazenados como *strings*  

### 🟢 Depois
- ✅ Todos os tipos de dados corrigidos, resultando em **redução de 60% no uso de memória**
- ✅ 100% dos valores ausentes tratados estrategicamente
- ✅ Inconsistências categóricas normalizadas
- ✅ Dataset robusto e pronto para análise e modelagem  

---

## 📖 Notebooks do Projeto

Os notebooks a seguir conduzem a análise completa e detalhada, com uma progressão lógica que guia o leitor por todo o fluxo de trabalho de ciência de dados.

### 🎯 [1. Análise e Limpeza de Dados](notebooks/01_analise_telco.ipynb)
Este é o notebook principal, com foco em **inspeção e limpeza de dados**. Ele demonstra o fluxo completo de trabalho, incluindo:
- **Diagnóstico de Problemas**: Identificação dos desafios nos dados brutos.
- **Limpeza Passo a Passo**: Aplicação das técnicas de tratamento de valores ausentes e inconsistências.
- **Visualização Inicial**: Gráficos que mostram o "antes" e "depois" da limpeza.

### 📈 [2. Exploração Aprofundada e Insights](notebooks/02_analise_exploratoria.ipynb)
Este notebook se aprofunda na **análise exploratória** para extrair insights e padrões. O foco está em:
- **Análise Univariada e Bivariada**: Explorando as relações entre as variáveis.
- **Visualizações Avançadas**: Gráficos detalhados para revelar insights sobre o `churn`.
- **Storytelling com Dados**: Transformando a análise em uma narrativa clara.

### ✅ [3. Validação Final e Preparação para Modelagem](notebooks/03_validacao_resultados.ipynb)
Esta etapa garante a **qualidade do dataset processado** e o prepara para a etapa de modelagem. O notebook inclui:
- **Testes de Qualidade**: Verificação final da integridade dos dados.
- **Checklist de Validação**: Garantindo que todos os requisitos foram cumpridos.
- **Documentação Técnica**: Resumo dos processos aplicados e das decisões tomadas.

---
#### 🔄 Fluxo Recomendado de Leitura:
1. **Comece pelo `01_analise_telco.ipynb`** para entender todo o processo de limpeza
2. **Explore o `02_analise_exploratoria.ipynb`** para insights profundos
3. **Finalize com `03_validacao_resultados.ipynb`** para verificação da qualidade

#### 💡 Dica para Recrutadores:
Cada notebook demonstra uma habilidade específica:
- **Notebook 1**: Habilidades técnicas com Pandas e limpeza de dados
- **Notebook 2**: Capacidade analítica e de visualização
- **Notebook 3**: Rigor metodológico e atenção à qualidade

[![Abrir Notebook Principal](https://img.shields.io/badge/📓-Abrir_Notebook_Principal-orange)](notebooks/01_analise_telco.ipynb)

---
## 🚀 Como Executar

Siga estes passos simples para rodar o projeto em seu ambiente local.

### 1. Preparação do Ambiente

```bash
# Clone o repositório
git clone https://github.com/Marlon99henrique/pandas-eda-limpeza-dados.git # Lembre-se de usar a URL real

# Navegue até a pasta do projeto
cd pandas-eda-limpeza-dados

# (Recomendado) Crie e ative um ambiente virtual
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instale todas as dependências
pip install -r ambiente/requirements.txt
````
### 2. Rodar os Notebooks

```bash
# Abrir o notebook principal diretamente
jupyter notebook notebooks/01_analise_telco.ipynb

```

---
## 📁 Estrutura do Projeto

```bash
pandas-eda-limpeza-dados/
│
├── dados/
│   ├── brutos/                   # 📂 Dados originais (não versionados)
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── processados/              # 📂 Dados limpos e processados
│   │   ├── telco_limpo.csv
│   │   ├── telco_limpo.pkl
│   │   └── telco_analise_pronta.feather
│   └── externos/                 # 📂 Dados de fontes externas
│       └── README.md
│
├── notebooks/                    # 📓 Jupyter Notebooks
│   ├── 01_analise_telco.ipynb    # 🎯 Notebook principal (análise completa)
│   ├── 02_analise_exploratoria.ipynb
│   └── 03_validacao_resultados.ipynb
│
├── src/                          # 🐍 Código fonte Python
│   ├── __init__.py
│   ├── limpeza_dados.py          # 🧹 Funções de limpeza e transformação
│   ├── validacao_dados.py        # ✅ Validação e qualidade de dados
│   ├── visualizacao.py           # 📊 Funções de visualização
│   └── utils.py                  # ⚙️ Utilitários e helpers
│
├── testes/                       # 🧪 Testes unitários
│   ├── __init__.py
│   ├── test_limpeza_dados.py     # ✅ Testes das funções de limpeza
│   ├── test_validacao_dados.py   # ✅ Testes de validação
│   ├── test_utils.py             # ✅ Testes dos utilitários
│   └──test_visualizacao.py       # ✅ Testes de visualização
|
├── docs/                         # 📚 Documentação
│   ├── metodologia.md            # 📋 Metodologia aplicada
│   ├── resultados_analise.md     # 📊 Resultados da análise
│   └── guia_uso.md               # 🚀 Guia de uso do projeto
│
├── relatorios/                   # 📈 Relatórios e visualizações
│   ├── figuras/                  # 🖼️ Gráficos e visualizações
│   │   ├── distribuicoes_antes.png
│   │   ├── distribuicoes_depois.png
│   │   ├── correlacoes.png
│   │   └── valores_ausentes.png
│   └── relatorio_final.pdf       # 📄 Relatório completo em PDF
│
├── ambiente/                     # 🐍 Configuração do ambiente
│   ├── requirements.txt          # 📋 Dependências do projeto
│   └── environment.yml           # ⚙️ Ambiente Conda (opcional)
│
├── config/                       # ⚙️ Arquivos de configuração
│   └── parametros.yaml           # 📁 Parâmetros e configurações
│
├── .gitignore                    # 🙈 Arquivos ignorados pelo Git
├── README.md                     # 📖 Este arquivo
├── LICENSE                       # ⚖️ Licença MIT
└── setup.py                      # 🐍 Script de instalação

```
### 🗂️ Legenda da Estrutura

- 📂 **Dados**: Armazenamento organizado de dados brutos e processados  
- 📓 **Notebooks**: Análises interativas e explorações  
- 🐍 **Source**: Código modularizado e reutilizável  
- 🧪 **Testes**: Garantia de qualidade do código  
- 📚 **Docs**: Documentação completa do projeto  
- 📈 **Relatórios**: Resultados e visualizações exportadas  
- 🐍 **Ambiente**: Configuração do ambiente de desenvolvimento  
- ⚙️ **Config**: Parâmetros e configurações centralizadas  

### 🎯 Principais Arquivos

-  1.`notebooks/01_analise_telco.ipynb` - Análise completa passo a passo  
-  2.`src/limpeza_dados.py` - Funções principais de limpeza  
-  3.`tests/test_limpeza_dados.py` - Testes das funcionalidades  
-  4.`requirements.txt` - Dependências do projeto

---
## 📈 Próximos Projetos da Série  
- ✅ Análise e Limpeza de Dados (este projeto)  
- 🔜 Análise de Séries Temporais com Pandas  
- 🔜 Feature Engineering para Machine Learning  

---
## 🤝 Contato
- 📩 E-mail: marlon.99henrique@gmail.com
- 🔗 [LinkedIn](https://www.linkedin.com/in/marlon-henrique-abdon-silva-8704a8217/)
- 🐙 [GitHub](https://github.com/Marlon99henrique)
- 🌐 [Meu Site](https://marlon99henrique.github.io/)
   
---
Desenvolvido com ❤️ por Marlon Henrique  
*Cientista de Dados | Análise de Dados*
