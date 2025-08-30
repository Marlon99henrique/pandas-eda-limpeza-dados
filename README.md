# 📊 EDA Completo: Telco Customer Churn  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-orange)  
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5%2B-red)  
![Seaborn](https://img.shields.io/badge/Seaborn-0.11%2B-lightblue)  
![License](https://img.shields.io/badge/License-MIT-green)  
![Status](https://img.shields.io/badge/Status-Concluído-brightgreen)  

---

## 🎯 Objetivo do Projeto  

Este é um projeto **profissional de Análise Exploratória de Dados (EDA)** aplicado ao dataset público **Telco Customer Churn**.  
O propósito é demonstrar:  

- **Domínio de bibliotecas Python** para análise e visualização  
- **Boas práticas de limpeza e preparação de dados**  
- **Storytelling com dados**, transformando números em insights claros  
- Estrutura modular, **simulando um fluxo real de ciência de dados**  

---

## 🔍 O Dataset  

- **Fonte**: [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- **Registros**: 7.043 clientes  
- **Variáveis**: 21 (demográficas, contratuais e de serviços)  
- **Desafios encontrados**:  
  - Valores ausentes  
  - Tipos incorretos (strings numéricas, datas mal formatadas)  
  - Inconsistências categóricas  

---

## 🛠️ Tecnologias e Habilidades  

### 🔧 Técnicas aplicadas
- Diagnóstico de qualidade dos dados  
- Tratamento de *missing values*  
- Normalização de variáveis categóricas  
- Criação de variáveis derivadas  
- Correção de tipos e inconsistências  

### 📊 EDA completo  
- Análises univariadas e bivariadas  
- Visualizações com **Matplotlib** e **Seaborn**  
- Heatmaps de correlação  
- Storytelling com insights sobre o churn  

### ✅ Validação final  
- Checklist de qualidade dos dados  
- Garantia de consistência pós-limpeza  
- Preparação para uso em **modelagem preditiva futura**  

---

## 📖 Estrutura dos Notebooks  

1. **[01 — Análise e Limpeza de Dados](notebooks/01_analise_telco.ipynb)**  
   - Diagnóstico inicial  
   - Limpeza e ajustes de dados  
   - Visualizações preliminares  

2. **[02 — Exploração Aprofundada e Insights](notebooks/02_analise_exploratoria.ipynb)**  
   - Análises estatísticas e visuais  
   - Padrões e correlações relevantes para churn  

3. **[03 — Validação Final](notebooks/03_validacao_resultados.ipynb)**  
   - Testes de qualidade  
   - Dataset pronto para modelagem  

---

## 📈 Resultados Relevantes  

- 🔍 Identificação de variáveis críticas associadas ao churn  
- 📉 Redução de **60% no uso de memória** com correções de tipos  
- ✅ Dataset **100% consistente** após tratamento de ausentes e inconsistências  
- 🚀 Base sólida para Machine Learning (próximos passos do projeto)  

---

## 🚀 Como Reproduzir  

Clone este repositório e configure o ambiente:  

```bash
# Clone o repositório
git clone https://github.com/Marlon99henrique/pandas-eda-limpeza-dados.git

# Acesse a pasta
cd pandas-eda-limpeza-dados

# Crie o ambiente virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instale as dependências
pip install -r ambiente/requirements.txt
```

Abra o notebook principal:
``` bash
jupyter notebook notebooks/01_analise_telco.ipynb
```
----
## 📂 Estrutura do Projeto
```bash
pandas-eda-limpeza-dados/
│
├── dados/             # Conjuntos de dados (brutos, processados, externos)
├── notebooks/         # Jupyter Notebooks (EDA completo)
├── src/               # Código Python modularizado (limpeza, visualização, utils)
├── testes/            # Testes unitários
├── docs/              # Documentação adicional
├── relatorios/        # Relatórios e visualizações finais
├── ambiente/          # Configuração de ambiente
├── config/            # Parâmetros de configuração
├── README.md          # Documentação principal

```
----

## 📌 Próximos Passos  
- 🔜 Feature Engineering para Machine Learning  
- 🔜 Modelagem preditiva para previsão de churn  
- 🔜 Automação de pipeline de EDA
  
----

## 🤝 Contato  
📩 E-mail: marlon.99henrique@gmail.com  
🔗 LinkedIn  
🐙 GitHub  
🌐 Portfólio Pessoal  

---
#### 📌 Desenvolvido com ❤️ por **Marlon Henrique**
Cientista de Dados | EDA | Visualização de Dados | Storytelling com Dados

