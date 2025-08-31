# 📊 Telco EDA — Análise Exploratória do Churn de Clientes

Este repositório contém uma análise exploratória completa do dataset **Telco Customer Churn**, disponível no [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).  
O objetivo é entender os padrões que levam clientes de uma empresa de telecomunicações a cancelar (churn) ou permanecer no serviço.

---

## 🎯 Objetivos
- Realizar uma **análise exploratória de dados (EDA)** completa.  
- Diagnosticar e tratar **valores ausentes e inconsistências**.  
- Explorar variáveis categóricas e numéricas.  
- Gerar **visualizações e insights** sobre o comportamento dos clientes.  
- Construir um **pipeline profissional de EDA** como etapa inicial de um projeto de ciência de dados.

---

## 🗂️ Estrutura do Repositório

```bash
telco-eda/
├── ambiente/
├── config/
├── docs/
├── notebooks/
│   └── eda_telco.ipynb
├── src/
├── testes/
├── data/
│   ├── raw/          # dados originais (NÃO versionar)
│   └── processed/    # dados tratados/derivados (NÃO versionar)
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── Makefile          # (opcional, facilita comandos)
└── setup.py

````


---

## 🛠️ Tecnologias Utilizadas
- **Python 3.10+**
- **Pandas** → manipulação de dados  
- **NumPy** → cálculos numéricos  
- **Matplotlib / Seaborn** → visualização de dados  
- **Jupyter Notebook** → execução e documentação da análise  

---

## 📊 Etapas da Análise
1. **Carregamento e diagnóstico inicial do dataset**  
2. **Tratamento de dados ausentes e inconsistentes**  
3. **Análise descritiva das variáveis**  
4. **Engenharia de features**  
5. **Visualização dos principais padrões**  
6. **Geração de insights finais sobre churn**

---

## 🔍 Principais Insights
- Clientes com **contrato mensal** possuem maior probabilidade de churn.  
- O uso de **fatura eletrônica (paperless billing)** está associado a maior cancelamento.  
- Clientes com **maior tempo de permanência (tenure)** tendem a permanecer.  
- Serviços adicionais (como **streaming e segurança online**) impactam positivamente na retenção.  

---

## 🚀 Como Executar
1. Clone este repositório:  
```bash
git clone https://github.com/Marlon99henrique/telco-eda.git
```
2. Acesse a pasta do projeto:
  ```bash
cd telco-eda
  ```
3.Instale as dependências:
```bash
pip install -r requirements.txt
```
4.pip install -r requirements.txt
```bash
jupyter notebook notebooks/eda_telco.ipynb
```
---
## 📚 Dataset
- **Nome:** Telco Customer Churn  
- **Fonte:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- **Registros:** 7.043 clientes  
- **Variáveis:** 21 colunas (demográficas, serviços contratados, billing, churn)  


---
## 👨‍💻 Autor
Projeto desenvolvido por **Marlon Henrique**  

- 🔗 [Portfólio](https://marlon99henrique.github.io/)  
- 💼 [LinkedIn](https://www.linkedin.com/in/seu-perfil)  
- 🐙 [GitHub](https://github.com/Marlon99henrique)  

