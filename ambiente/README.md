# 🐍 Configuração do Ambiente - Telco Customer Churn

Este diretório contém os arquivos de configuração para reproduzir o ambiente de desenvolvimento do projeto.  

## 📋 Opção 1: Usando pip (Recomendado para iniciantes)  

### Instalação:
```bash
# Navegue até a pasta do projeto
cd pandas-eda-limpeza-dados

# Instale as dependências
pip install -r ambiente/requirements.txt
```

### Verificação:  
```bash
# Verifique se todas as dependências foram instaladas
pip list
```
---
## ⚙️ Opção 2: Usando Conda (Recomendado para ciência de dados)  
### Instalação:  
```bash
# Navegue até a pasta do projeto
cd pandas-eda-limpeza-dados

# Crie o ambiente conda
conda env create -f ambiente/environment.yml

# Ative o ambiente
conda activate telco-churn-env
```

### Verificação:
```bash
# Verifique se o ambiente foi criado
conda env list

# Verifique as dependências instaladas
conda list
```
---
## 🚀 Como Usar o Ambiente  
### Para desenvolvimento:

```bash
# Ative o ambiente conda (se usando conda)
conda activate telco-churn-env

# Ou use o pip normalmente

# Execute o Jupyter Notebook
jupyter notebook

# Ou execute o Jupyter Lab
jupyter lab

```
### Para testes:
``` bash
# Execute todos os testes
python -m pytest testes/ -v

# Execute testes com cobertura
python -m pytest testes/ --cov=src --cov-report=html

```
----
## 📦 Gerenciamento de Dependências  
### Para atualizar requirements.txt:  

```bash
pip freeze > ambiente/requirements.txt
```
---
## 🔧 Solução de Problemas
### Erro de versão de pacote:

```bash
# Se houver conflito de versões, tente:
pip install --upgrade nome_do_pacote

# Ou instale uma versão específica:
pip install nome_do_pacote==versão

```
### Problemas com conda:

```bash
# Recrie o ambiente se necessário
conda env remove -n telco-churn-env
conda env create -f ambiente/environment.yml
```
---
## 📊 Versões Testadas
- ✅ Python 3.9.16
- ✅ Windows 10/11
- ✅ Ubuntu 20.04 LTS
- ✅ macOS Monterey

---
## 💡 Dicas
-  1 .Sempre use ambiente virtual para evitar conflitos entre projetos
-  2 .Mantenha as dependências atualizadas mas com versões específicas
-  3 .Documente problemas encontrados durante a instalação
-  4 .Use o mesmo ambiente em desenvolvimento e produção

---
## 🆘 Suporte
Se encontrar problemas na instalação:
-  1 .Verifique se sua versão do Python é compatível (3.8+)
-  2 .Consulte a documentação das bibliotecas
-  3 .Verifique issues no GitHub do projeto
-  4 .Consulte fóruns como Stack Overflow

