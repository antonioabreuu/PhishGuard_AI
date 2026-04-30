# 🛡️ PhishGuard AI

### Visão Geral
Detector de links maliciosos que utiliza Machine Learning (Random Forest) e explicabilidade (SHAP) para identificar tentativas de phishing em URLs. O projeto foca em transparência, demonstrando exatamente quais características da URL influenciaram a decisão da IA.

### 🚀 Como Rodar
1. Clone o repositório.
2. Instale as dependências: `pip install -r requirements.txt`
3. Execute o dashboard: `streamlit run dashboard.py`

### 🏗️ Arquitetura
O sistema processa a URL através de um pipeline de extração de features lexicais, seguido pela classificação do modelo treinado e geração de métricas de impacto via SHAP.

### 📊 Métricas e Dados
- **Dataset:** Baseado em dados do PhishTank e URLs benignas documentadas.
- **Métricas:** O modelo busca maximizar o *Recall* para reduzir falsos negativos em ambiente de segurança.

### ⚠️ Aviso Ético
Este é um laboratório educacional. Não utilize como única camada de defesa e nunca realize ataques contra terceiros.