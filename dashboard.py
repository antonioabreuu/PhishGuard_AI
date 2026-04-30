"""
PhishGuard AI - Multilingual Streamlit Dashboard (EN / PT-BR)
Native Streamlit charts, auto Light/Dark theme, SHAP explainability,
session history and Markdown report export.
"""

from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd
import shap
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

from data_prep import URLFeatureEngineer
from train_model import MODEL_PATH, load_model

# ---------------------------------------------------------------------------
# Translation dictionary
# ---------------------------------------------------------------------------

LANGUAGES: Dict[str, Dict[str, Any]] = {
    "English": {
        # Page
        "page_title":    "PhishGuard AI",
        "page_header":   "🛡️ PhishGuard AI",
        "page_subtitle": (
            "Enter a URL below to classify it as **phishing** or **benign** "
            "using a Random Forest model with SHAP explainability."
        ),
        # Sidebar
        "sidebar_language_label":  "🌐 Language",
        "sidebar_about_header":    "About",
        "sidebar_about_text": (
            "PhishGuard AI uses Machine Learning to detect malicious links. "
            "It acts like a digital detective, analyzing the structure of a URL "
            "(such as its length or unusual characters) to identify phishing attempts.\n\n"
            "We use a **Random Forest** algorithm to make the prediction, and **SHAP** "
            "technology to explain exactly *why* a link was flagged as dangerous or safe, "
            "ensuring the AI's decision is completely transparent.\n\n"
            "**Stack:** Python · scikit-learn · SHAP · Streamlit"
        ),
        "tech_stack_title": "⚙️ Tech Stack Details",
        "tech_rf": "**Random Forest:** An ensemble machine learning algorithm that uses multiple decision trees to make highly accurate predictions.",
        "tech_python": "**Python:** The core programming language powering the backend logic and data processing.",
        "tech_sklearn": "**scikit-learn:** The industry-standard Python library used to train and evaluate our machine learning model.",
        "tech_shap": "**SHAP:** A game-theoretic approach to explain the output of any machine learning model, providing transparency.",
        "tech_streamlit": "**Streamlit:** The framework used to build this interactive web application entirely in Python.",
        "sidebar_history_header":  "🕘 Session History",
        "sidebar_history_empty":   "No analyses yet.",
        # Input / form
        "input_placeholder": "https://example.com/login?user=test",
        "input_label":       "Target URL",
        "button_analyse":    "🔍 Analyse",
        "button_new_analysis": "🔄 New Analysis",
        # Spinners
        "spinner_model":     "Loading PhishGuard AI model…",
        "spinner_explainer": "Initialising SHAP explainer…",
        "spinner_analysis":  "Analysing URL…",
        # Errors & warnings
        "error_model_not_found": (
            "⚠️ Trained model not found. "
            "Expected path: `{path}`. "
            "Please run `train_model.py` first, then restart the dashboard."
        ),
        "warning_empty_url":        "Please enter a URL before submitting.",
        "error_feature_extraction": "Feature extraction failed: {exc}",
        "error_inference":          "Model inference failed: {exc}",
        "warning_shap": (
            "SHAP explanation could not be generated: {exc}. "
            "This can happen when the mock model was trained on very few samples."
        ),
        # Verdict
        "verdict_phishing_title": "⚠️ PHISHING DETECTED",
        "verdict_phishing_prob":  "Phishing probability: <strong>{pct:.1f}%</strong>",
        "verdict_benign_title":   "✅ BENIGN URL",
        "verdict_benign_prob":    "Phishing probability: <strong>{pct:.1f}%</strong>",
        # Gauge
        "gauge_header": "Risk Confidence",
        "gauge_label":  "Phishing confidence: {pct:.1f}%",
        "gauge_legend": "🟢 Low (< 40%) | 🟡 Suspicious (40–70%) | 🔴 Critical (> 70%)",
        # SOC metric cards
        "metric_url_length": "URL Length",
        "metric_entropy":    "Entropy",
        "metric_subdomains": "Subdomains",
        "help_url_length":   "⚠️ Alert if > 75 characters. Phishing URLs are often long due to encoded payloads or random tokens.",
        "help_entropy":      "⚠️ Alert if > 4.5. High entropy suggests obfuscation or randomised subdomains.",
        "help_subdomains":   "⚠️ Alert if > 1. Multiple subdomains are rarely seen on legitimate services.",
        # Status labels for feature table
        "status_alert":  "🔴 Alert",
        "status_ok":     "🟢 OK",
        "status_yes":    "Yes",
        "status_no":     "No",
        # Download report
        "download_label":    "📥 Download Report (.md)",
        "download_filename": "phishguard_report.md",
        "download_mime":     "text/markdown",
        # SHAP section
        "shap_header":  "🔍 SHAP Feature Explanation",
        "shap_caption": (
            "The Impact (%) column shows how much each feature increased or decreased "
            "the AI's final phishing risk score. The Importance bar displays the absolute SHAP value "
            "(the raw mathematical weight of the feature), acting as a thermometer of relevance regardless of direction."
        ),
        "shap_col_feature":    "Feature",
        "shap_col_impact":     "Impact (%)",
        "shap_col_impact_abs": "Importance",
        "shap_feature_names": {
            "url_length":            "URL Length",
            "dot_count":             "Dot Count",
            "has_ip_host":           "IP as Host",
            "has_https":             "Protocol HTTPS",
            "suspicious_char_count": "Suspicious Chars",
            "subdomain_count":       "Subdomain Count",
            "has_risk_word":         "Risk Word Present",
            "entropy":               "Entropy",
        },
        # Feature table
        "features_header":      "📋 Extracted Features",
        "features_col_feature": "Feature",
        "features_col_value":   "Value",
        # Educational expander
        "expander_label": "📚 Understand the Metrics",
        "expander_intro": (
            "The model analyses **lexical properties** of the URL itself — "
            "no page content is fetched. Below is what each feature means "
            "from a security perspective, including numeric thresholds used "
            "as reference baselines."
        ),
        "expander_features": {
            "url_length": (
                "**URL Length** — Legitimate sites tend to use short, readable "
                "URLs. Attackers often embed encoded payloads or random tokens, "
                "inflating the length significantly. "
                "**Baseline: URLs longer than 75 characters are considered suspicious.**"
            ),
            "dot_count": (
                "**Dot Count** — Excessive dots can indicate subdomain abuse, "
                "e.g. `paypal.com.attacker.net`, where the real domain is the "
                "last one but the eye is drawn to the brand name at the start. "
                "**Baseline: More than 3 dots is a warning sign.**"
            ),
            "has_ip_host": (
                "**IP as Host** — Using a raw IP address (`http://192.168.1.1/login`) "
                "bypasses DNS-based reputation filters and brand recognition, "
                "a common phishing evasion technique. "
                "**Baseline: Any IP as host is an immediate red flag.**"
            ),
            "has_https": (
                "**HTTPS Presence** — HTTPS confirms encryption, *not* "
                "legitimacy. Many phishing pages now use valid TLS certificates "
                "to appear trustworthy, so this feature alone is weak. "
                "**Baseline: Absence of HTTPS increases risk; presence alone does not confirm safety.**"
            ),
            "suspicious_char_count": (
                "**Suspicious Characters** — Characters such as `@`, `-`, `=`, "
                "`?`, `%` are misused to obscure the real destination. "
                "**Baseline: More than 4 suspicious characters is a strong signal.**"
            ),
            "subdomain_count": (
                "**Subdomain Count** — Deep subdomain chains are rarely seen "
                "on legitimate services and are a strong phishing signal. "
                "**Baseline: More than 1 subdomain is already suspicious.**"
            ),
            "has_risk_word": (
                "**Risk Words** — Terms like *login*, *secure*, *verify*, "
                "*update*, *password* are frequently placed in phishing URLs. "
                "**Baseline: Presence of any risk word increases phishing probability.**"
            ),
            "entropy": (
                "**Shannon Entropy** — Measures randomness in the URL string. "
                "High entropy often indicates obfuscation or base64-encoded parameters. "
                "**Baseline: Values above 4.5 indicate high risk.**"
            ),
        },
        # Report template
        "report_title":          "# PhishGuard AI — Analysis Report",
        "report_date_label":     "**Date:**",
        "report_url_label":      "**URL:**",
        "report_verdict_label":  "**Verdict:**",
        "report_verdict_phish":  "🔴 PHISHING",
        "report_verdict_benign": "🟢 BENIGN",
        "report_prob_label":     "**Phishing Probability:**",
        "report_features_title": "## Extracted Features",
        "report_disclaimer": (
            "> ⚠️ This report was generated by PhishGuard AI, an educational tool. "
            "Always combine results with additional security controls."
        ),
        # Disclaimer
        "disclaimer": (
            "⚠️ **Disclaimer:** PhishGuard AI is an educational tool. "
            "Always combine its output with additional security controls. "
            "False positives and false negatives are expected."
        ),
    },

    "Português": {
        # Page
        "page_title":    "PhishGuard AI",
        "page_header":   "🛡️ PhishGuard AI",
        "page_subtitle": (
            "Insira uma URL abaixo para classificá-la como **phishing** ou **benigna** "
            "usando um modelo Random Forest com explicabilidade SHAP."
        ),
        # Sidebar
        "sidebar_language_label":  "🌐 Idioma",
        "sidebar_about_header":    "Sobre",
        "sidebar_about_text": (
            "O PhishGuard AI usa Inteligência Artificial para detectar links maliciosos. "
            "Ele funciona como um detetive digital, analisando a estrutura de um site "
            "(como o tamanho do link ou o uso de caracteres estranhos) para identificar tentativas de phishing.\n\n"
            "Usamos um modelo **Random Forest** para fazer a previsão, e a tecnologia **SHAP** "
            "para explicar exatamente *por que* um link foi classificado como perigoso ou seguro, "
            "garantindo total transparência na decisão.\n\n"
            "**Stack:** Python · scikit-learn · SHAP · Streamlit"
        ),
        "tech_stack_title": "⚙️ Detalhes da Stack",
        "tech_rf": "**Random Forest:** Um algoritmo de machine learning que usa dezenas de 'árvores de decisão' juntas para dar um veredito altamente preciso.",
        "tech_python": "**Python:** A linguagem de programação base que processa a lógica e os dados do projeto.",
        "tech_sklearn": "**scikit-learn:** A biblioteca padrão da indústria usada para treinar e avaliar a nossa Inteligência Artificial.",
        "tech_shap": "**SHAP:** A tecnologia matemática que 'abre a caixa preta' da IA, explicando o motivo exato de cada decisão.",
        "tech_streamlit": "**Streamlit:** O framework que usamos para transformar o código Python neste painel web interativo.",
        "sidebar_history_header":  "🕘 Histórico da Sessão",
        "sidebar_history_empty":   "Nenhuma análise ainda.",
        # Input / form
        "input_placeholder": "https://exemplo.com/login?usuario=teste",
        "input_label":       "URL alvo",
        "button_analyse":    "🔍 Analisar",
        "button_new_analysis": "🔄 Nova Análise",
        # Spinners
        "spinner_model":     "Carregando modelo PhishGuard AI…",
        "spinner_explainer": "Iniciando explicador SHAP…",
        "spinner_analysis":  "Analisando URL…",
        # Errors & warnings
        "error_model_not_found": (
            "⚠️ Modelo treinado não encontrado. "
            "Caminho esperado: `{path}`. "
            "Execute `train_model.py` primeiro e reinicie o dashboard."
        ),
        "warning_empty_url":        "Por favor, insira uma URL antes de enviar.",
        "error_feature_extraction": "Falha na extração de features: {exc}",
        "error_inference":          "Falha na inferência do modelo: {exc}",
        "warning_shap": (
            "Não foi possível gerar a explicação SHAP: {exc}. "
            "Isso pode ocorrer quando o modelo mock foi treinado com poucos exemplos."
        ),
        # Verdict
        "verdict_phishing_title": "⚠️ PHISHING DETECTADO",
        "verdict_phishing_prob":  "Probabilidade de phishing: <strong>{pct:.1f}%</strong>",
        "verdict_benign_title":   "✅ URL BENIGNA",
        "verdict_benign_prob":    "Probabilidade de phishing: <strong>{pct:.1f}%</strong>",
        # Gauge
        "gauge_header": "Nível de Risco",
        "gauge_label":  "Grau de Certeza do Modelo: {pct:.1f}%",
        "gauge_legend": "🟢 Baixo (< 40%) | 🟡 Suspeito (40–70%) | 🔴 Crítico (> 70%)",
        # SOC metric cards
        "metric_url_length": "Tamanho da URL",
        "metric_entropy":    "Entropia",
        "metric_subdomains": "Subdomínios",
        "help_url_length":   "⚠️ Alerta se > 75 caracteres. URLs de phishing costumam ser longas por conterem payloads codificados ou tokens aleatórios.",
        "help_entropy":      "⚠️ Alerta se > 4.5. Alta entropia indica ofuscação ou subdomínios aleatorizados.",
        "help_subdomains":   "⚠️ Alerta se > 1. Múltiplos subdomínios raramente aparecem em serviços legítimos.",
        # Status labels for feature table
        "status_alert": "🔴 Alerta",
        "status_ok":    "🟢 OK",
        "status_yes":   "Sim",
        "status_no":    "Não",
        # Download report
        "download_label":    "📥 Baixar Relatório (.md)",
        "download_filename": "relatorio_phishguard.md",
        "download_mime":     "text/markdown",
        # SHAP section
        "shap_header":  "🔍 Explicação SHAP por Feature",
        "shap_caption": (
            "A coluna Impacto (%) mostra o quanto cada feature aumentou ou diminuiu "
            "o risco final calculado pela IA. A barra de Importância exibe o valor SHAP absoluto "
            "(o peso matemático bruto da variável), servindo como um termômetro de relevância independente da direção."
        ),
        "shap_col_feature":    "Feature",
        "shap_col_impact":     "Impacto (%)",
        "shap_col_impact_abs": "Importância",
        "shap_feature_names": {
            "url_length":            "Tamanho da URL",
            "dot_count":             "Qtd. de Pontos",
            "has_ip_host":           "IP como Host",
            "has_https":             "Protocolo HTTPS",
            "suspicious_char_count": "Chars Suspeitos",
            "subdomain_count":       "Qtd. Subdomínios",
            "has_risk_word":         "Termos Sensíveis",
            "entropy":               "Entropia",
        },
        # Feature table
        "features_header":      "📋 Features Extraídas",
        "features_col_feature": "Feature",
        "features_col_value":   "Valor",
        # Educational expander
        "expander_label": "📚 Entenda as Métricas",
        "expander_intro": (
            "O modelo analisa **propriedades lexicais** da própria URL — "
            "nenhum conteúdo de página é acessado. Veja abaixo o que cada "
            "feature significa do ponto de vista de segurança, incluindo os "
            "limites numéricos usados como referência."
        ),
        "expander_features": {
            "url_length": (
                "**Tamanho da URL** — Sites legítimos tendem a usar URLs curtas "
                "e legíveis. Atacantes frequentemente embutem payloads codificados "
                "ou tokens aleatórios, inflando o comprimento da URL. "
                "**Baseline: URLs com mais de 75 caracteres são consideradas suspeitas.**"
            ),
            "dot_count": (
                "**Quantidade de Pontos** — Pontos excessivos podem indicar abuso "
                "de subdomínio. "
                "**Baseline: Mais de 3 pontos acende um alerta.**"
            ),
            "has_ip_host": (
                "**IP como Host** — Usar um endereço IP bruto burla filtros de "
                "reputação baseados em DNS. "
                "**Baseline: Qualquer IP como host é um sinal imediato de risco.**"
            ),
            "has_https": (
                "**Presença de HTTPS** — HTTPS confirma criptografia, *não* "
                "legitimidade. Muitas páginas de phishing já usam certificados TLS válidos. "
                "**Baseline: Ausência de HTTPS aumenta o risco; presença isolada não garante segurança.**"
            ),
            "suspicious_char_count": (
                "**Caracteres Suspeitos** — Caracteres como `@`, `-`, `=`, `?`, `%` "
                "são usados para ocultar o destino real. "
                "**Baseline: Mais de 4 caracteres suspeitos é um forte indicador.**"
            ),
            "subdomain_count": (
                "**Número de Subdomínios** — Cadeias profundas de subdomínios "
                "raramente aparecem em serviços legítimos. "
                "**Baseline: Mais de 1 subdomínio já é suspeito.**"
            ),
            "has_risk_word": (
                "**Palavras de Risco** — Termos como *login*, *seguro*, *verificar* "
                "são frequentemente inseridos em URLs de phishing. "
                "**Baseline: A presença de qualquer palavra de risco aumenta a probabilidade de phishing.**"
            ),
            "entropy": (
                "**Entropia de Shannon** — Mede a aleatoriedade da string da URL. "
                "**Baseline: Valores acima de 4.5 indicam risco alto.**"
            ),
        },
        # Report template
        "report_title":          "# PhishGuard AI — Relatório de Análise",
        "report_date_label":     "**Data:**",
        "report_url_label":      "**URL:**",
        "report_verdict_label":  "**Veredito:**",
        "report_verdict_phish":  "🔴 PHISHING",
        "report_verdict_benign": "🟢 BENIGNA",
        "report_prob_label":     "**Probabilidade de Phishing:**",
        "report_features_title": "## Features Extraídas",
        "report_disclaimer": (
            "> ⚠️ Este relatório foi gerado pelo PhishGuard AI, uma ferramenta educacional. "
            "Combine sempre os resultados com outros controles de segurança."
        ),
        # Disclaimer
        "disclaimer": (
            "⚠️ **Aviso:** O PhishGuard AI é uma ferramenta educacional. "
            "Combine sempre seus resultados com outros controles de segurança. "
            "Falsos positivos e falsos negativos são esperados."
        ),
    },
}

# ---------------------------------------------------------------------------
# Page configuration  (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state["history"] = []

# ---------------------------------------------------------------------------
# Custom CSS — no hardcoded text colours; inherits from active theme
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
        .phishing-alert {
            background-color: #ff4b4b22;
            border-left: 5px solid #ff4b4b;
            padding: 1rem 1.5rem;
            border-radius: 6px;
            margin-bottom: 1rem;
        }
        .benign-ok {
            background-color: #21c35422;
            border-left: 5px solid #21c354;
            padding: 1rem 1.5rem;
            border-radius: 6px;
            margin-bottom: 1rem;
        }
        .section-header {
            font-size: 1.1rem;
            font-weight: 600;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — language selector (rendered before any translated string is used)
# ---------------------------------------------------------------------------

with st.sidebar:
    selected_language: str = st.radio(
        label="🌐 Language / Idioma",
        options=list(LANGUAGES.keys()),
        index=0,
        horizontal=False,
    )

T: Dict[str, Any] = LANGUAGES[selected_language]

with st.sidebar:
    st.divider()
    st.header(T["sidebar_about_header"])
    st.markdown(T["sidebar_about_text"])

    # Nova seção de Detalhes da Stack na barra lateral
    st.sidebar.divider()
    with st.sidebar.expander(T["tech_stack_title"]):
        st.markdown(T["tech_rf"])
        st.markdown(T["tech_python"])
        st.markdown(T["tech_sklearn"])
        st.markdown(T["tech_shap"])
        st.markdown(T["tech_streamlit"])

    st.divider()
    st.header(T["sidebar_history_header"])
    history: List[Dict[str, Any]] = st.session_state["history"]
    if not history:
        st.caption(T["sidebar_history_empty"])
    else:
        for entry in reversed(history):
            emoji = "🔴" if entry["is_phishing"] else "🟢"
            st.markdown(f"{emoji} `{entry['url'][:45]}{'…' if len(entry['url']) > 45 else ''}`")

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_model() -> RandomForestClassifier | None:
    """
    Load the trained Random Forest model from disk (session-cached).

    Returns:
        Fitted RandomForestClassifier, or None if the file is missing.
    """
    try:
        return load_model(MODEL_PATH)
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def get_explainer(
    _classifier: RandomForestClassifier,
) -> shap.TreeExplainer:
    """
    Build a SHAP TreeExplainer for the classifier (session-cached).

    The leading underscore prevents Streamlit from hashing the sklearn
    object as a cache key.

    Args:
        _classifier: Fitted RandomForestClassifier instance.

    Returns:
        Configured shap.TreeExplainer.
    """
    return shap.TreeExplainer(_classifier)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def extract_features(url: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Run URLFeatureEngineer on a single URL.

    Args:
        url: Raw URL string supplied by the user.

    Returns:
        Tuple of (feature_dict, single-row feature DataFrame).
    """
    engineer = URLFeatureEngineer()
    feature_dict: Dict[str, Any] = engineer.extract_features(url)
    feature_df: pd.DataFrame = pd.DataFrame(
        [feature_dict], columns=URLFeatureEngineer.FEATURE_COLUMNS
    )
    return feature_dict, feature_df


def _get_status(value: Any, alert_condition: bool) -> str:
    """
    Return a localised status label based on an alert condition.

    Uses the global `selected_language` variable (not T, which may not yet
    be bound at call time) to guarantee correct PT-BR labels.

    Args:
        value:           The raw feature value (unused directly; kept for
                         potential future use in label composition).
        alert_condition: True when the value exceeds the risk threshold.

    Returns:
        Localised status string, e.g. '🔴 Alert' or '🟢 OK'.
    """
    return T["status_alert"] if alert_condition else T["status_ok"]


def _fmt_pct(v: float) -> str:
    """Format a SHAP value as a signed percentage string."""
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.1f}%"


def generate_report(
    url: str,
    is_phishing: bool,
    probability: float,
    feature_dict: Dict[str, Any],
) -> str:
    """
    Generates a rich Markdown report with baselines and status.

    Args:
        url:          The analysed URL.
        is_phishing:  True if the model classified the URL as phishing.
        probability:  Estimated phishing probability [0, 1].
        feature_dict: Mapping of feature names to their extracted values.

    Returns:
        A multi-line Markdown string ready for download.
    """
    # Detect if language is Portuguese
    lang_is_pt = "Português" in selected_language

    def _get_status_and_baseline(feature: str, value: Any) -> str:
        """Return status and baseline for a feature based on thresholds."""
        rules = {
            "url_length": {"is_alert": value > 75, "baseline": "< 75"},
            "dot_count": {"is_alert": value > 3, "baseline": "< 4"},
            "has_ip_host": {"is_alert": value == 1, "baseline": "0"},
            "has_https": {"is_alert": value == 0, "baseline": "1"},
            "suspicious_char_count": {"is_alert": value > 4, "baseline": "< 5"},
            "subdomain_count": {"is_alert": value > 1, "baseline": "< 2"},
            "has_risk_word": {"is_alert": value == 1, "baseline": "0"},
            "entropy": {"is_alert": value > 4.5, "baseline": "< 4.5"},
        }
        rule = rules.get(feature, {"is_alert": False, "baseline": "-"})
        if rule["is_alert"]:
            return (
                f"🔴 Alerta (Ideal: {rule['baseline']})"
                if lang_is_pt
                else f"🔴 Alert (Ideal: {rule['baseline']})"
            )
        return (
            f"🟢 Normal (Ideal: {rule['baseline']})"
            if lang_is_pt
            else f"🟢 Normal (Ideal: {rule['baseline']})"
        )

    def _format_value(value: Any) -> str:
        """Format a value as string with appropriate decimal places."""
        if isinstance(value, float):
            return str(int(value)) if value.is_integer() else f"{value:.3f}"
        return str(value)

    # Get verdict and timestamp
    verdict_str = T["report_verdict_phish"] if is_phishing else T["report_verdict_benign"]
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build feature table rows with status and baseline
    md_lines = [
        T["report_title"],
        "",
        f"{T['report_date_label']} {current_time}",
        f"{T['report_url_label']} `{url}`",
        f"{T['report_verdict_label']} {verdict_str}",
        f"{T['report_prob_label']} {probability * 100:.1f}%",
        "",
        T["report_features_title"],
        "",
        f"| {T['features_col_feature']} | {T['features_col_value']} | Status |",
        "|---|---|---|",
    ]

    # Add each feature with translated name, value, and status
    for feat, val in feature_dict.items():
        fmt_val = _format_value(val)
        status = _get_status_and_baseline(feat, val)
        feat_name_translated = T["shap_feature_names"].get(feat, feat)
        md_lines.append(f"| {feat_name_translated} | {fmt_val} | {status} |")

    md_lines.append("")
    md_lines.append(T["report_disclaimer"])

    return "\n".join(md_lines)


def render_verdict(is_phishing: bool, probability: float) -> None:
    """
    Render a colour-coded verdict banner with HTML-safe probability text.

    Args:
        is_phishing: True when the model predicts phishing.
        probability: Estimated phishing probability [0, 1].
    """
    pct = probability * 100
    if is_phishing:
        prob_html = T["verdict_phishing_prob"].format(pct=pct)
        st.markdown(
            f"""
            <div class="phishing-alert">
                <h3 style="color:#ff4b4b; margin:0">{T["verdict_phishing_title"]}</h3>
                <p style="margin:0.4rem 0 0 0; font-size:1rem">{prob_html}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        prob_html = T["verdict_benign_prob"].format(pct=pct)
        st.markdown(
            f"""
            <div class="benign-ok">
                <h3 style="color:#21c354; margin:0">{T["verdict_benign_title"]}</h3>
                <p style="margin:0.4rem 0 0 0; font-size:1rem">{prob_html}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_probability_gauge(probability: float) -> None:
    """
    Render a progress bar with a risk-level legend below it.

    Args:
        probability: Estimated phishing probability [0, 1].
    """
    st.markdown(
        f'<p class="section-header">{T["gauge_header"]}</p>',
        unsafe_allow_html=True,
    )
    st.progress(
        value=probability,
        text=T["gauge_label"].format(pct=probability * 100),
    )
    st.caption(T["gauge_legend"])


def render_soc_metrics(feature_dict: Dict[str, Any]) -> None:
    """
    Render three SOC-focused KPI cards with contextual threshold tooltips.

    Args:
        feature_dict: Mapping of feature name to its computed value.
    """
    m1, m2, m3 = st.columns(3)
    m1.metric(
        label=T["metric_url_length"],
        value=int(feature_dict["url_length"]),
        help=T["help_url_length"],
    )
    m2.metric(
        label=T["metric_entropy"],
        value=f"{feature_dict['entropy']:.3f}",
        help=T["help_entropy"],
    )
    m3.metric(
        label=T["metric_subdomains"],
        value=int(feature_dict["subdomain_count"]),
        help=T["help_subdomains"],
    )


def render_download_button(
    url: str,
    is_phishing: bool,
    probability: float,
    feature_dict: Dict[str, Any],
) -> None:
    """
    Render a download button that exports the analysis as a Markdown report.

    Args:
        url:          The analysed URL.
        is_phishing:  Classification result.
        probability:  Phishing probability [0, 1].
        feature_dict: Extracted feature values.
    """
    report_md = generate_report(url, is_phishing, probability, feature_dict)
    st.download_button(
        label=T["download_label"],
        data=report_md,
        file_name=T["download_filename"],
        mime=T["download_mime"],
    )


def render_shap_chart(explainer: shap.TreeExplainer, feature_df: pd.DataFrame, is_phishing: bool) -> None:

    st.markdown(f'<p class="section-header">{T["shap_header"]}</p>', unsafe_allow_html=True)

    st.caption(T["shap_caption"])



    # Detecção de idioma

    lang_is_pt = T.get("features_col_value", "") == "Valor"

    

    col_meaning = "Efeito na Decisão" if lang_is_pt else "Effect on Decision"

    lbl_phishing = "🔴 Aumenta o Risco" if lang_is_pt else "🔴 Increases Risk"

    lbl_benign = "🟢 Reduz o Risco" if lang_is_pt else "🟢 Decreases Risk"

    lbl_neutral = "⚪ Neutro" if lang_is_pt else "⚪ Neutral"



    # --- LÓGICA DINÂMICA DE NOMES ---

    # Cria uma cópia do dicionário original para podermos alterar os nomes livremente

    dynamic_names = dict(T["shap_feature_names"])

    

    # Extrai os valores reais da URL que está sendo analisada

    val_https = feature_df["has_https"].iloc[0]

    val_risk = feature_df["has_risk_word"].iloc[0]

    val_ip = feature_df["has_ip_host"].iloc[0]

    

    # Altera o nome da feature baseado no valor e no idioma

    if lang_is_pt:

        dynamic_names["has_https"] = "Presença de HTTPS" if val_https == 1 else "Ausência de HTTPS"

        dynamic_names["has_risk_word"] = "Termos Sensíveis Encontrados" if val_risk == 1 else "Ausência de Termos Sensíveis"

        dynamic_names["has_ip_host"] = "Acesso via IP" if val_ip == 1 else "Acesso via Domínio Padrão"

    else:

        dynamic_names["has_https"] = "Presence of HTTPS" if val_https == 1 else "Absence of HTTPS"

        dynamic_names["has_risk_word"] = "Sensitive Terms Found" if val_risk == 1 else "No Sensitive Terms"

        dynamic_names["has_ip_host"] = "Access via IP" if val_ip == 1 else "Standard Domain Access"

    # --------------------------------



    # Aplica os nomes dinâmicos antes de passar para o explicador do SHAP

    translated_df = feature_df.rename(columns=dynamic_names)



    shap_obj = explainer(translated_df)

    # Force a explicação a ser sempre sobre o RISCO (Classe 1)
    class_idx = 1  # Foco total no risco de phishing

    explanation = shap_obj[0, :, class_idx]



    col_feature    = T["shap_col_feature"]

    col_impact     = T["shap_col_impact"]

    col_impact_abs = T["shap_col_impact_abs"]



    raw_values = explanation.values.tolist()



    def _fmt_pct(v: float) -> str:

        sign = "+" if v >= 0 else ""

        return f"{sign}{v * 100:.1f}%"



    def _get_meaning(v: float) -> str:

        # Se o valor for positivo, ele empurra para Phishing (Aumenta Risco)
        if v > 0.001:

            return lbl_phishing

        # Se o valor for negativo, ele puxa para Benigno (Reduz Risco)
        elif v < -0.001:

            return lbl_benign

        return lbl_neutral



    shap_df = (

        pd.DataFrame({

            col_feature:    list(explanation.feature_names),

            col_impact:     [_fmt_pct(v) for v in raw_values],

            col_meaning:    [_get_meaning(v) for v in raw_values],

            col_impact_abs: [abs(v) for v in raw_values],

        })

        .sort_values(col_impact_abs, ascending=False)

        .reset_index(drop=True)

    )



    max_abs = float(shap_df[col_impact_abs].max()) or 1.0



    st.dataframe(

        shap_df,

        use_container_width=True,

        hide_index=True,

        column_config={

            col_feature: st.column_config.TextColumn(col_feature),

            col_impact:  st.column_config.TextColumn(col_impact),

            col_meaning: st.column_config.TextColumn(col_meaning),

            col_impact_abs: st.column_config.ProgressColumn(

                col_impact_abs,

                min_value=0.0,

                max_value=max_abs,

                format="", 

            ),

        },

    )


def render_feature_table(feature_dict: Dict[str, Any]) -> None:
    # Correção: Detecta o idioma verificando se a coluna Valor está traduzida
    lang_is_pt = T.get("features_col_value", "") == "Valor"
    
    col_status = "Status"
    lbl_normal = "🟢 Normal"
    lbl_alert = "🔴 Alerta" if lang_is_pt else "🔴 Alert"
    lbl_target = "Esperado" if lang_is_pt else "Expected"

    def _format_value(value: Any) -> str:
        if isinstance(value, float):
            return str(int(value)) if value.is_integer() else f"{value:.3f}"
        return str(value)

    def _get_status(feature: str, value: Any) -> str:
        rules = {
            "url_length": {"is_alert": value > 75, "baseline": "< 75"},
            "dot_count": {"is_alert": value > 3, "baseline": "< 4"},
            "has_ip_host": {"is_alert": value == 1, "baseline": "0"},
            "has_https": {"is_alert": value == 0, "baseline": "1"},
            "suspicious_char_count": {"is_alert": value > 4, "baseline": "< 5"},
            "subdomain_count": {"is_alert": value > 1, "baseline": "< 2"},
            "has_risk_word": {"is_alert": value == 1, "baseline": "0"},
            "entropy": {"is_alert": value > 4.5, "baseline": "< 4.5"},
        }
        rule = rules.get(feature, {"is_alert": False, "baseline": "-"})
        
        status_text = lbl_alert if rule["is_alert"] else lbl_normal
        return f"{status_text} ({lbl_target}: {rule['baseline']})"

    st.markdown(f'<p class="section-header">{T["features_header"]}</p>', unsafe_allow_html=True)
    
    table_data = []
    for feat, val in feature_dict.items():
        fmt_val = _format_value(val)
        status = _get_status(feat, val)
        feat_name_translated = T["shap_feature_names"].get(feat, feat)
        table_data.append([feat_name_translated, fmt_val, status])

    display_df = pd.DataFrame(
        table_data, 
        columns=[T["features_col_feature"], T["features_col_value"], col_status]
    )

    st.table(display_df.set_index(T["features_col_feature"]))


def render_educational_expander() -> None:
    """
    Render a collapsible expander with security explanations and numeric
    baselines for each feature.
    """
    with st.expander(T["expander_label"], expanded=False):
        st.markdown(T["expander_intro"])
        st.divider()
        for explanation in T["expander_features"].values():
            st.markdown(explanation)
            st.write("")


def _append_history(url: str, is_phishing: bool) -> None:
    """
    Append a completed analysis to the session history (max 5 entries).

    Older entries beyond the limit are discarded so the sidebar stays concise.

    Args:
        url:         The analysed URL.
        is_phishing: Classification result.
    """
    st.session_state["history"].append({"url": url, "is_phishing": is_phishing})
    if len(st.session_state["history"]) > 5:
        st.session_state["history"] = st.session_state["history"][-5:]


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Render the PhishGuard AI Streamlit dashboard.

    Features:
        - Enter-key form submission via st.form.
        - Auto Light/Dark theme (no hardcoded colours).
        - Multilingual UI (EN / PT-BR).
        - SOC KPI metric cards with threshold tooltips.
        - Human-readable SHAP impact table with signed percentages.
        - Static feature table via st.table (no clickable cells).
        - Session history in the sidebar (last 5 analyses).
        - Markdown report export via st.download_button.
        - Educational metrics expander with numeric baselines.
    """
    st.title(T["page_header"])
    st.markdown(T["page_subtitle"])
    st.divider()

    # --- Model loading ---
    with st.spinner(T["spinner_model"]):
        classifier = get_model()

    if classifier is None:
        st.error(
            T["error_model_not_found"].format(path=MODEL_PATH.resolve()),
            icon="🚫",
        )
        st.stop()

    with st.spinner(T["spinner_explainer"]):
        explainer = get_explainer(classifier)

    # --- URL input form (Enter-key enabled) ---
    with st.form(key="analyze_form", border=False):
        url_input: str = st.text_input(
            label=T["input_label"],
            placeholder=T["input_placeholder"],
            # Mantém a URL na caixa de texto mesmo após recarregar
            value=st.session_state.get("analyzed_url", "") 
        )
        analyse_clicked: bool = st.form_submit_button(
            T["button_analyse"],
            type="primary",
            use_container_width=False,
        )

    # Se o botão for clicado, salva a URL na "memória" da sessão
    if analyse_clicked:
        if not url_input or url_input.strip() == "":
            st.warning(T["warning_empty_url"])
            st.session_state.analyzed_url = None
            st.stop()
        st.session_state.analyzed_url = url_input.strip()

    # --- Analysis ---
    # Só executa a análise se houver uma URL guardada na memória
    if st.session_state.get("analyzed_url"):
        url = st.session_state["analyzed_url"]

        with st.spinner(T["spinner_analysis"]):
            try:
                feature_dict, feature_df = extract_features(url)
            except Exception as exc:
                st.error(T["error_feature_extraction"].format(exc=exc))
                st.stop()

            try:
                predicted_class: int = int(classifier.predict(feature_df)[0])
                phishing_proba: float = float(
                    classifier.predict_proba(feature_df)[0][1]
                )
            except Exception as exc:
                st.error(T["error_inference"].format(exc=exc))
                st.stop()

        is_phishing = bool(predicted_class)

        # --- CORREÇÃO DO HISTÓRICO ---
        # Inicializa o histórico se não existir
        if "history" not in st.session_state:
            st.session_state["history"] = []
            
        # Só adiciona se o histórico estiver vazio OU se a última URL for diferente da atual
        if not st.session_state["history"] or st.session_state["history"][-1]["url"] != url:
            st.session_state["history"].append({"url": url, "is_phishing": is_phishing})
            # Mantém apenas as últimas 5 consultas para não poluir a tela
            if len(st.session_state["history"]) > 5:
                st.session_state["history"].pop(0)

        st.divider()

        # SOC KPI cards — full width
        render_soc_metrics(feature_dict)
        st.write("")

        # Download button — immediately below KPI cards
        render_download_button(url, is_phishing, phishing_proba, feature_dict)

        st.write("")

        # Verdict + gauge  |  Feature table
        left_col, right_col = st.columns([2, 3])
        with left_col:
            render_verdict(is_phishing, phishing_proba)
            render_probability_gauge(phishing_proba)
        with right_col:
            render_feature_table(feature_dict)

        st.divider()

        # SHAP impact table
        try:
            render_shap_chart(explainer, feature_df, is_phishing)
        except Exception as exc:
            st.warning(T["warning_shap"].format(exc=exc))

        st.divider()

        # Educational expander
        render_educational_expander()

        # Botão para resetar a sessão atual rapidamente (agora bilíngue)
        if st.button(T["button_new_analysis"], type="secondary"):
            st.session_state["analyzed_url"] = None
            st.rerun()

        st.divider()
        st.caption(T["disclaimer"])


if __name__ == "__main__":
    main()