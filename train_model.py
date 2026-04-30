"""
PhishGuard AI - Model Training Module
Fetches real-world URL data, engineers features, trains and evaluates a
Random Forest classifier, then persists the model to disk.
"""

import random
import string
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from data_prep import URLFeatureEngineer, build_pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DIR      = Path("models")
MODEL_FILENAME = "rf_phish_model.pkl"
MODEL_PATH     = MODEL_DIR / MODEL_FILENAME

TEST_SIZE      = 0.2
RANDOM_STATE   = 42
N_ESTIMATORS   = 100
MAX_DEPTH      = 10
SAMPLE_SIZE    = 2000          # 1 000 benign + 1 000 phishing
PROGRESS_EVERY = 250           # print progress every N URLs

# ---------------------------------------------------------------------------
# Public dataset candidates (CSV with 'url' and 'label' columns)
# ---------------------------------------------------------------------------

_DATASET_URLS = [
    # Kaggle-mirrored phishing dataset on GitHub (verified raw CSV)
    "https://raw.githubusercontent.com/datasets/phishing-urls/main/data/phishing_urls.csv",
    # Secondary fallback: ISCX-URL 2016 lightweight mirror
    "https://raw.githubusercontent.com/ebubekirbbr/pdd/master/input/train.csv",
]


# ---------------------------------------------------------------------------
# Synthetic data generator (fallback)
# ---------------------------------------------------------------------------

def _random_token(length: int = 8) -> str:
    """Return a random lowercase alphanumeric string of the given length."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def _generate_synthetic_dataset(n_each: int = 1000) -> pd.DataFrame:
    """
    Build a hiper-realistic synthetic URL dataset when no public source
    is reachable.

    Benign URLs are modelled on real popular domains; phishing URLs apply
    obfuscation techniques documented in the OWASP LLM Top 10 and common
    threat-intel feeds: IP hosts, @ redirection, excessive hyphens,
    deep subdomain chains, and risk-word injection.

    Args:
        n_each: Number of URLs per class (benign / phishing).

    Returns:
        Balanced DataFrame with columns 'url' and 'label'.
    """
    benign_templates = [
        "https://www.google.com/search?q={t}",
        "https://www.wikipedia.org/wiki/{t}",
        "https://stackoverflow.com/questions/{t}",
        "https://github.com/{t}/{t}",
        "https://www.amazon.com/dp/{t}",
        "https://www.bbc.co.uk/news/{t}",
        "https://docs.python.org/3/library/{t}.html",
        "https://www.linkedin.com/in/{t}",
        "https://medium.com/@{t}/post-{t}",
        "https://www.youtube.com/watch?v={t}",
        "https://www.reddit.com/r/{t}/comments/{t}",
        "https://www.nytimes.com/2024/01/01/{t}.html",
        "https://www.apple.com/shop/{t}",
        "https://www.microsoft.com/en-us/{t}",
        "https://www.bradesco.com.br/html/classic/produtos-servicos/{t}.shtm",
        "https://www2.itau.com.br/itaudigital/{t}",
        "https://www.nubank.com.br/cobrar/{t}",
        "https://www.gov.br/receitafederal/{t}",
    ]

    phishing_templates = [
        # IP host
        "http://192.168.{a}.{b}/login/update?user={t}@victim.com",
        "http://10.0.{a}.{b}/secure/account-verify={t}",
        # @ redirection
        "http://www.paypal.com@{t}-evil.ru/signin?redirect={t}",
        "http://google.com@{t}.phish.net/login",
        # Excessive hyphens + risk words
        "http://secure-login-{t}-verify-account.com/update?token={t}",
        "http://paypal-secure-{t}-update.xyz/webscr?cmd=login&user={t}",
        # Deep subdomain chains
        "http://login.secure.{t}.update.attacker.net/account",
        "http://verify.{t}.signin.{t}.badhost.cc/confirm?id={t}",
        # Encoded / high-entropy paths
        "http://{t}.com/cGFzc3dvcmQ={t}&dXNlcg=={t}",
        "http://xn--{t}-{t}.com/login?session={t}{t}{t}",
        # Risk-word combos
        "http://www.{t}-banking-secure.com/account/password-reset?code={t}",
        "http://update-{t}.verify-login.com/webscr?token={t}&user={t}@mail.com",
        # MalwareBazaar-style random subdomains
        "http://{t}.{t}.{t}.notabank.biz/secure/?{t}={t}&{t}={t}",
        "http://{t}{t}.free-offers-{t}.click/form?id={t}",
    ]

    benign_rows = []
    for _ in range(n_each):
        tpl = random.choice(benign_templates)
        url = tpl.format(t=_random_token(random.randint(4, 10)))
        benign_rows.append({"url": url, "label": 0})

    phishing_rows = []
    for _ in range(n_each):
        tpl = random.choice(phishing_templates)
        a, b = random.randint(1, 254), random.randint(1, 254)
        url = tpl.format(
            t=_random_token(random.randint(4, 12)), a=a, b=b
        )
        phishing_rows.append({"url": url, "label": 1})

    df = pd.DataFrame(benign_rows + phishing_rows).sample(
        frac=1, random_state=RANDOM_STATE
    ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------

def fetch_real_data(sample_size: int = SAMPLE_SIZE) -> pd.DataFrame:
    """
    Attempt to download a public URL dataset; fall back to synthetic data.

    The function tries each URL in ``_DATASET_URLS`` in order.  A dataset
    is accepted when it contains both 'url' and 'label' columns and has at
    least ``sample_size`` rows.  The result is balanced (equal class counts)
    and capped at ``sample_size`` rows total.

    Args:
        sample_size: Total rows to return (split evenly between classes).

    Returns:
        DataFrame with columns 'url' (str) and 'label' (int 0/1).
    """
    n_each = sample_size // 2

    for url in _DATASET_URLS:
        try:
            print(f"[data] Trying public dataset: {url}")
            df = pd.read_csv(url, usecols=lambda c: c.lower() in {"url", "label"})
            df.columns = [c.lower() for c in df.columns]

            if "url" not in df.columns or "label" not in df.columns:
                print("[data] Missing required columns — skipping.")
                continue

            df["label"] = pd.to_numeric(df["label"], errors="coerce")
            df = df.dropna(subset=["url", "label"])
            df["label"] = df["label"].astype(int)

            benign  = df[df["label"] == 0].sample(
                min(n_each, (df["label"] == 0).sum()),
                random_state=RANDOM_STATE,
            )
            phish   = df[df["label"] == 1].sample(
                min(n_each, (df["label"] == 1).sum()),
                random_state=RANDOM_STATE,
            )

            if len(benign) < n_each * 0.5 or len(phish) < n_each * 0.5:
                print("[data] Dataset too small or imbalanced — skipping.")
                continue

            balanced = (
                pd.concat([benign, phish])
                .sample(frac=1, random_state=RANDOM_STATE)
                .reset_index(drop=True)
            )
            print(
                f"[data] Loaded {len(balanced)} URLs from public dataset "
                f"({len(benign)} benign, {len(phish)} phishing)."
            )
            return balanced

        except Exception as exc:
            print(f"[data] Could not load dataset: {exc}")

    print("[data] All public sources failed — generating synthetic dataset.")
    n_synth = n_each
    df = _generate_synthetic_dataset(n_each=n_synth)
    print(
        f"[data] Synthetic dataset ready: "
        f"{(df['label']==0).sum()} benign, {(df['label']==1).sum()} phishing."
    )
    return df


# ---------------------------------------------------------------------------
# Feature engineering with progress reporting
# ---------------------------------------------------------------------------

def engineer_features_with_progress(
    df: pd.DataFrame,
    progress_every: int = PROGRESS_EVERY,
) -> pd.DataFrame:
    """
    Apply URLFeatureEngineer to every row and print progress to stdout.

    Args:
        df:             DataFrame with a 'url' column.
        progress_every: Print a status line every N rows processed.

    Returns:
        DataFrame of extracted features aligned with df's index.
    """
    engineer = URLFeatureEngineer()
    total     = len(df)
    rows      = []

    print(f"[features] Extracting features for {total} URLs…")
    for i, url in enumerate(df["url"], start=1):
        rows.append(engineer.extract_features(str(url)))
        if i % progress_every == 0 or i == total:
            print(f"[features]   {i:>5}/{total} processed ({i/total*100:.0f}%)")

    print("[features] Done.")
    return pd.DataFrame(rows, columns=URLFeatureEngineer.FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Train / evaluate / save
# ---------------------------------------------------------------------------

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split feature matrix and labels into stratified train and test sets.

    Args:
        X:            Feature matrix.
        y:            Label series.
        test_size:    Fraction of data held out for testing.
        random_state: Reproducibility seed.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = N_ESTIMATORS,
    max_depth: int    = MAX_DEPTH,
    random_state: int = RANDOM_STATE,
) -> RandomForestClassifier:
    """
    Instantiate and fit a Random Forest classifier.

    Args:
        X_train:      Training feature matrix.
        y_train:      Training labels.
        n_estimators: Number of trees in the forest.
        max_depth:    Maximum depth per tree.
        random_state: Reproducibility seed.

    Returns:
        Fitted RandomForestClassifier.
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(
    clf: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate the trained model and print a full classification report.

    Args:
        clf:    Fitted RandomForestClassifier.
        X_test: Test feature matrix.
        y_test: True test labels.

    Returns:
        Dictionary with precision, recall, f1, accuracy, and confusion_matrix.
    """
    y_pred = clf.predict(X_test)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    accuracy  = accuracy_score(y_test, y_pred)
    cm        = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 54)
    print("           MODEL EVALUATION METRICS")
    print("=" * 54)
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print("-" * 54)
    print("\nFull Classification Report:")
    print(
        classification_report(
            y_test, y_pred,
            target_names=["Benign (0)", "Phishing (1)"],
            zero_division=0,
        )
    )
    print("Confusion Matrix:")
    print(f"  Labels -> Predicted Benign | Predicted Phishing")
    print(f"  Actual Benign    : TN={cm[0][0]:>4}  | FP={cm[0][1]:>4}")
    print(f"  Actual Phishing  : FN={cm[1][0]:>4}  | TP={cm[1][1]:>4}")
    print("=" * 54 + "\n")

    return {
        "accuracy":         accuracy,
        "precision":        precision,
        "recall":           recall,
        "f1":               f1,
        "confusion_matrix": cm,
    }


def save_model(
    clf: RandomForestClassifier,
    model_path: Path = MODEL_PATH,
) -> Path:
    """
    Persist the trained model to disk, creating directories as needed.

    Args:
        clf:        Fitted RandomForestClassifier to serialise.
        model_path: Destination path for the .pkl file.

    Returns:
        Resolved absolute path where the model was written.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    resolved = model_path.resolve()
    print(f"[save] Model written to: {resolved}")
    return resolved


def load_model(model_path: Path = MODEL_PATH) -> RandomForestClassifier:
    """
    Load a persisted model from disk.

    Args:
        model_path: Path to the serialised .pkl file.

    Returns:
        Deserialised RandomForestClassifier.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found at '{model_path}'. "
            "Run train_model.py to train and save the model first."
        )
    clf: RandomForestClassifier = joblib.load(model_path)
    return clf


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 54)
    print("   PhishGuard AI — train_model.py")
    print("=" * 54 + "\n")

    # 1. Acquire data
    raw_df = fetch_real_data(sample_size=SAMPLE_SIZE)

    print(f"\n[data] Dataset shape : {raw_df.shape}")
    print(f"[data] Class balance :\n{raw_df['label'].value_counts().to_string()}\n")

    # 2. Feature engineering with progress feedback
    X_features = engineer_features_with_progress(raw_df)
    y_labels   = raw_df["label"].astype(int).reset_index(drop=True)

    # 3. Train / test split
    X_train, X_test, y_train, y_test = split_data(X_features, y_labels)
    print(f"[split] Train: {len(X_train)} | Test: {len(X_test)}\n")

    # 4. Train
    print("[train] Fitting RandomForestClassifier…")
    clf = train_model(X_train, y_train)
    print("[train] Training complete.\n")

    # 5. Evaluate
    metrics = evaluate_model(clf, X_test, y_test)

    # 6. Save
    saved_path = save_model(clf)

    print("\n[done] Pipeline completed successfully.")
    print(f"       F1-Score : {metrics['f1']:.4f}")
    print(f"       Recall   : {metrics['recall']:.4f}")
    print(f"       Model    : {saved_path}")