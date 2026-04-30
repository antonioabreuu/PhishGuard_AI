"""
PhishGuard AI - Data Preparation Module
Ingestion and feature engineering for phishing URL classification.
"""

import math
import re
import socket
from collections import Counter
from typing import Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder


RISK_WORDS = [
    "login", "secure", "update", "verify", "account", "banking",
    "confirm", "password", "signin", "webscr", "ebayisapi", "paypal",
]

SUSPICIOUS_CHARS = ["@", "-", "=", "?", "%", "&", "#", "~"]


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load a CSV dataset containing URLs and labels.

    Args:
        filepath: Path to the CSV file. Expected columns: 'url', 'label'
                  where label is 1 for phishing and 0 for benign.

    Returns:
        DataFrame with 'url' and 'label' columns.
    """
    df = pd.read_csv(filepath, usecols=["url", "label"])
    df = df.dropna(subset=["url", "label"])
    df["url"] = df["url"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    return df


def generate_mock_dataframe() -> pd.DataFrame:
    """
    Generate a mock DataFrame with 10 URL examples for testing.

    Returns:
        DataFrame with 'url' and 'label' columns (1=phishing, 0=benign).
    """
    mock_data = {
        "url": [
            "http://192.168.1.1/login/secure-update?user=admin@evil.com",
            "https://www.google.com/search?q=python",
            "http://paypal-secure-verify.com/account/confirm=true",
            "https://github.com/scikit-learn/scikit-learn",
            "http://ebayisapi.evil-host.ru/signin?redirect=phish@bad.com",
            "https://stackoverflow.com/questions/tagged/pandas",
            "http://193.0.0.1/banking/update-password?token=abc&session=xyz",
            "https://docs.python.org/3/library/math.html",
            "http://secure-login.account-verify.com/webscr?cmd=login",
            "https://www.wikipedia.org/wiki/Machine_learning",
        ],
        "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    }
    return pd.DataFrame(mock_data)


def _calculate_entropy(text: str) -> float:
    """
    Calculate Shannon entropy for a given string.

    Args:
        text: Input string to compute entropy on.

    Returns:
        Shannon entropy value as float.
    """
    if not text:
        return 0.0
    frequency = Counter(text)
    total = len(text)
    entropy = -sum(
        (count / total) * math.log2(count / total)
        for count in frequency.values()
        if count > 0
    )
    return round(entropy, 4)


def _has_ip_in_host(url: str) -> int:
    """
    Detect whether the host portion of a URL is an IP address.

    Args:
        url: Raw URL string.

    Returns:
        1 if host is an IP address, 0 otherwise.
    """
    ip_pattern = re.compile(
        r"(?:https?://)?(\d{1,3}(?:\.\d{1,3}){3})(?:[:/]|$)"
    )
    match = ip_pattern.search(url)
    if match:
        try:
            socket.inet_aton(match.group(1))
            return 1
        except socket.error:
            return 0
    return 0


def _count_subdomains(url: str) -> int:
    """
    Count the number of subdomains in the host portion of a URL.

    Args:
        url: Raw URL string.

    Returns:
        Number of subdomains (dots in host minus 1, floored at 0).
    """
    host_pattern = re.compile(r"(?:https?://)?([^/?\s]+)")
    match = host_pattern.search(url)
    if not match:
        return 0
    host = match.group(1).split(":")[0]
    dot_count = host.count(".")
    return max(0, dot_count - 1)


def _has_risk_word(url: str) -> int:
    """
    Check if the URL contains any known phishing risk words.

    Args:
        url: Raw URL string.

    Returns:
        1 if any risk word is found, 0 otherwise.
    """
    url_lower = url.lower()
    return int(any(word in url_lower for word in RISK_WORDS))


class URLFeatureEngineer:
    """
    Extracts lexical features from raw URL strings for phishing detection.

    Features extracted:
        - url_length: Total character length of the URL.
        - dot_count: Number of dot characters in the URL.
        - has_ip_host: Whether the host is an IP address (binary).
        - has_https: Whether the URL uses HTTPS (binary).
        - suspicious_char_count: Count of suspicious characters (@, -, =, ?, etc.).
        - subdomain_count: Number of subdomains detected.
        - has_risk_word: Presence of phishing-related keywords (binary).
        - entropy: Shannon entropy of the full URL string.
    """

    FEATURE_COLUMNS = [
        "url_length",
        "dot_count",
        "has_ip_host",
        "has_https",
        "suspicious_char_count",
        "subdomain_count",
        "has_risk_word",
        "entropy",
    ]

    def extract_features(self, url: str) -> dict:
        """
        Extract all lexical features from a single URL.

        Args:
            url: Raw URL string.

        Returns:
            Dictionary mapping feature names to their computed values.
        """
        return {
            "url_length": len(url),
            "dot_count": url.count("."),
            "has_ip_host": _has_ip_in_host(url),
            "has_https": int(url.lower().startswith("https://")),
            "suspicious_char_count": sum(url.count(ch) for ch in SUSPICIOUS_CHARS),
            "subdomain_count": _count_subdomains(url),
            "has_risk_word": _has_risk_word(url),
            "entropy": _calculate_entropy(url),
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature extraction to all URLs in a DataFrame.

        Args:
            df: DataFrame containing a 'url' column.

        Returns:
            DataFrame of extracted features with columns matching FEATURE_COLUMNS.
        """
        features = df["url"].apply(self.extract_features)
        return pd.DataFrame(features.tolist(), columns=self.FEATURE_COLUMNS)


def build_pipeline(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Full data preparation pipeline: ingest raw data and return features and labels.

    This function accepts a raw DataFrame (as loaded from CSV or mock generator),
    applies URL feature engineering, and returns processed inputs ready for
    scikit-learn model training.

    Args:
        raw_df: DataFrame with at minimum 'url' (str) and 'label' (int) columns.

    Returns:
        Tuple of:
            - X (pd.DataFrame): Feature matrix with shape (n_samples, n_features).
            - y (pd.Series): Integer label series (1=phishing, 0=benign).

    Raises:
        ValueError: If required columns 'url' or 'label' are missing.
    """
    required_columns = {"url", "label"}
    missing = required_columns - set(raw_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in DataFrame: {missing}")

    raw_df = raw_df.dropna(subset=["url", "label"]).reset_index(drop=True)
    raw_df["url"] = raw_df["url"].astype(str).str.strip()

    engineer = URLFeatureEngineer()
    X: pd.DataFrame = engineer.transform(raw_df)
    y: pd.Series = raw_df["label"].astype(int).rename("label")

    return X, y


if __name__ == "__main__":
    print("=== PhishGuard AI — data_prep.py smoke test ===\n")

    mock_df = generate_mock_dataframe()
    print("Mock DataFrame:")
    print(mock_df.to_string(index=False))
    print()

    X, y = build_pipeline(mock_df)

    print("Feature Matrix (X):")
    print(X.to_string(index=False))
    print()

    print("Labels (y):")
    print(y.values)
    print()

    print(f"X shape: {X.shape} | y shape: {y.shape}")
    print("Feature columns:", list(X.columns))