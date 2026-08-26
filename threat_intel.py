import re
import math
import pandas as pd
import numpy as np

from urllib.parse import urlparse

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. URL FEATURE EXTRACTION
# ============================================================

SUSPICIOUS_WORDS = [
    "login",
    "signin",
    "verify",
    "verification",
    "account",
    "secure",
    "update",
    "password",
    "confirm",
    "bank",
    "payment",
    "wallet",
    "free",
    "bonus",
    "claim",
]


def entropy(text):
    """Calculate Shannon entropy."""
    if not text:
        return 0

    probabilities = [
        text.count(char) / len(text)
        for char in set(text)
    ]

    return -sum(p * math.log2(p) for p in probabilities)


def extract_features(url):
    """Convert a URL into numerical threat-intelligence features."""

    url = str(url)

    parsed = urlparse(url if "://" in url else "http://" + url)

    hostname = parsed.netloc
    path = parsed.path
    query = parsed.query

    # Remove username/password and port information
    hostname_clean = hostname.split("@")[-1].split(":")[0]

    features = {}

    # Basic URL statistics
    features["url_length"] = len(url)
    features["hostname_length"] = len(hostname_clean)
    features["path_length"] = len(path)
    features["query_length"] = len(query)

    # Character counts
    features["num_dots"] = url.count(".")
    features["num_hyphens"] = url.count("-")
    features["num_underscores"] = url.count("_")
    features["num_slashes"] = url.count("/")
    features["num_question_marks"] = url.count("?")
    features["num_equals"] = url.count("=")
    features["num_ampersands"] = url.count("&")
    features["num_percent"] = url.count("%")
    features["num_at"] = url.count("@")

    # Digits
    features["num_digits"] = sum(c.isdigit() for c in url)

    # Letters
    features["num_letters"] = sum(c.isalpha() for c in url)

    # Digit-to-length ratio
    features["digit_ratio"] = (
        features["num_digits"] / len(url)
        if len(url) > 0
        else 0
    )

    # HTTPS
    features["uses_https"] = int(parsed.scheme.lower() == "https")

    # IP address instead of normal hostname
    ip_pattern = r"^\d{1,3}(\.\d{1,3}){3}$"
    features["is_ip_address"] = int(
        bool(re.match(ip_pattern, hostname_clean))
    )

    # Number of subdomains
    if hostname_clean:
        features["num_subdomains"] = max(
            len(hostname_clean.split(".")) - 2,
            0
        )
    else:
        features["num_subdomains"] = 0

    # Query parameters
    features["num_query_params"] = (
        len(query.split("&")) if query else 0
    )

    # Suspicious keyword count
    lower_url = url.lower()

    features["suspicious_word_count"] = sum(
        word in lower_url
        for word in SUSPICIOUS_WORDS
    )

    # URL entropy
    features["url_entropy"] = entropy(url)

    # Long hostname
    features["long_hostname"] = int(
        features["hostname_length"] > 50
    )

    # Many special characters
    special_chars = sum(
        not c.isalnum() for c in url
    )

    features["special_char_ratio"] = (
        special_chars / len(url)
        if len(url) > 0
        else 0
    )

    return features


# ============================================================
# 2. LOAD PHIUSIIL DATASET
# ============================================================

DATASET = "urls.csv"

print("\n[+] Loading PhiUSIIL dataset...")

df = pd.read_csv(DATASET)

print(f"[+] Dataset size: {len(df):,} URLs")

# PhiUSIIL uses:
# label = 1 -> legitimate
# label = 0 -> phishing
#
# Our project:
# target = 0 -> benign
# target = 1 -> malicious

df = df[["URL", "label"]].dropna()

df = df.rename(columns={
    "URL": "url",
    "label": "target"
})

# Convert label to integer
df["target"] = df["target"].astype(int)

# Invert labels:
# legitimate (1) -> benign (0)
# phishing (0)   -> malicious (1)

df["target"] = 1 - df["target"]

print("\n[+] Class distribution:")

print(
    df["target"]
    .value_counts()
    .rename({
        0: "Benign",
        1: "Malicious"
    })
)

# Use 20,000 samples initially
df = df.sample(
    n=20000,
    random_state=42
)

print(f"\n[+] Using {len(df):,} URLs for this experiment.")


df = df.sample(
    n=20000,
    random_state=42
)

# ============================================================
# 3. FEATURE EXTRACTION
# ============================================================

print("\n[+] Extracting URL features...")

feature_df = pd.DataFrame(
    df["url"].apply(extract_features).tolist()
)

X = feature_df
y = df["target"]

print(f"[+] Features extracted: {X.shape[1]}")

print("\nFeatures:")
for feature in X.columns:
    print(" -", feature)


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\n[+] Training samples:", len(X_train))
print("[+] Testing samples:", len(X_test))


# ============================================================
# 5. RANDOM FOREST MODEL
# ============================================================

print("\n[+] Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

model.fit(X_train, y_train)

print("[+] Model training complete.")


# ============================================================
# 6. EVALUATION
# ============================================================

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_probability)

print("\n" + "=" * 60)
print("THREAT INTELLIGENCE MODEL RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

print("\nPercentage format:")
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")
print(f"ROC-AUC  : {auc * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Benign", "Malicious"],
        zero_division=0,
    )
)


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benign", "Malicious"],
    yticklabels=["Benign", "Malicious"],
)

plt.title("URL Threat Detection - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)

plt.show()


# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_,
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\n" + "=" * 60)
print("TOP THREAT INDICATORS")
print("=" * 60)

print(importance.head(10).to_string(index=False))


plt.figure(figsize=(10, 6))

sns.barplot(
    data=importance.head(10),
    x="importance",
    y="feature",
)

plt.title("Top URL Threat Indicators")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)

plt.show()


# ============================================================
# 9. THREAT SCORING FUNCTION
# ============================================================

def calculate_threat_score(url):
    """
    Return a 0-100 threat score.
    """

    features = extract_features(url)

    input_df = pd.DataFrame([features])

    probability = model.predict_proba(input_df)[0][1]

    score = round(probability * 100, 2)

    if score >= 80:
        severity = "CRITICAL"
    elif score >= 60:
        severity = "HIGH"
    elif score >= 30:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    return score, severity


# ============================================================
# 10. TEST URLS
# ============================================================

test_urls = [
    "https://www.google.com",
    "https://www.microsoft.com",
    "http://192.168.1.10/login",
    "http://secure-login-account-verification.example.com/login",
]

print("\n" + "=" * 60)
print("THREAT INTELLIGENCE SCAN")
print("=" * 60)

for url in test_urls:

    score, severity = calculate_threat_score(url)

    print("\nURL:", url)
    print("Threat Score:", score, "/ 100")
    print("Severity:", severity)


# ============================================================
# 11. SAVE RESULTS
# ============================================================

results = X_test.copy()

results["actual"] = y_test.values
results["predicted"] = y_pred
results["malicious_probability"] = y_probability

results.to_csv(
    "threat_detection_results.csv",
    index=False
)

print("\n[+] Results saved to threat_detection_results.csv")
print("[+] Confusion matrix saved to confusion_matrix.png")
print("[+] Feature importance saved to feature_importance.png")

print("\n[+] Project complete.")
