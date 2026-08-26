# URL Threat Detection using Machine Learning
A machine learning project for detecting potentially malicious and phishing URLs using URL-based threat intelligence features.

The project extracts features from URLs and uses a Random Forest Classifier to classify URLs as Benign or Malicious. It also generates a 0–100 threat score and severity level for individual URLs.

## Features
 - URL-based feature extraction
 - Phishing and malicious URL classification
 - Random Forest machine learning model
 - Threat score from 0–100
 - Severity classification: Low, Medium, High, Critical
 - Confusion matrix and feature importance visualization
 - CSV export of detection results

## Dataset
This project uses the PhiUSIIL Phishing URL Dataset.
The experiment uses a sample of 20,000 URLs from the dataset, with an 80/20 train-test split.

### Dataset file:

urls.csv

### URL Features
The model extracts 24 features, including:
 - URL and hostname length
 - Number of dots, slashes, digits, and special characters
 - HTTPS usage
 - IP address detection
 - Number of subdomains
 - Query parameters
 - Suspicious keywords
 - URL entropy
 - Digit and special-character ratios

## Model
The project uses a Random Forest classifier:
```text
RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42
)
```
## Installation
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage
Place urls.csv in the project directory and run:
```bash
python threat_intel.py
```
The program trains the model, evaluates it, scans example URLs, and generates the output files.

## Output
The following files are generated:
 - confusion_matrix.png
 - feature_importance.png
 - threat_detection_results.csv

Detailed model results and visualizations are included in the repository.
