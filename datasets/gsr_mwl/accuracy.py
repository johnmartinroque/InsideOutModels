import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# Folder paths
HIGH_PATH = "High_MWL"
LOW_PATH = "Low_MWL"
PERSONAL_PATH = "personal_Low_MWL/person_Low_MWL.csv"

# ------------------------------------------------------------
# FEATURE EXTRACTION
# ------------------------------------------------------------

def extract_gsr_features(df):
    """Extract statistical and GSR-related features."""
    features = []

    for col in df.columns:
        coldata = df[col].dropna()
        if len(coldata) == 0:
            continue

        # Basic stats
        base_feats = [
            np.mean(coldata),
            np.std(coldata),
            np.min(coldata),
            np.max(coldata),
            np.median(coldata),
            np.percentile(coldata, 25),
            np.percentile(coldata, 75),
            np.ptp(coldata),
        ]
        features.extend(base_feats)

        # Difference-based features
        if len(coldata) > 1:
            diff = np.diff(coldata)
            diff_feats = [
                np.mean(np.abs(diff)),
                np.std(diff),
                np.max(np.abs(diff)),
            ]
        else:
            diff_feats = [0, 0, 0]

        features.extend(diff_feats)

    return features


# ------------------------------------------------------------
# LOAD REFERENCE DATA
# ------------------------------------------------------------

def load_reference_data():
    X = []
    y = []

    # Load HIGH MWL training data
    for i in range(2, 26):
        file = f"p{i}h.csv"
        path = os.path.join(HIGH_PATH, file)

        if os.path.exists(path):
            df = pd.read_csv(path, header=None)
            df = df.apply(pd.to_numeric, errors="coerce").dropna()

            if not df.empty:
                X.append(extract_gsr_features(df))
                y.append(1)

    # Load LOW MWL training data
    for i in range(2, 26):
        file = f"p{i}l.csv"
        path = os.path.join(LOW_PATH, file)

        if os.path.exists(path):
            df = pd.read_csv(path, header=None)
            df = df.apply(pd.to_numeric, errors="coerce").dropna()

            if not df.empty:
                X.append(extract_gsr_features(df))
                y.append(0)

    return np.array(X), np.array(y)


# ------------------------------------------------------------
# LOAD PERSONAL DATA (TEST ONLY)
# ------------------------------------------------------------

def load_personal_data():
    df = pd.read_csv(PERSONAL_PATH, header=None)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    # If only 1 column, duplicate it to match training feature size
    if df.shape[1] == 1:
        df = pd.concat([df, df], axis=1)

    features = extract_gsr_features(df)

    return np.array([features])



# ------------------------------------------------------------
# MODEL EVALUATION
# ------------------------------------------------------------

def evaluate_models(X_train, y_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Neural Network": MLPClassifier(max_iter=1000, hidden_layer_sizes=(100, 50))
    }

    results = {}

    print("\n======================================")
    print("MODEL ACCURACY USING PERSONAL DATA")
    print("======================================\n")

    # PERSONAL LABEL → Based on folder: Low MWL → label = 0
    true_label = np.array([0])

    for name, model in models.items():
        try:
            # Fit on training set
            model.fit(X_train_scaled, y_train)

            # Predict on personal data
            pred = model.predict(X_test_scaled)

            accuracy = accuracy_score(true_label, pred)

            results[name] = accuracy

            print(f"{name}: {accuracy:.4f} | Predicted: {pred[0]} (0=Low, 1=High)")

        except Exception as e:
            print(f"{name} ERROR → {e}")
            results[name] = None

    return results


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":
    print("Loading reference dataset...")
    X_train, y_train = load_reference_data()

    print(f"Loaded {len(X_train)} reference samples.")
    print(f"Feature size: {X_train.shape[1]}")

    print("\nLoading personal data (test only)...")
    X_test = load_personal_data()

    print("Evaluating models...\n")
    results = evaluate_models(X_train, y_train, X_test)

    print("\n======================================")
    print("SUMMARY TABLE")
    print("======================================")
    print(f"{'Model':<25} {'Accuracy':<10}")
    print("-" * 40)

    for m, acc in results.items():
        print(f"{m:<25} {acc:.4f}")
