import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# ================= LOAD DATA =================
file_path = os.path.join(os.path.dirname(__file__), "heart_rate_emotion_dataset.csv")
df = pd.read_csv(file_path)

print("Original Dataset Size:", len(df))

# ================= REMOVE UNWANTED EMOTIONS =================
df = df[~df["Emotion"].isin(["disgust", "surprise", "fear", "angry"])]

print("Filtered Dataset Size:", len(df))
print("Remaining Emotions:", df["Emotion"].unique())


# ================= PREPARE DATA =================
X = df[["HeartRate"]]   # Feature
y = df["Emotion"]       # Target

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ================= MODELS =================
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

print("\n================ MODEL ACCURACY RESULTS ================\n")

for name, model in models.items():
    
    # Use Pipeline for scaling (important for SVM, KNN, Logistic)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    
    print(f"{name} Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("------------------------------------------------------")