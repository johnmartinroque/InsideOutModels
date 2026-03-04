import pandas as pd
import os
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# ================= LOAD DATA =================
file_path = os.path.join(os.path.dirname(__file__), "heart_rate_emotion_dataset.csv")
df = pd.read_csv(file_path)

# Remove unwanted emotions
df = df[~df["Emotion"].isin(["disgust", "surprise", "fear", "angry"])]

# Features and target
X = df[["HeartRate"]]
y = df["Emotion"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ================= TRAIN LOGISTIC REGRESSION =================
lr_model = LogisticRegression()
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", lr_model)
])
pipeline.fit(X_train, y_train)

# ================= RANDOM BPM PREDICTION =================
num_samples = 50
random_bpms = np.random.randint(50, 120, size=(num_samples, 1))  # 50-120 BPM

# Predict classes
y_random_pred = pipeline.predict(random_bpms)

# Predict probabilities
y_random_proba = pipeline.predict_proba(random_bpms)

# Map back to emotion labels
predicted_emotions = label_encoder.inverse_transform(y_random_pred)

# Get max confidence per prediction
prediction_confidence = np.max(y_random_proba, axis=1)

# Create a DataFrame to view results
test_df = pd.DataFrame({
    "Random_BPM": random_bpms.flatten(),
    "Predicted_Emotion": predicted_emotions,
    "Confidence": prediction_confidence
})

print("\nRandom BPM Predictions with Confidence:")
print(test_df)