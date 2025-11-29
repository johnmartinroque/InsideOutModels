import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# ===== IMPORT FUNCTION FROM accuracy.py =====
from accuracy import load_and_prepare_data, evaluate_models

HIGH_PATH = "High_MWL"
LOW_PATH = "Low_MWL"

# ===== LOAD DATA AND MODELS =====
print("Loading data...")
X, y = load_and_prepare_data(HIGH_PATH, LOW_PATH)
results, X_train, X_test, y_train, y_test, models = evaluate_models(X, y)

# ===== STYLE SETTINGS =====
plt.style.use('seaborn-v0_8')

output_dir = "visualizations"
os.makedirs(output_dir, exist_ok=True)

print("Generating larger, clearer plots...")

# Common font sizes
TITLE_FONTSIZE = 20
AXIS_FONTSIZE = 16
TICK_FONTSIZE = 14

# ----------------------------------------------------
# 1. Dataset distribution
# ----------------------------------------------------
plt.figure(figsize=(10, 8))
sns.countplot(x=y)
plt.title("Dataset Distribution: Low vs High MWL", fontsize=TITLE_FONTSIZE)
plt.xticks([0, 1], ["Low MWL", "High MWL"], fontsize=TICK_FONTSIZE)
plt.xlabel("Class", fontsize=AXIS_FONTSIZE)
plt.ylabel("Count", fontsize=AXIS_FONTSIZE)
plt.tight_layout()
plt.savefig(f"{output_dir}/dataset_distribution.png")
plt.close()

# ----------------------------------------------------
# 2. PCA Visualization (2D)
# ----------------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", alpha=0.7)
plt.title("PCA Visualization of GSR Features", fontsize=TITLE_FONTSIZE)
plt.xlabel("PC1", fontsize=AXIS_FONTSIZE)
plt.ylabel("PC2", fontsize=AXIS_FONTSIZE)
plt.colorbar(scatter, label="0 = Low MWL, 1 = High MWL")
plt.tight_layout()
plt.savefig(f"{output_dir}/pca_plot.png")
plt.close()

# ----------------------------------------------------
# 3. Feature correlation heatmap
# ----------------------------------------------------
pca_corr = PCA(n_components=min(20, X.shape[1]))  # max 20 PCs
X_pca_corr = pca_corr.fit_transform(X)

df_pca = pd.DataFrame(X_pca_corr, columns=[f"PC{i+1}" for i in range(X_pca_corr.shape[1])])

plt.figure(figsize=(16, 14))
sns.heatmap(df_pca.corr(), cmap="coolwarm", cbar=True, annot=False)
plt.title("Correlation Heatmap (First 20 PCA Components)", fontsize=TITLE_FONTSIZE)
plt.tight_layout()
plt.savefig(f"{output_dir}/heatmap_pca.png")
plt.close()
# ----------------------------------------------------
# 4. Model accuracy comparison
# ----------------------------------------------------
model_names = []
accuracies = []

for name, result in results.items():
    if result is not None:
        model_names.append(name)
        accuracies.append(result["accuracy"])

plt.figure(figsize=(14, 10))
sns.barplot(x=accuracies, y=model_names, palette="viridis")
plt.title("Model Test Accuracies", fontsize=TITLE_FONTSIZE)
plt.xlabel("Accuracy", fontsize=AXIS_FONTSIZE)
plt.ylabel("Model", fontsize=AXIS_FONTSIZE)
plt.xlim(0, 1)
plt.xticks(fontsize=TICK_FONTSIZE)
plt.yticks(fontsize=TICK_FONTSIZE)
plt.tight_layout()
plt.savefig(f"{output_dir}/model_accuracies.png")
plt.close()

# ----------------------------------------------------
# 5. Confusion matrix (per model)
# ----------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train2)
X_test_scaled = scaler.transform(X_test2)

for name, model in models.items():

    # Scale for specific models
    if name in ["SVM", "Neural Network", "K-Nearest Neighbors"]:
        model.fit(X_train_scaled, y_train2)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train2, y_train2)
        preds = model.predict(X_test2)

    cm = confusion_matrix(y_test2, preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}", fontsize=TITLE_FONTSIZE)
    plt.xlabel("Predicted", fontsize=AXIS_FONTSIZE)
    plt.ylabel("Actual", fontsize=AXIS_FONTSIZE)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cm_{name.replace(' ', '_')}.png")
    plt.close()

# ----------------------------------------------------
# 6. Cross-validation boxplot
# ----------------------------------------------------
cv_data = {}

for name, result in results.items():
    if result is not None:
        cv_data[name] = result["cv_scores"]

plt.figure(figsize=(14, 10))
sns.boxplot(data=pd.DataFrame(cv_data))
plt.title("Cross-Validation Performance per Model", fontsize=TITLE_FONTSIZE)
plt.ylabel("Accuracy", fontsize=AXIS_FONTSIZE)
plt.xticks(rotation=45, fontsize=TICK_FONTSIZE)
plt.yticks(fontsize=TICK_FONTSIZE)
plt.tight_layout()
plt.savefig(f"{output_dir}/cv_boxplot.png")
plt.close()

print("\nAll visualizations saved in:")
print(f"➡ {os.path.abspath(output_dir)}")
print("\nDone!")
