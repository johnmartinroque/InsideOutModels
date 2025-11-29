import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ============================
#   IMPORT FROM accuracy.py
# ============================
from accuracy import df, models, X, y

plt.style.use("seaborn-v0_8")

# Create output folder
output_dir = "visualizations"
os.makedirs(output_dir, exist_ok=True)

print("Generating visualizations...")

# ============================================================
# 1. Dataset distribution
# ============================================================
plt.figure(figsize=(10, 6))
sns.countplot(x=y, palette="viridis")
plt.title("Dataset Distribution (Calm vs Stressed)", fontsize=18)
plt.xlabel("Emotional State", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.tight_layout()
plt.savefig(f"{output_dir}/dataset_distribution.png")
plt.close()

# ============================================================
# 2. PCA Visualization
# ============================================================
print("Preparing PCA...")

if X.shape[1] == 1:
    # Only one feature → PCA 2D impossible → downgrade
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], np.zeros_like(X_pca[:, 0]),
                c=pd.Categorical(y).codes, cmap="coolwarm", alpha=0.7)
    plt.title("PCA Visualization (1D Feature → 1 Component)", fontsize=18)
    plt.xlabel("PC1", fontsize=14)
    plt.yticks([])
else:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1],
                c=pd.Categorical(y).codes, cmap="coolwarm", alpha=0.7)
    plt.title("PCA Visualization (2D)", fontsize=18)
    plt.xlabel("PC1", fontsize=14)
    plt.ylabel("PC2", fontsize=14)

plt.tight_layout()
plt.savefig(f"{output_dir}/pca_plot.png")
plt.close()

# ============================================================
# 3. Correlation Heatmap
# ============================================================
df_corr = pd.DataFrame(X, columns=["GSR_Values"])

plt.figure(figsize=(8, 6))
sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (Only 1 Feature)", fontsize=18)
plt.tight_layout()
plt.savefig(f"{output_dir}/heatmap_features.png")
plt.close()

# ============================================================
# 4. Model Accuracy Bar Graph
# ============================================================
from accuracy import results   # results list created in accuracy.py

model_names = [r["Model"] for r in results]
accuracies = [r["Test Accuracy"] for r in results]

plt.figure(figsize=(12, 8))
sns.barplot(x=accuracies, y=model_names, palette="magma")
plt.title("Model Test Accuracies (GSR → Emotional State)", fontsize=20)
plt.xlabel("Accuracy", fontsize=16)
plt.ylabel("Model", fontsize=16)
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig(f"{output_dir}/model_accuracies.png")
plt.close()

# ============================================================
# 5. Confusion Matrices for Each Model
# ============================================================
print("Generating confusion matrices...")

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train2)
X_test_scaled = scaler.transform(X_test2)

for name, model in models.items():
    # Scale-required models
    if name in ["Logistic Regression", "SVM (RBF Kernel)",
                "K-Nearest Neighbors", "Neural Network (MLP)"]:
        model.fit(X_train_scaled, y_train2)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train2, y_train2)
        preds = model.predict(X_test2)

    cm = confusion_matrix(y_test2, preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}", fontsize=18)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("Actual", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cm_{name.replace(' ', '_')}.png")
    plt.close()

# ============================================================
# 6. Cross-validation boxplot
# ============================================================
cv_dict = {r["Model"]: r["CV Mean"] for r in results}

plt.figure(figsize=(10, 6))
sns.barplot(x=list(cv_dict.keys()), y=list(cv_dict.values()), palette="cubehelix")
plt.xticks(rotation=45, ha="right")
plt.ylabel("CV Mean Accuracy")
plt.title("Cross-Validation Mean Accuracy per Model", fontsize=18)
plt.tight_layout()
plt.savefig(f"{output_dir}/cv_scores.png")
plt.close()

print("\nAll visualizations saved inside:")
print(f"➡ {os.path.abspath(output_dir)}")
print("Done!")
