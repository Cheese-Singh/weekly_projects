import pandas as pd
import kagglehub
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import numpy as np

path = kagglehub.dataset_download("ritwikb3/heart-disease-statlog")
print("Path to dataset files:", path)
dataset_path = "/Users/ekamveersingh/.cache/kagglehub/datasets/ritwikb3/heart-disease-statlog/versions/1"

file_name = "Heart_disease_statlog.csv"  
file_path = os.path.join(dataset_path, file_name)

df = pd.read_csv(file_path)

X = df.drop("target", axis=1).values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=81, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

kernels = [ 'linear' , 'poly' , 'rbf' ]
results = []

for kernel in kernels:
    svm = SVC(kernel=kernel, gamma="scale", C=1, degree=3)
    svm.fit(X_train_scaled, y_train)
    y_pred = svm.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    results.append({
        'Kernel':kernel.capitalize(),
        'Accuracy':round(acc,4)
    }
    )

results_df = pd.DataFrame(results)
print("\nSVM Kernel Comparison on Dataset\n")
print(results_df)

best_kernel = results_df.loc[results_df['Accuracy'].idxmax(), 'Kernel']
print(f"\nBest kernel: {best_kernel}")

X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=81, stratify=y
)

svm = SVC(kernel='poly', gamma='scale', C=1)
svm.fit(X_train, y_train)

def plot_svm_boundaries(X_train, y_train, kernel='poly', C=1.0, filename="plot.png"):
    svm = SVC(kernel=kernel, C=C, gamma='scale')
    svm.fit(X_train, y_train)

    plt.figure(figsize=(10, 7))
   
    h = .02 
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", edgecolors='k', s=40)

    plt.scatter(
        svm.support_vectors_[:, 0],
        svm.support_vectors_[:, 1],
        s=100, facecolors='none', edgecolors='yellow', 
        linewidths=1.5, label='Support Vectors'
    )

    plt.title(f"SVM Boundary: Kernel={kernel}, C={C}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()

    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.show()
    plt.close()

plot_svm_boundaries(X_train, y_train, kernel, C=10.0, filename="plot1.png")
plot_svm_boundaries(X_train, y_train, kernel, C=0.1, filename="plot10.png")
plot_svm_boundaries(X_train, y_train, kernel, C=100.0, filename="plot100.png")