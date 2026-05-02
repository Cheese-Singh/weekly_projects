from sklearn.decomposition import PCA
from A4_Sklearn_preprocessing_pipeline import build_preprocessor
from sklearn.manifold import TSNE
import plotly.express as px
from plotly.subplots import make_subplots

import os
import zipfile
import pandas as pd

os.system("kaggle datasets download -d zalando-research/fashionmnist")

with zipfile.ZipFile("fashionmnist.zip", "r") as zip_ref:
    zip_ref.extractall("fashion_mnist_data")

df = pd.read_csv("fashion_mnist_data/fashion-mnist_train.csv")

label_names = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

df = df.sample(n=3000, random_state=42)
pca = PCA(n_components=2, random_state=42)

X = df.drop(columns=["label"])
y = df["label"]

preprocessor = build_preprocessor(drop_columns=[])
X_processed = preprocessor.fit_transform(X)
X_pca = pca.fit_transform(X_processed)

tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42)
X_tsne = tsne.fit_transform(X_processed)

plot_df = pd.DataFrame({
    "PCA_1": X_pca[:, 0],
    "PCA_2": X_pca[:, 1],
    "TSNE_1": X_tsne[:, 0],
    "TSNE_2": X_tsne[:, 1],
    "label": y.values
})

plot_df["class_name"] = plot_df["label"].map(label_names)
plot_df["label"] = plot_df["label"].astype(str)

fig = make_subplots(rows=1, cols=2, subplot_titles=["PCA", "t-SNE"])
pca_fig = px.scatter(plot_df, x="PCA_1", y="PCA_2", color="class_name", title="PCA")
tsne_fig = px.scatter(plot_df, x="TSNE_1", y="TSNE_2", color="class_name", title="t-SNE")

for trace in pca_fig.data:
    fig.add_trace(trace, row=1, col=1)

for trace in tsne_fig.data:
    fig.add_trace(trace, row=1, col=2)

fig.update_layout(height=600, width=1200, title_text="PCA vs t-SNE Visualization of Fashion MNIST")
fig.show()
fig.write_html("dimension_reduction_comparison.html")