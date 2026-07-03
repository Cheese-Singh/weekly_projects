# ML Foundations

A progressive series of machine learning scripts, each focused on a single concept and designed to build intuition from the ground up.

---

## Technical Objectives

This repository documents:
- Systematic exploration of classical ML concepts through hands-on implementation
- Progression from model comparison and visualization to automated data workflows
- Development of reusable utilities that reduce boilerplate in future projects

Each script emphasizes understanding underlying mechanics rather than relying solely on high-level abstractions.

---

## Contents

### [`Everything_classifier.py`](./A1_Everything_classifier.py)
* **Description:** Runs KNN, SVM, Decision Tree, and other classifiers on any toy dataset and prints accuracy scores in a formatted table.
* **Stack:** `scikit-learn`, `pandas`, `tabulate`

---

### [`Visualizing_hyperplanes.py`](./A2_Visualizing_hyperplanes.py)
* **Description:** Plots SVM decision boundaries on 2D data, showing how they shift with different C values and kernels. Uses PCA for dimensionality reduction where needed.
* **Outputs:** Saved plots in root directory (`plot1.png`, `plot10.png`, etc.)
* **Stack:** `scikit-learn`, `matplotlib`, `numpy`

---

### [`Automated_EDA_template.py`](./A3_Automated_EDA_template.py)
* **Description:** A drop-in EDA function that takes any CSV and automatically generates correlation heatmaps and histograms for all numerical columns.
* **Outputs:** Saved reports in `eda_outputs/`
* **Stack:** `pandas`, `seaborn`

---

### [`Sklearn_preprocessing_pipeline.py`](./A4_Sklearn_preprocessing_pipeline.py)
* **Description:** Demonstrates an automated preprocessing workflow for a dirty tabular dataset using `Pipeline` and `ColumnTransformer`. Handles missing-value imputation, numerical feature scaling, categorical encoding, and column dropping inside a single reusable sklearn preprocessing object.
* **Dataset:** Titanic dataset from Kaggle
* **Focus:** Zero manual data-cleaning outside the pipeline; separates numerical and categorical transformations cleanly.
* **Stack:** `pandas`, `scikit-learn`, `kaggle`

---

### [`PCA_vs_TSNE_fashion_mnist.py`](./A5_PCA_vs_TSNE_fashion_mnist.py)
* **Description:** Compares PCA and t-SNE on the Fashion-MNIST dataset using an interactive side-by-side Plotly visualization. Demonstrates the difference between linear dimensionality reduction and nonlinear neighborhood-preserving projection.
* **Dataset:** Fashion-MNIST from Kaggle
* **Focus:** Dimensionality reduction, cluster visualization, PCA vs t-SNE comparison, reusable preprocessing pipeline integration.
* **Output:** Interactive HTML visualization saved as `dimension_reduction_comparison.html`
* **Stack:** `pandas`, `scikit-learn`, `plotly`, `kaggle`

---

### [`Audio_processing_and_ASR.py`](./audio_processing_and_ASR.py)
* **Description:** A comprehensive exploration of audio engineering fundamentals and local automatic speech recognition (ASR). It features live wake-word audio listening, low-level binary WAV data manipulation, runtime waveform plotting, multi-buffered microphone recording, and hardware-accelerated local transcription.
* **Dataset/Input:** Live local microphone stream and a standard test file (harvard.wav)
* **Focus:** Digital signal processing basics, handling audio byte-buffers, real-time audio I/O streaming, and Apple Silicon optimized deep-learning transcription engines.
* **Stack:** `pyaudio`, `wave`, `speech_recognition`, `mlx-whisper`, `numpy`, `matplotlib`
