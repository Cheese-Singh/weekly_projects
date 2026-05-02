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

### [`Everything_classifier.py`](./01_Everything_classifier.py)
* **Description:** Runs KNN, SVM, Decision Tree, and other classifiers on any toy dataset and prints accuracy scores in a formatted table.
* **Stack:** `scikit-learn`, `pandas`, `tabulate`

---

### [`Visualizing_hyperplanes.py`](./02_Visualizing_hyperplanes.py)
* **Description:** Plots SVM decision boundaries on 2D data, showing how they shift with different C values and kernels. Uses PCA for dimensionality reduction where needed.
* **Outputs:** Saved plots in root directory (`plot1.png`, `plot10.png`, etc.)
* **Stack:** `scikit-learn`, `matplotlib`, `numpy`

---

### [`Automated_EDA_template.py`](./03_Automated_EDA_template.py)
* **Description:** A drop-in EDA function that takes any CSV and automatically generates correlation heatmaps and histograms for all numerical columns.
* **Outputs:** Saved reports in `eda_outputs/`
* **Stack:** `pandas`, `seaborn`

---

### [`Sklearn_preprocessing_pipeline.py`](./04_Sklearn_preprocessing_pipeline.py)
* **Description:** Demonstrates an automated preprocessing workflow for a dirty tabular dataset using `Pipeline` and `ColumnTransformer`. Handles missing-value imputation, numerical feature scaling, categorical encoding, and column dropping inside a single reusable sklearn preprocessing object.
* **Dataset:** Titanic dataset from Kaggle
* **Focus:** Zero manual data-cleaning outside the pipeline; separates numerical and categorical transformations cleanly.
* **Stack:** `pandas`, `scikit-learn`, `kaggle`
