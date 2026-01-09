# Machine Learning Tasks

A curated set of end‑to‑end machine learning notebooks and mini‑projects across core algorithms and workflows. This repository is designed to help learners practice real ML pipelines and to demonstrate practical, production‑aware skills to teams.

## Highlights

- **End‑to‑end workflows:** Data ingestion, cleaning, feature engineering, modeling, and evaluation.
- **Breadth of methods:** Linear/Logistic Regression, SVM, Decision Trees/Random Forest, Gradient Boosting, Naive Bayes, PCA, and Clustering.
- **Medical and tabular datasets:** ECG arrhythmia (MIT‑BIH), student performance, and customer segmentation.
- **Robust evaluation:** Proper metrics (ROC‑AUC, F1, precision/recall, silhouette/CH/DB indices), visualizations, and clear comparisons.
- **Reproducibility:** Per‑project `requirements.txt` and structured notebooks.

## Repository Structure

- [clustering/](clustering) — K‑Means, K‑Medoids, and Hierarchical clustering with PCA visualizations and cluster quality metrics.
- [data-preprocessing/](data-preprocessing) — End‑to‑end cleaning and preparation (missing values, encoding, scaling) for the cardio dataset.
- [gaussian-naieve-bayes-and-decision-tree/](gaussian-naieve-bayes-and-decision-tree) — Baseline classifiers with interpretable trees.
- [gradient-boosting-and-random-forrest/](gradient-boosting-and-random-forrest) — Ensemble methods for stronger tabular performance.
- [linear-regression/](linear-regression) — Regression on math and Portuguese datasets; feature effects and evaluation.
- [logistic-regression-and-support-vector-machine/](logistic-regression-and-support-vector-machine) — Binary classification with LR and SVM; scaling and margin insights.
- [principal-component-analysis/](principal-component-analysis) — PCA for dimensionality reduction with clear variance analysis and model impact.

Each folder contains:

- A primary notebook (e.g., [model.ipynb](principal-component-analysis/model.ipynb))
- A focused [README.md](principal-component-analysis/README.md)
- A dedicated [requirements.txt](principal-component-analysis/requirements.txt)
- A local dataset (when applicable)

## Quick Start

1. Ensure Python 3.10+ is installed.
2. Create a virtual environment (recommended):

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

3. Install dependencies for the task you want to run (example for PCA):

```bash
pip install -r principal-component-analysis/requirements.txt
```

4. Open any notebook in VS Code or Jupyter and run cells top‑to‑bottom.

## How to Run Notebooks in VS Code

- Open the folder [Machine-Learning-Tasks](Machine-Learning-Tasks) in VS Code.
- Open a notebook (e.g., [clustering/model.ipynb](clustering/model.ipynb)).
- Select the Python interpreter from your virtual env.
- Run cells sequentially; figures and tables will render inline.

## Datasets

- PCA, Tree‑based, and Ensemble tasks use the ECG **MIT‑BIH Arrhythmia Database** CSVs included under each project (e.g., [principal-component-analysis/Dataset/MIT-BIH Arrhythmia Database.csv](principal-component-analysis/Dataset/MIT-BIH%20Arrhythmia%20Database.csv)).
- Regression tasks use student performance datasets (see [linear-regression/Dataset](linear-regression/Dataset)).
- Clustering uses customer data (see [clustering/Dataset/Customer-Data - 2.csv](clustering/Dataset/Customer-Data%20-%202.csv)).

When a dataset link is provided (e.g., Kaggle), follow its terms; local CSVs are included for convenience.

## What This Demonstrates

- **Practical ML:** Clean, well‑structured notebooks implementing real pipelines end to end.
- **Modeling depth:** Comparative studies across algorithms with appropriate preprocessing.
- **Evaluation rigor:** The right metrics for the task (classification, regression, clustering) and honest trade‑offs.
- **Visualization & communication:** Clear charts (ROC, confusion matrices, variance explained, dendrograms) and narrative.
- **Reproducibility:** Isolated environments with per‑project requirements; consistent results across runs.
- **Domain context:** Work with medical time‑series‑derived tabular data and education/customer datasets.

## Per‑Project Setup Examples

- Clustering:
  - Install: `pip install -r clustering/requirements.txt`
  - Run: open [clustering/model.ipynb](clustering/model.ipynb); explore K‑Means, K‑Medoids (via `sklearn-extra`), and Hierarchical.
- PCA:
  - Install: `pip install -r principal-component-analysis/requirements.txt`
  - Run: open [principal-component-analysis/model.ipynb](principal-component-analysis/model.ipynb); analyze explained variance and model impact.
- Gradient Boosting / Random Forest:
  - Install: `pip install -r gradient-boosting-and-random-forrest/requirements.txt`
  - Run: open [gradient-boosting-and-random-forrest/model.ipynb](gradient-boosting-and-random-forrest/model.ipynb); compare ensembles.

## Skills Covered

- Data wrangling, imputation, encoding, and scaling
- Feature selection and dimensionality reduction (PCA)
- Supervised learning: linear/logistic regression, SVM, trees, ensembles, Naive Bayes
- Unsupervised learning: clustering (K‑Means, K‑Medoids, Hierarchical)
- Model validation: train/test splits, cross‑validation, metric selection
- Visualization: Seaborn/Matplotlib, PCA plots, ROC curves, confusion matrices, dendrograms

## Author & Collaborators

- **Maintainer:** Youssef Hassanien
- **Contributor:** Omar Khaled (e.g., clustering work in [clustering/model.ipynb](clustering/model.ipynb))

## License

See [LICENSE](LICENSE) for terms.

## Roadmap

- Add experiment tracking (e.g., MLflow) and lightweight config files.
- Extend datasets and add more robust feature engineering examples.
- Provide script variants for non‑notebook runs where appropriate.
