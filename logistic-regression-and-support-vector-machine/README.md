# Logistic Regression for ECG Arrhythmia Classification

## Overview

This project implements binary classification models to detect cardiac arrhythmias from ECG (Electrocardiogram) data using the MIT-BIH Arrhythmia Database. The analysis compares Logistic Regression and Support Vector Machine (SVM) models for classifying normal heartbeats versus various types of arrhythmias.

The notebook (`model.ipynb`) provides a complete pipeline from data loading and preprocessing to model training, evaluation, and comparison.

## Dataset

**Source**: MIT-BIH Arrhythmia Database

- **File**: `MIT-BIH Arrhythmia Database.csv`
- **Origin**: PhysioNet's MIT-BIH Arrhythmia Database, a widely-used benchmark for arrhythmia detection
- **Description**: Contains extracted features from ECG signals for heartbeat classification

### Class Distribution (Original):

- **N (Normal)**: Normal heartbeats
- **VEB (Ventricular Ectopic Beat)**: Premature ventricular contractions
- **SVEB (Supraventricular Ectopic Beat)**: Premature atrial contractions
- **F (Fusion)**: Fusion of ventricular and normal beats
- **Q (Unknown)**: Unclassified beats

### Binary Classification Mapping:

- **Class 0 (Normal)**: N beats
- **Class 1 (Arrhythmia)**: VEB, SVEB, F, Q beats combined

### Features:

- 30+ ECG-derived features (morphological, temporal, and statistical measures)
- All features are numerical and ready for machine learning

## Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab
- Required Python packages (see Dependencies section)

## Installation

1. Clone or download this repository
2. Navigate to the `logistic-regression` directory
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dependencies

The following Python packages are required:

- `pandas`: Data manipulation and analysis
- `matplotlib`: Data visualization
- `seaborn`: Statistical data visualization
- `scikit-learn`: Machine learning library (for preprocessing, modeling, and evaluation)

## Project Structure

```
logistic-regression/
├── model.ipynb              # Main analysis notebook
├── README.md                # This file
├── requirements.txt         # Python dependencies
└── Dataset/
    └── MIT-BIH Arrhythmia Database.csv
```

## Usage

1. Ensure the dataset file is in the `Dataset/` directory
2. Open `model.ipynb` in Jupyter Notebook
3. Run the cells sequentially from top to bottom
4. Model training and evaluation results will be displayed

## Analysis Pipeline

### 1. Data Loading

- Load the MIT-BIH Arrhythmia dataset
- Display basic information (shape, head)
- Check for missing values

### 2. Data Preparation

- Separate features (X) and labels (y)
- Features: All columns except the first two (likely identifiers)
- Labels: 'type' column containing arrhythmia classifications

### 3. Exploratory Data Analysis

- Generate boxplots for all numerical features
- Visualize distributions to identify outliers and feature ranges

### 4. Feature Scaling

- Apply StandardScaler for feature normalization
- Visualize distributions before and after scaling
- Ensure features are on comparable scales for model training

### 5. Label Transformation

- Convert multi-class problem to binary classification:
  - Normal (N) → 0
  - Arrhythmias (VEB, SVEB, F, Q) → 1
- Display class distribution after transformation

### 6. Model Training

- Split data: 70% training, 30% testing (random_state=101)
- Train two models:
  - **Logistic Regression**: LogisticRegressionCV with newton-cholesky solver, max_iter=100000
  - **Support Vector Machine**: SVC with polynomial kernel, balanced class weights, C=1.0

### 7. Model Evaluation

- Training accuracy for both models
- Test set predictions and comprehensive evaluation

## Evaluation Metrics

Both models are evaluated using:

- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area Under the Receiver Operating Characteristic curve
- **Confusion Matrix**: Detailed breakdown of predictions

## Models Details

### Logistic Regression

- **Algorithm**: LogisticRegressionCV (cross-validated logistic regression)
- **Solver**: newton-cholesky
- **Max Iterations**: 100,000
- **Regularization**: Built-in cross-validation for optimal C parameter

### Support Vector Machine

- **Kernel**: Polynomial (degree 3)
- **C Parameter**: 1.0
- **Class Weights**: Balanced (handles class imbalance)
- **Probability**: Enabled for AUC calculation
- **Cache Size**: 1000 MB

## Key Features

- **Medical Application**: Real-world ECG arrhythmia detection
- **Binary Classification**: Simplified from multi-class to clinically relevant binary problem
- **Model Comparison**: Direct comparison between probabilistic (Logistic) and boundary-based (SVM) approaches
- **Comprehensive Evaluation**: Multiple metrics for thorough model assessment
- **Visualization**: Boxplots and histograms for data understanding

## Results Interpretation

The notebook provides:

- Training accuracies for model comparison
- Detailed test performance metrics
- Confusion matrices showing prediction breakdowns
- ROC-AUC scores for classification quality assessment

## Clinical Relevance

- **Sensitivity (Recall)**: Critical for medical diagnosis - minimizing false negatives
- **Precision**: Important for reducing false alarms in clinical settings
- **F1-Score**: Balances sensitivity and precision for overall diagnostic performance
- **AUC**: Measures model's ability to distinguish between classes

## Notes

- The dataset combines multiple arrhythmia types into a single "arrhythmia" class for binary classification
- StandardScaler is used to normalize features, important for both Logistic Regression and SVM
- SVM uses polynomial kernel which can capture non-linear relationships in ECG features
- Class balancing is applied to SVM to handle potential class imbalance
- LogisticRegressionCV automatically selects optimal regularization strength

## Limitations

- Binary classification may oversimplify the clinical problem (different arrhythmias have different clinical significance)
- Feature engineering and selection could potentially improve performance
- Hyperparameter tuning beyond default values could optimize results

## Contributing

Potential improvements:

- Implement feature selection techniques
- Try different kernels for SVM (RBF, sigmoid)
- Add cross-validation for more robust evaluation
- Experiment with ensemble methods
- Implement hyperparameter optimization
- Add more advanced metrics (specificity, NPV, PPV)

## License

This project is part of the Machine Learning in Medicine course assignments.

## References

- MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/1.0.0/
- Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia Database. IEEE Engineering in Medicine and Biology Magazine, 20(3), 45-50.
