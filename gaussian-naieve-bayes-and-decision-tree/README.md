# Gaussian Naive Bayes and Decision Tree for ECG Arrhythmia Classification

## Overview

This project implements probabilistic and tree-based classification models for detecting cardiac arrhythmias from ECG (Electrocardiogram) data using the MIT-BIH Arrhythmia Database. The analysis compares Gaussian Naive Bayes and Decision Tree classifiers for binary classification of normal heartbeats versus arrhythmias.

The notebook (`model.ipynb`) provides a complete machine learning pipeline from data preprocessing to model training, evaluation, and comparison, with detailed tree metadata for the Decision Tree model.

## Dataset

**Source**: MIT-BIH Arrhythmia Database

- **File**: `MIT-BIH Arrhythmia Database.csv`
- **Origin**: PhysioNet's MIT-BIH Arrhythmia Database, a benchmark dataset for arrhythmia detection
- **Description**: Contains extracted morphological and statistical features from ECG signals

### Class Distribution (Original):

- **N (Normal)**: Normal sinus rhythm beats
- **VEB (Ventricular Ectopic Beat)**: Premature ventricular contractions
- **SVEB (Supraventricular Ectopic Beat)**: Premature atrial contractions
- **F (Fusion)**: Fusion beats of ventricular and normal complexes
- **Q (Unknown)**: Unclassified or questionable beats

### Binary Classification Mapping:

- **Class 0 (Normal)**: N beats
- **Class 1 (Arrhythmia)**: VEB, SVEB, F, Q beats combined

### Features:

- 30+ ECG-derived features including:
  - Morphological features (amplitude, duration, shape characteristics)
  - Temporal features (RR intervals, timing measurements)
  - Statistical features (mean, variance, skewness, etc.)
- All features are numerical and pre-extracted for machine learning

## Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab
- Required Python packages (see Dependencies section)

## Installation

1. Clone or download this repository
2. Navigate to the `gaussian-naieve-bayes-and-decision-tree` directory
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
gaussian-naieve-bayes-and-decision-tree/
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
4. Model training results and tree metadata will be displayed

## Analysis Pipeline

### 1. Data Loading

- Load the MIT-BIH Arrhythmia dataset
- Display dataset dimensions and preview
- Check for missing values

### 2. Data Preparation

- Separate features (X) and target labels (y)
- Features: All numerical columns except identifiers
- Target: 'type' column with arrhythmia classifications

### 3. Exploratory Data Analysis

- Generate boxplots for all feature distributions
- Identify potential outliers and feature ranges
- Assess data quality and preprocessing needs

### 4. Feature Scaling

- Apply StandardScaler for z-score normalization
- Visualize feature distributions before and after scaling
- Ensure features have zero mean and unit variance

### 5. Label Transformation

- Convert multi-class to binary classification:
  - Normal (N) → 0
  - Arrhythmias (VEB, SVEB, F, Q) → 1
- Display transformed class distribution

### 6. Model Training

- Split data: 70% training, 30% testing (random_state=101)
- Train two models:
  - **Gaussian Naive Bayes**: Probabilistic classifier assuming feature independence
  - **Decision Tree**: Tree-based classifier with optimized hyperparameters

### 7. Model Evaluation

- Training accuracies for both models
- Comprehensive test set evaluation metrics

## Models Details

### Gaussian Naive Bayes

- **Algorithm**: GaussianNB from scikit-learn
- **Assumptions**: Features are independent, normally distributed given the class
- **Advantages**: Fast training, works well with small datasets, probabilistic outputs
- **Limitations**: Independence assumption may not hold for correlated ECG features

### Decision Tree

- **Algorithm**: DecisionTreeClassifier with optimized parameters
- **Criterion**: Entropy (information gain)
- **Max Depth**: 7 (prevents overfitting)
- **Min Samples Split**: 4
- **Min Samples Leaf**: 2
- **Class Weights**: Balanced (1:1 ratio)
- **Max Features**: None (uses all features)
- **Random State**: 42 for reproducibility

## Evaluation Metrics

Both models are evaluated using:

- **Accuracy**: Overall correct classification rate
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives) - Critical for medical diagnosis
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area Under the ROC curve - Measures discriminative ability
- **Confusion Matrix**: Detailed prediction breakdown

## Decision Tree Metadata

The notebook provides comprehensive tree information:

- **Tree Depth**: Actual depth of the trained tree
- **Number of Leaves**: Terminal nodes in the tree
- **Number of Classes**: Output classes (2 for binary classification)
- **Max Features**: Features considered for best splits
- **Feature Names**: Names of input features
- **Number of Outputs**: Number of output targets

## Key Features

- **Medical Application**: ECG arrhythmia detection with clinical relevance
- **Model Diversity**: Comparison between probabilistic (Naive Bayes) and rule-based (Decision Tree) approaches
- **Optimized Decision Tree**: Hyperparameter tuning for medical classification
- **Comprehensive Evaluation**: Multiple metrics for thorough model assessment
- **Tree Interpretability**: Decision Tree provides explainable rules for predictions

## Results Interpretation

The notebook outputs:

- Training performance comparison
- Detailed test metrics for both models
- Confusion matrices showing prediction patterns
- Decision Tree structural information
- ROC-AUC scores for classification quality

## Clinical Relevance

- **High Recall**: Critical for arrhythmia detection to minimize missed cases
- **Balanced Precision**: Important to reduce false alarms in clinical monitoring
- **F1-Score**: Balances sensitivity and precision for diagnostic performance
- **Tree Interpretability**: Decision rules can provide clinical insights

## Notes

- The dataset combines multiple arrhythmia types for binary classification
- StandardScaler is crucial for Gaussian Naive Bayes (assumes normal distribution)
- Decision Tree parameters were optimized to minimize false negatives (comment in code)
- Class weights are balanced despite the optimization comment showing 1:1
- Tree depth of 7 provides good balance between complexity and generalization

## Model Comparison Insights

- **Gaussian Naive Bayes**: Fast, probabilistic, works well when independence assumption holds approximately
- **Decision Tree**: Interpretable, handles non-linear relationships, can overfit without proper tuning
- **Medical Context**: Decision Tree's interpretability may be preferred for clinical decision support

## Limitations

- Binary classification oversimplifies arrhythmia types (different arrhythmias have different clinical implications)
- Naive Bayes independence assumption may not perfectly hold for ECG features
- Decision Tree may not generalize well to new ECG morphologies
- Feature engineering could potentially improve both models

## Contributing

Potential improvements:

- Implement feature selection to reduce dimensionality
- Try other tree algorithms (Random Forest, Gradient Boosting)
- Experiment with different Naive Bayes variants
- Add cross-validation for more robust evaluation
- Implement cost-sensitive learning for medical applications
- Add feature importance analysis for Decision Tree

## License

This project is part of the Machine Learning in Medicine course assignments.

## References

- MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/1.0.0/
- Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia Database. IEEE Engineering in Medicine and Biology Magazine, 20(3), 45-50.
- Scikit-learn Documentation: Gaussian Naive Bayes and Decision Tree Classifiers
