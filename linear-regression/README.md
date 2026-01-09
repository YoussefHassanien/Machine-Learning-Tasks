# Linear Regression Models for Student Performance Prediction

## Overview

This project implements linear regression models to predict student final grades (G3) in Mathematics and Portuguese language courses. The analysis includes comprehensive data preprocessing, model training, evaluation, and comparison between Ordinary Least Squares (OLS) and Lasso regression techniques.

Two separate Jupyter notebooks are provided:

- `math_model.ipynb`: Linear regression for Mathematics course grades
- `portuguese_model.ipynb`: Linear regression for Portuguese language course grades

## Dataset

The datasets used are from the UCI Machine Learning Repository - Student Performance datasets.

### Files:

- `student-mat.csv`: Mathematics course data
- `student-por.csv`: Portuguese language course data

### Features (33 attributes):

- **Demographic**: school, sex, age, address, famsize, Pstatus
- **Family Background**: Medu, Fedu, Mjob, Fjob, guardian
- **School Related**: reason, traveltime, studytime, failures, schoolsup, famsup, paid, activities, nursery, higher, internet, absences
- **Personal**: famrel, freetime, goout, Dalc, Walc, health, romantic
- **Academic**: G1 (first period grade), G2 (second period grade)
- **Target**: G3 (final grade, 0-20 scale)

All categorical variables are encoded, and numerical features are normalized for modeling.

## Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab
- Required Python packages (see Dependencies section)

## Installation

1. Clone or download this repository
2. Navigate to the `linear-regression` directory
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dependencies

The following Python packages are required:

- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `matplotlib`: Data visualization
- `seaborn`: Statistical data visualization
- `scikit-learn`: Machine learning library (for preprocessing, modeling, and evaluation)

## Project Structure

```
linear-regression/
├── math_model.ipynb          # Mathematics course model
├── portuguese_model.ipynb    # Portuguese course model
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── Dataset/
    ├── metadata.txt          # Dataset description
    ├── Original/             # Raw datasets
    │   ├── student-mat.csv
    │   └── student-por.csv
    └── Preprocessed/         # Processed datasets
        ├── student-mat.csv
        └── student-por.csv
```

## Usage

1. Ensure the dataset files are in the `Dataset/Original/` directory
2. Open either `math_model.ipynb` or `portuguese_model.ipynb` in Jupyter Notebook
3. Run the cells sequentially from top to bottom
4. The preprocessed data will be saved in `Dataset/Preprocessed/`
5. Model results and evaluations will be displayed in the output

## Data Preprocessing Pipeline

### 1. Data Loading

- Load the respective dataset (Math or Portuguese)
- Create a working copy for preprocessing

### 2. Missing Values Check

- Verify no missing values exist in the datasets

### 3. Outlier Visualization

- Generate boxplots for all numerical features
- Identify potential outliers in distributions

### 4. Categorical Encoding

- Use Label Encoding for categorical variables:
  - Binary categories: sex, school, address, famsize, Pstatus, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic
  - Nominal categories: Mjob, Fjob, reason, guardian
- Transform categorical strings to numerical values

### 5. Feature Normalization

- Apply Min-Max scaling to all numerical features (excluding target G3)
- Visualize distributions before and after scaling
- Ensure all features are on the same scale [0,1]

### 6. Data Export

- Save preprocessed datasets to `Dataset/Preprocessed/`

### 7. Train-Test Split

- Split data into training (80%) and testing (20%) sets
- Use stratified random sampling with random_state=42 for reproducibility

## Modeling Approach

### Models Implemented

1. **Ordinary Least Squares (OLS) Regression**

   - Standard linear regression without regularization
   - Fits all features to minimize residual sum of squares

2. **Lasso Regression**
   - Linear regression with L1 regularization
   - Alpha = 0.01, max_iter = 2000
   - Performs automatic feature selection by shrinking coefficients

### Evaluation Metrics

- **R² Score**: Proportion of variance explained (higher is better)
- **Mean Absolute Error (MAE)**: Average absolute prediction error
- **Root Mean Square Error (RMSE)**: Square root of average squared prediction error

### Validation Techniques

1. **Holdout Validation**

   - Single train-test split evaluation
   - Assesses model performance on unseen data

2. **Cross-Validation**
   - 5-fold cross-validation
   - Provides robust performance estimates
   - Checks for model stability across different data splits

### Overfitting/Underfitting Analysis

- Compare training vs test R² scores
- Identify overfitting (high training, low test scores)
- Identify underfitting (both scores low)
- Assess model generalization capability

### Feature Importance

- Extract and rank feature coefficients from OLS model
- Identify most influential predictors for final grades

## Results Summary

Both notebooks follow identical structure and provide:

- Preprocessing visualizations (boxplots, histograms)
- Model training and prediction
- Comprehensive evaluation metrics
- Overfitting/underfitting diagnostics
- Cross-validation results
- Feature importance ranking

## Key Findings

- Compare model performance between Math and Portuguese courses
- Identify which features most influence final grades
- Assess regularization impact (Lasso vs OLS)
- Evaluate model stability through cross-validation

## Notes

- The datasets contain no missing values, simplifying preprocessing
- Categorical encoding uses Label Encoding (ordinal for some nominal variables)
- Min-Max scaling preserves relationships while normalizing ranges
- Lasso regularization helps prevent overfitting and performs feature selection
- Cross-validation provides more reliable performance estimates than single split

## Contributing

Feel free to modify and improve the models. Potential enhancements:

- Try different encoding methods (One-Hot Encoding)
- Experiment with other regularization techniques (Ridge, Elastic Net)
- Add feature engineering or selection methods
- Implement hyperparameter tuning
- Compare with other regression algorithms

## License

This project is part of the Machine Learning in Medicine course assignments.

## References

- UCI Machine Learning Repository: Student Performance Dataset
- Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance.
