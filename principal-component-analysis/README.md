# Principal Component Analysis (PCA) for ECG Arrhythmia Classification

This project demonstrates the application of Principal Component Analysis (PCA) for dimensionality reduction in the context of ECG arrhythmia classification using the MIT-BIH Arrhythmia Database. The analysis compares the performance of a Random Forest classifier with and without PCA to evaluate the effectiveness of dimensionality reduction in maintaining classification accuracy while potentially improving computational efficiency.

## Dataset

The project uses the **MIT-BIH Arrhythmia Database** from Kaggle (https://www.kaggle.com/datasets/sadmansakib7/ecg-arrhythmia-classification-dataset/data). This dataset contains ECG signals classified into different arrhythmia types.

- **Features**: 30 numerical features representing ECG signal characteristics
- **Target**: Multi-class labels converted to binary classification (Normal vs. Arrhythmia)
- **Classes**: 
  - 0: Normal (N)
  - 1: Arrhythmia (VEB, SVEB, F, Q)

## Project Structure

```
principle-component-analysis/
├── model.ipynb              # Main Jupyter notebook with complete analysis
├── README.md                # This file
├── requirements.txt         # Python dependencies
└── Dataset/
    └── MIT-BIH Arrhythmia Database.csv  # ECG arrhythmia dataset
```

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning algorithms and preprocessing

## Usage

1. Open the `model.ipynb` notebook in Jupyter Lab/Notebook
2. Run the cells sequentially to execute the complete analysis
3. The notebook includes:
   - Data loading and exploration
   - Preprocessing and normalization
   - PCA implementation and parameter experimentation
   - Model training and evaluation
   - Comprehensive visualizations

## Methodology

### 1. Data Preprocessing
- **Missing Values**: Check for and handle any null values
- **Feature Selection**: Extract numerical features (columns 2 onwards)
- **Normalization**: Apply StandardScaler to standardize features
- **Label Transformation**: Convert multi-class labels to binary classification

### 2. Principal Component Analysis (PCA)
- **Dimensionality Reduction**: Reduce 30 features to principal components
- **Variance Analysis**: Analyze explained variance by each component
- **Parameter Experiments**:
  - Different numbers of components (5, 10, 15, 20, 25)
  - SVD solver options ('auto', 'full', 'randomized')
  - Whitening parameter (True/False)

### 3. Model Training
- **Algorithm**: Random Forest Classifier
- **Comparison**: Train models with and without PCA
- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - AUC-ROC
  - Training and prediction time

### 4. Visualization
- **Data Distribution**: Box plots and histograms before/after scaling
- **PCA Results**: Variance explained plots, 2D and 3D scatter plots
- **Model Comparison**: Bar charts comparing metrics
- **ROC Curves**: Performance visualization
- **Confusion Matrices**: Classification results

## Key Findings

### Performance Comparison

Based on the test metrics, PCA actually **decreases** classification performance compared to the original features:

| Metric          | With PCA | Without PCA | Difference |
| --------------- | -------- | ----------- | ---------- |
| Accuracy        | 0.973    | 0.990       | -0.017     |
| Precision       | 0.953    | 0.976       | -0.023     |
| Recall          | 0.774    | 0.922       | -0.148     |
| F1-Score        | 0.854    | 0.948       | -0.094     |
| AUC-ROC         | 0.987    | 0.998       | -0.011     |
| Prediction Time | 0.258s   | 0.234s      | +0.024s    |

### PCA Parameter Insights

- **n_components**: Even with optimal component selection, PCA cannot match the performance of original features
- **svd_solver**: Different solvers show minimal impact on the overall poorer performance
- **whiten**: Whitening does not compensate for the information loss in dimensionality reduction

### Visualization Insights

- 2D and 3D PCA plots show some class separation but lose important discriminatory information
- Variance explained analysis shows that even high-variance components cannot capture the complex decision boundaries that Random Forest uses effectively

## Results Summary

**Contrary to expectations, PCA significantly degrades the performance of Random Forest classifier on this ECG arrhythmia dataset.** The original 30 features provide superior classification accuracy compared to any PCA-reduced representation. This demonstrates that Random Forest and decision tree-based methods generally do not benefit from PCA preprocessing, as they can effectively handle high-dimensional data and the linear dimensionality reduction of PCA may remove non-linear relationships crucial for accurate classification.

The findings highlight the importance of evaluating dimensionality reduction techniques empirically rather than assuming universal benefits, particularly for ensemble methods like Random Forest that are robust to the curse of dimensionality.

## Contributing

This is an educational project for the Machine Learning in Medicine course. For improvements or questions, please refer to the course materials or contact the project maintainer.

## License

This project is part of the Machine Learning in Medicine course assignments.

## References

- MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/1.0.0/
- Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia Database. IEEE Engineering in Medicine and Biology Magazine, 20(3), 45-50.