# Data Preprocessing for Cardiovascular Disease Dataset

## Overview

This Jupyter notebook (`data-preprocessing.ipynb`) provides a comprehensive data preprocessing pipeline for the Cardiovascular Disease dataset. The preprocessing steps include data loading, missing value handling, feature engineering, outlier detection and treatment, categorical encoding, feature normalization, and data splitting for machine learning tasks.

The notebook transforms raw cardiovascular health data into a clean, normalized dataset suitable for training machine learning models to predict cardiovascular disease risk.

## Dataset

**Input Dataset:** `cardio_train.csv`

- Source: Cardiovascular disease prediction dataset
- Format: CSV with semicolon (;) delimiter
- Contains patient health metrics and cardiovascular disease indicators

**Key Features:**

- `id`: Patient identifier
- `age`: Age in days
- `gender`: Gender (1: female, 2: male)
- `height`: Height in cm
- `weight`: Weight in kg
- `ap_hi`: Systolic blood pressure
- `ap_lo`: Diastolic blood pressure
- `cholesterol`: Cholesterol level
- `gluc`: Glucose level
- `smoke`: Smoking status
- `alco`: Alcohol intake
- `active`: Physical activity
- `cardio`: Presence of cardiovascular disease (target variable)

## Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab
- Required Python packages (see Dependencies section)

## Installation

1. Clone or download this repository
2. Navigate to the `data-preprocessing` directory
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
- `scikit-learn`: Machine learning library (for MinMaxScaler and train_test_split)

## Usage

1. Ensure `cardio_train.csv` is in the same directory as the notebook
2. Open `data-preprocessing.ipynb` in Jupyter Notebook
3. Run the cells sequentially from top to bottom
4. The processed data will be saved as `cardio_train_processed.csv`

## Preprocessing Steps

### 1. Imports and Data Loading

- Import necessary libraries
- Load the dataset using pandas
- Create a copy for preprocessing

### 2. Missing Values Handling

- Check for missing values in each column
- Report the count of missing values per column

### 3. Feature Engineering

- Convert age from days to years (divide by 365.25 and round to 2 decimal places)

### 4. Outlier Detection and Handling

- Identify numerical columns: `age`, `height`, `weight`, `ap_hi`, `ap_lo`
- Use Interquartile Range (IQR) method to detect outliers
- Visualize distributions before outlier handling using boxplots
- Cap outliers by replacing values beyond 1.5\*IQR bounds with the boundary values
- Visualize distributions after outlier handling

### 5. Categorical Encoding

- Convert `gender` column to one-hot encoded dummy variables
- Create `is_female` and `is_male` columns
- Drop the original `gender` column

### 6. Feature Normalization

- Select features to scale: `age`, `height`, `weight`, `ap_hi`, `ap_lo`
- Visualize distributions before scaling using histograms
- Apply Min-Max scaling to normalize features to [0, 1] range
- Visualize distributions after scaling

### 7. Data Export

- Save the fully preprocessed dataset to `cardio_train_processed.csv`

### 8. Data Splitting

- Reload the processed data
- Prepare features (X) and target (Y) - Note: The current implementation uses `id` as target, which may need adjustment for actual modeling
- Split data into training (80%) and testing (20%) sets using stratified sampling
- Display shapes of resulting datasets

## Output

**Processed Dataset:** `cardio_train_processed.csv`

- Contains all original features plus engineered ones
- Age converted to years
- Outliers handled
- Categorical variables encoded
- Numerical features normalized
- Ready for machine learning model training

**Split Datasets:**

- `x_train`, `y_train`: Training features and labels
- `x_test`, `y_test`: Testing features and labels

## Visualization

The notebook includes several visualizations:

- Boxplots of numerical features before and after outlier handling
- Histograms of features before and after normalization

## Notes

- The current data splitting uses `id` as the target variable, which is likely incorrect for cardiovascular disease prediction. Adjust the target variable selection based on your modeling requirements.
- Outlier handling uses capping (Winsorization) rather than removal to preserve data points.
- Min-Max scaling is used for normalization, ensuring all features are on the same scale.

## Contributing

Feel free to modify and improve the preprocessing pipeline. Ensure that any changes maintain data integrity and follow best practices for data preprocessing.

## License

This project is part of the Machine Learning in Medicine course assignments.
