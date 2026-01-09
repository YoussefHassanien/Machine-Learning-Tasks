# Customer Segmentation using Clustering Algorithms

This project demonstrates comprehensive customer segmentation analysis using multiple clustering algorithms including K-Means, K-Medoids, and Hierarchical Clustering. The analysis explores various parameter configurations and compares the performance of different clustering approaches on customer data to identify optimal segmentation strategies.

**Author**: Omar Khaled  
**Note**: This notebook was created by Omar Khaled, not Youssef Hassanien.

## Dataset

The project uses customer data for segmentation analysis.

- **Source**: Customer-Data - 2.csv
- **Features**: Customer behavioral and demographic attributes (excluding customer ID)
- **Preprocessing**: Standard scaling and missing value imputation
- **Target**: Unsupervised clustering to identify customer segments

## Project Structure

```
clustering/
├── model.ipynb              # Main Jupyter notebook with complete clustering analysis
├── README.md                # This file
├── requirements.txt         # Python dependencies
└── Dataset/
    └── Customer-Data - 2.csv  # Customer dataset for segmentation
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
- **sklearn-extra**: Additional clustering algorithms (K-Medoids)
- **scipy**: Scientific computing (hierarchical clustering)

## Usage

1. Open the `model.ipynb` notebook in Jupyter Lab/Notebook
2. Run the cells sequentially to execute the complete clustering analysis
3. The notebook includes:
   - Data loading and preprocessing
   - Parameter experimentation for each algorithm
   - Performance evaluation and visualization
   - Comparative analysis of all methods

## Methodology

### 1. Data Preprocessing

- **Data Loading**: Import customer dataset
- **Missing Values**: Imputation using median values
- **Feature Selection**: Remove customer ID, use behavioral features
- **Scaling**: StandardScaler for feature normalization

### 2. K-Means Clustering

**Parameters Explored**:

- `n_clusters`: 2, 3, 4, 5, 6, 8 clusters
- `init`: 'k-means++', 'random' initialization methods
- `n_init`: 1, 5, 10, 20, 50 initialization attempts
- `max_iter`: 10, 50, 100, 300, 500, 1000 maximum iterations

### 3. K-Medoids Clustering

**Parameters Explored**:

- `n_clusters`: 2, 3, 4, 5, 6 clusters
- `metric`: 'euclidean', 'manhattan', 'cosine' distance metrics
- `init`: 'k-medoids++', 'random', 'heuristic' initialization methods

### 4. Hierarchical Clustering

**Parameters Explored**:

- `linkage`: 'ward', 'complete', 'average', 'single' linkage methods
- `n_clusters`: 2, 3, 4, 5, 6, 8 clusters
- **Dendrogram Analysis**: Visual inspection of cluster hierarchy

### 5. Performance Evaluation

**Metrics Used**:

- **Silhouette Score**: Measures cluster cohesion and separation (-1 to 1, higher is better)
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster variance (higher is better)
- **Davies-Bouldin Index**: Average similarity between clusters (lower is better)

### 6. Visualization

- **2D Scatter Plots**: PCA-reduced visualization of clusters
- **Dendrograms**: Hierarchical clustering tree structures
- **Performance Comparisons**: Bar charts comparing algorithm metrics
- **Parameter Effects**: Analysis of how parameters affect clustering quality

## Key Findings

### K-Means Clustering

- **n_clusters**: Optimal number determined by silhouette analysis and elbow method
- **Initialization**: 'k-means++' provides more stable and better results than random initialization
- **n_init**: Higher values (10-20) improve stability with minimal performance cost
- **Convergence**: Algorithm typically converges well before maximum iterations

### K-Medoids Clustering

- **Robustness**: More robust to outliers than K-Means due to medoid-based centers
- **Distance Metrics**: Euclidean distance generally performs best for this dataset
- **Initialization**: 'k-medoids++' provides good initial medoid selection
- **Computational Cost**: More expensive than K-Means but better outlier handling

### Hierarchical Clustering

- **Ward Linkage**: Produces balanced, compact clusters (generally best performance)
- **Complete Linkage**: Creates tight clusters, sensitive to outliers
- **Average Linkage**: Balanced approach between Ward and Single
- **Single Linkage**: Can create chain-like structures, useful for elongated clusters
- **Dendrograms**: Provide intuitive visualization of cluster relationships

### Algorithm Comparison

- **Performance**: Hierarchical with Ward linkage often shows best silhouette scores
- **Interpretability**: Hierarchical methods provide cluster hierarchy information
- **Scalability**: K-Means most efficient for large datasets
- **Robustness**: K-Medoids best for datasets with outliers

## Results Summary

The analysis demonstrates that different clustering algorithms perform variably depending on data characteristics and evaluation metrics. For customer segmentation:

- **Best Overall**: Hierarchical clustering with Ward linkage typically shows superior silhouette scores
- **Most Robust**: K-Medoids provides better outlier handling
- **Most Efficient**: K-Means offers best computational performance
- **Most Interpretable**: Hierarchical methods provide dendrogram-based insights

The choice of algorithm depends on specific business requirements, data characteristics, and computational constraints. Parameter tuning significantly impacts clustering quality, making systematic experimentation essential for optimal customer segmentation.

## Author

**Omar Khaled** - Creator of this clustering analysis notebook.  
_Note: This work was developed by Omar Khaled as part of the Machine Learning in Medicine course assignments._
