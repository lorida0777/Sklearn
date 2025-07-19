# Scikit-Learn Machine Learning Tutorial Collection

This repository contains a comprehensive collection of Jupyter notebooks demonstrating various machine learning concepts and techniques using Python's scikit-learn library. The project covers everything from basic supervised learning to advanced pipeline implementations.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Files Description](#files-description)
- [Topics Covered](#topics-covered)
- [Dependencies](#dependencies)
- [Getting Started](#getting-started)
- [Datasets](#datasets)

## 🎯 Project Overview

This collection serves as a practical learning resource for machine learning with scikit-learn, covering fundamental concepts through hands-on implementations. Each notebook focuses on specific aspects of machine learning, from data preprocessing to model evaluation.

## 📁 Files Description

### Jupyter Notebooks

#### `20_PYTHON_SKLEARN_KNN_LinearRegression_et_SUPERVISED_LEARNING.ipynb`
**Purpose**: Introduction to supervised learning with scikit-learn
- **Topics Covered**:
  - Linear Regression implementation
  - Support Vector Regression (SVR)
  - K-Nearest Neighbors (KNN)
  - Basic regression analysis and visualization
  - Model fitting and prediction

#### `21_model_selection.ipynb`
**Purpose**: Model selection techniques and strategies
- **Topics Covered**:
  - Train-test split methodology
  - Iris dataset exploration
  - Data visualization for classification
  - Model comparison techniques
  - Performance evaluation strategies

#### `22_cross_validation_sklearn.ipynb`
**Purpose**: Cross-validation techniques for robust model evaluation
- **Topics Covered**:
  - K-Fold cross-validation
  - LeaveOneOut cross-validation
  - Cross-validation scoring with scikit-learn
  - Model performance assessment
  - Validation strategies for different scenarios

#### `22_data_pre_processing.ipynb`
**Purpose**: Comprehensive data preprocessing techniques
- **Topics Covered**:
  - LabelEncoder for categorical data
  - OrdinalEncoder for ordinal data
  - Data transformation techniques
  - Handling different data types
  - Preprocessing pipelines

#### `23_feature_selection.ipynb`
**Purpose**: Feature selection methods and dimensionality reduction
- **Topics Covered**:
  - VarianceThreshold for feature filtering
  - SelectKBest for statistical feature selection
  - Chi-square test for feature selection
  - Feature importance analysis
  - Dimensionality reduction techniques

#### `imputer_netoyage_de_donnees.ipynb`
**Purpose**: Data cleaning and missing value imputation (French: "Data Cleaning Imputer")
- **Topics Covered**:
  - SimpleImputer for handling missing values
  - KNNImputer for advanced imputation
  - Different imputation strategies (mean, median, mode)
  - Handling NaN values in datasets
  - Data quality improvement techniques

#### `metriques_de_regression.ipynb`
**Purpose**: Regression evaluation metrics (French: "Regression Metrics")
- **Topics Covered**:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Median Absolute Error
  - Model performance evaluation
  - Regression analysis with Boston housing data

#### `pipeline_avancee.ipynb`
**Purpose**: Advanced pipeline creation and management (French: "Advanced Pipeline")
- **Topics Covered**:
  - Pipeline creation with make_pipeline
  - Column transformers
  - StandardScaler integration
  - SGDClassifier implementation
  - End-to-end machine learning workflows
  - Titanic dataset analysis

### Datasets

#### `boston_house_prices.csv` (34.7 KB)
**Description**: Boston Housing Dataset
- **Content**: Housing data with 13 features and target prices
- **Features**: CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV
- **Use Case**: Regression analysis and price prediction
- **Format**: CSV with 506 samples

#### `titanic3.xls` (278 KB)
**Description**: Titanic Passenger Dataset
- **Content**: Passenger information and survival data
- **Use Case**: Classification problems and survival prediction
- **Format**: Excel spreadsheet
- **Features**: Passenger demographics, ticket information, survival status

## 🎓 Topics Covered

### 1. **Supervised Learning**
- Linear Regression
- Support Vector Machines
- K-Nearest Neighbors
- Classification and Regression

### 2. **Data Preprocessing**
- Encoding categorical variables
- Handling missing values
- Data transformation
- Feature scaling

### 3. **Model Selection & Validation**
- Train-test splitting
- Cross-validation techniques
- Model comparison
- Performance metrics

### 4. **Feature Engineering**
- Feature selection methods
- Dimensionality reduction
- Statistical feature selection
- Variance-based filtering

### 5. **Pipeline Development**
- Creating ML pipelines
- Column transformers
- Automated workflows
- Integration of preprocessing and modeling

### 6. **Model Evaluation**
- Regression metrics
- Classification metrics
- Performance visualization
- Error analysis

## 🛠️ Dependencies

The notebooks require the following Python libraries:

```python
- numpy
- matplotlib
- pandas
- seaborn
- scikit-learn
- jupyter
```

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd sklearn-tutorial
   ```

2. **Install dependencies**:
   ```bash
   pip install numpy matplotlib pandas seaborn scikit-learn jupyter
   ```

3. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

4. **Recommended Learning Order**:
   1. Start with `20_PYTHON_SKLEARN_KNN_LinearRegression_et_SUPERVISED_LEARNING.ipynb`
   2. Continue with `imputer_netoyage_de_donnees.ipynb`
   3. Progress through `22_data_pre_processing.ipynb`
   4. Learn model selection with `21_model_selection.ipynb`
   5. Master validation with `22_cross_validation_sklearn.ipynb`
   6. Explore feature selection in `23_feature_selection.ipynb`
   7. Study metrics with `metriques_de_regression.ipynb`
   8. Complete with `pipeline_avancee.ipynb`

## 📊 Datasets Information

Both datasets are included for practical learning:
- **Boston Housing**: Perfect for regression tasks and understanding feature impact on house prices
- **Titanic**: Ideal for classification problems and survival analysis

## 🎯 Learning Objectives

By working through these notebooks, you will:
- Master scikit-learn's core functionality
- Understand the complete machine learning workflow
- Learn best practices for data preprocessing
- Develop skills in model selection and evaluation
- Create robust ML pipelines
- Apply various algorithms to real-world datasets

## 📝 Notes

- Some notebooks contain French comments and titles, reflecting the educational context
- Each notebook is self-contained with clear examples
- Code is well-commented for educational purposes
- Practical examples use real datasets for hands-on learning

---

*This collection provides a comprehensive introduction to machine learning with scikit-learn, suitable for beginners to intermediate practitioners.*
