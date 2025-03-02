# Fish Toxicity Prediction Using Machine Learning

## Overview
This project aims to predict fish toxicity (LC50 values) based on chemical descriptors using various machine learning models. The dataset used contains chemical descriptors such as `CIC0`, `SM1_Dz(Z)`, `GATS1i`, `NdsCH`, `NdssC`, and `MLOGP`, with the target variable being `LC50 [-LOG(mol/L)]`. The project involves data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and comparison of regression and classification models. Additionally, clustering techniques are applied to explore the underlying structure of the data.

---

## Key Features
1. **Data Preprocessing:**
   - Handling missing values by filling them with the median.
   - Removing outliers using the Interquartile Range (IQR) method.
   - Feature scaling using standardization.
   - Binning the target variable (`LC50`) for classification tasks.

2. **Exploratory Data Analysis (EDA):**
   - Descriptive statistics and correlation analysis.
   - Visualization of data distributions, correlations, and missing values.
   - Pair plots and heatmaps for feature relationships.

3. **Feature Engineering:**
   - Polynomial feature transformation for regression models.
   - Principal Component Analysis (PCA) for dimensionality reduction.

4. **Model Implementation:**
   - **Regression Models:**
     - Linear Regression
     - Ridge Regression
     - Lasso Regression
     - ElasticNet Regression
     - Polynomial Regression
     - Support Vector Regression (SVR)
     - Decision Tree Regression
     - Random Forest Regression
     - XGBoost Regression
   - **Classification Models:**
     - Logistic Regression
     - K-Nearest Neighbors (KNN)
     - Support Vector Machine (SVM)
     - Decision Tree
     - Random Forest
     - Gradient Boosting
     - XGBoost
   - **Clustering Models:**
     - KMeans
     - DBSCAN
     - Birch
     - Affinity Propagation
     - Gaussian Mixture Model (GMM)

5. **Model Evaluation:**
   - Regression metrics: R2 Score, Adjusted R2 Score, RMSE, MSE, MAE.
   - Classification metrics: Accuracy, Precision, Recall, F1 Score, Confusion Matrix.
   - Clustering metrics: Silhouette Score, Calinski-Harabasz Score, Davies-Bouldin Score.

6. **Visualization:**
   - ROC-AUC curves for classification models.
   - Confusion matrices for classification models.
   - Feature importance plots for tree-based models.
   - Clustering evaluation metrics comparison.

---

## Results

### **Regression Models:**
| Model                  | R2 Score   | Adjusted R2 Score | RMSE      | MSE       | MAE       |
|------------------------|------------|-------------------|-----------|-----------|-----------|
| Linear Regression      | 0.6807     | 0.6737            | 0.7016    | 0.4923    | 0.5134    |
| Ridge Regression       | 0.6807     | 0.6737            | 0.7016    | 0.4923    | 0.5134    |
| Lasso Regression       | 0.6066     | 0.5980            | 0.7787    | 0.6065    | 0.5877    |
| ElasticNet Regression  | 0.6066     | 0.5980            | 0.7787    | 0.6065    | 0.5877    |
| Polynomial Regression  | -311.0999  | 315.8138          | 17.0353   | 290.2025  | 3.4943    |
| Support Vector Regressor (SVR) | 0.7250     | 0.7189            | 0.6512    | 0.4240    | 0.4823    |
| Decision Tree Regression | 0.3061     | 0.2908            | 1.0344    | 1.0699    | 0.7527    |
| Random Forest Regression | 0.7136     | 0.7073            | 0.6645    | 0.4415    | 0.4975    |
| XGBoost Regression     | 0.6500     | 0.6400            | 0.7400    | 0.5400    | 0.5400    |

### **Classification Models:**
| Model                  | Accuracy   | Precision  | Recall     | F1 Score   |
|------------------------|------------|------------|------------|------------|
| Logistic Regression    | 0.8584     | 0.8556     | 0.8584     | 0.8558     |
| K-Nearest Neighbors (KNN) | 0.7554     | 0.7563     | 0.7554     | 0.7529     |
| Support Vector Machine (SVM) | 0.9099     | 0.9122     | 0.9099     | 0.9095     |
| Decision Tree          | 0.9914     | 0.9918     | 0.9914     | 0.9915     |
| Random Forest          | 0.9785     | 0.9793     | 0.9785     | 0.9780     |
| Gradient Boosting      | 0.9914     | 0.9918     | 0.9914     | 0.9915     |
| XGBoost                | 0.9914     | 0.9916     | 0.9914     | 0.9914     |

### **Clustering Models:**
| Model                  | Silhouette Score | Calinski-Harabasz Score | Davies-Bouldin Score |
|------------------------|------------------|-------------------------|----------------------|
| KMeans                 | 0.267            | 401.047                 | 1.246                |
| DBSCAN                 | -0.285           | 7.239                   | 2.110                |
| Birch                  | 0.249            | 307.846                 | 1.227                |
| Affinity Propagation   | 0.261            | 172.744                 | 1.071                |
| Gaussian Mixture Model (GMM) | 0.121            | 55.837                  | 2.672                |

---

## Key Insights

### **Regression Models:**
- **Best Performers:** Support Vector Regressor (SVR) and Random Forest Regression achieved the highest R2 scores (0.725 and 0.7136, respectively) and the lowest RMSE values.
- **Worst Performer:** Polynomial Regression performed poorly, with a negative R2 score, indicating overfitting.

### **Classification Models:**
- **Best Performers:** Decision Tree, Random Forest, Gradient Boosting, and XGBoost achieved near-perfect accuracy (0.9914) and F1 scores (0.9915).
- **Moderate Performers:** Logistic Regression and SVM performed well but were slightly less accurate than tree-based models.
- **Worst Performer:** K-Nearest Neighbors (KNN) had the lowest accuracy (0.7554) and F1 score (0.7529).

### **Clustering Models:**
- **Best Performer:** KMeans achieved the highest Calinski-Harabasz score (401.047) and a reasonable Silhouette score (0.267), indicating relatively good cluster separation.
- **Worst Performer:** DBSCAN had a negative Silhouette score (-0.285), suggesting poor clustering performance.

---

## How to Run the Code
1. **Prerequisites:**
   - Python 3.x
   - Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `lightgbm`, `statsmodels`

2. **Install Dependencies:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm statsmodels
   ```

3. **Run the Code:**
   - Download the dataset (`qsar_fish_toxicity.csv`) and update the file path in the code.
   - Execute the Python script to preprocess the data, train the models, and evaluate their performance.

---

## Code Structure
- **Data Loading and Preprocessing:**
  - Load the dataset and handle missing values.
  - Remove outliers and scale features.
  - Bin the target variable for classification tasks.

- **Exploratory Data Analysis (EDA):**
  - Visualize data distributions, correlations, and missing values.
  - Generate pair plots and heatmaps.

- **Model Implementation:**
  - Train and evaluate regression, classification, and clustering models.
  - Generate evaluation metrics and visualizations.

- **Visualization:**
  - Plot ROC-AUC curves, confusion matrices, and feature importance.
  - Compare clustering evaluation metrics.

---

## Future Improvements
1. **Hyperparameter Tuning:**
   - Use grid search or random search to optimize hyperparameters for each model.
2. **Feature Engineering:**
   - Experiment with additional features or feature transformations.
3. **Ensemble Methods:**
   - Implement ensemble models like Stacking or Voting to improve performance.
4. **Handling Class Imbalance:**
   - Address class imbalance using techniques like SMOTE or class weights.
5. **Cross-Validation:**
   - Use cross-validation to ensure the models generalize well to unseen data.

---

## Conclusion
This project demonstrates the application of various machine learning models to predict fish toxicity based on chemical descriptors. The best-performing models for regression were SVR and Random Forest, while Decision Tree, Random Forest, Gradient Boosting, and XGBoost excelled in classification tasks. For clustering, KMeans performed the best. By incorporating additional features, handling class imbalance, and using cross-validation, the performance of the models can be further improved.



**Note:** This project is for educational purposes only and should not be used for actual toxicity prediction without further validation.
