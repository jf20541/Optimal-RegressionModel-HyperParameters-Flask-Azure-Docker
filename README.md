# Regression-OptimalParameters-Flask-Azure


## Objective
Monitor and optimize regression models **(DecisionTreeRegressor, RandomForestRegressor, XGBRegressor, SVR, KNeighborsRegressor)** and optimize each model's hyper-parameters using Tree-structured Parzen Estimator Approach (TPE). Evaluated the model's performance based on RMSE given a different approach of feature engineering (One-Hot Encoding, Target Encoding, etc) for the house prediction dataset. 

## Regression Models 
**XGBoost Regressor: TRAINING_CLEAN**
```
Performance: 128238.40 RMSE
Optimal Hyper-Parameters:
'eta': 0.30000000000000004, 'gamma': 0.1, 'max_depth': 5, 'min_child_weight': 1, 'subsample': 0.9}
```

**RandomForest Regressor: TRAINING_CLEAN**
```
Performance: 117951.42 RMSE
Optimal Hyper-Parameters:
'n_estimators': 951, 'max_depth': 16, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'auto'
```
**Decision Tree Regressor: TRAINING_CLEAN**
```
Performance: 183979.50
Optimal Hyper-Parameters:
'max_depth': 5, 'min_samples_split': 18, 'min_samples_leaf': 8}.
```
**KNeighbors Regressor with Scaling: TRAINING_OHE**
```
Performance: 0.5549328284676253
Optimal Hyper-Parameters:
'n_neighbors': 8, 'weights': 'uniform', 'p': 2
```
**Support Vector Regressor with Scaling: TRAINING_OHE**
```
Performance: 0.5276328030789721
Optimal Hyper-Parameters:
'kernel': 'rbf', 'svm-regularization': 0.9830618662438664, 'degree': 3.0}
```
## Metrics and Optuna (Optimization Framework)

Metric:\
![](https://latex.codecogs.com/svg.latex?%5Cfn_phv%20%5Clarge%20RMSE%20%3D%20%5Csqrt%7B%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7D%28Predicted_%7Bi%7D%20-%20Actual_%7Bi%7D%5E%7B%7D%29%5E%7B2%7D%7D%7BN%7D%7D)


Tree-structured Parzen Estimator Approach (TPE):
- Fits one Gaussian Mixture Model (GMM) to the set of parameter values associated with the best objective values, and another GMM to the remaining parameter values. It chooses the parameter value x that maximizes the ratio.




## Repository File Structure
    ├── src          
    │   ├── optimal_model.py        # Extract optimal Regression Model with its optimal hyper-parameter for deployment
    │   ├── train_no_scaling.py     # Optimal (RandomForest, DecisionTree, XGBRegressor) without scaled data and optimized hyper-parameters
    │   ├── train_scaling.py        # Optimal (KNearestNeighbor & SVM) with scaled data and optimized hyper-parameters.
    │   └── config.py               # Define path as global variable
    ├── inputs
    │   ├── train.csv               # Training dataset
    │   └── train_clean.csv         # Cleaned data, featured engineered, scaled
    ├── plots
    │   ├── price_cleaned_distribution.png   
    │   ├── price_conditions_yrbuilt.png
    │   ├── price_skewed_distribution.png   
    │   ├── train_clean_corr.png
    │   ├── train_ohe.png
    │   ├── train_target_encode_corr.png
    │   └── year_built_dist.png
    ├── notebooks
    │   └── house_price_eda.ipynb   # Exploratory Data Analysis and Feature Engineering
    ├── requierments.txt            # Packages used for project
    ├── sources.txt                 # Sources
    └── README.md
 
## Step-by-Step

## Data
[Kaggle Dataset](https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction)
```bash
Target  
Price                  int64

Features: 
Bedrooms               int64
Bathrooms              int64
Sqft_living            int64
sqft_lot               int64
floors                 int64
waterfront             int64
view                   int64
condition              int64
sqft_above             int64
sqft_basement          int64
yr_built               int64
yr_renovated           int64
city                   int64
statezip               int64
```

