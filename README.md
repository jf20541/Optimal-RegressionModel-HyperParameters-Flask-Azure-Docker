# Regression-OptimalParameters-Flask-Azure


## Objective

## Regression Models 
**XGBoost Regressor:**

**RandomForest Regressor:**

**Decision Tree Regressor:**

**KNeighbors Regressor with Scaling:**

**Support Vector Regressor with Scaling:**


## Metrics and Optuna (Optimization Framework)



## Repository File Structure
    ├── src          
    │   ├── optimal_model.py        # Extract optimal Regression Model with its optimal hyper-parameter for deployment
    │   ├── train_no_scaling.py     # Optimal (RandomForest, DecisionTree, XGBRegressor) without scaled data and optimized hyper-parameters
    │   ├── train_scaling.py        # Optimal (KNearestNeighbor & SVM) with scaled data and optimized hyper-parameters.
    │   └── config.py               # Define path as global variable
    ├── inputs
    │   ├── train.csv               # Training dataset
    │   └── train_clean.csv         # Cleaned data, featured engineered, scaled
    ├── notebooks
    │   └── house_price_eda.ipynb   # Exploratory Data Analysis and Feature Engineering
    ├── requierments.txt            # Packages used for project
    ├── sources.txt                 # Sources
    └── README.md
    
## Output
```bash
Optimal Model: RandomForestRegressor

Optimal Hyper-Parameters:
'n_estimators': 951
'max_depth': 16
'min_samples_split': 5
'min_samples_leaf': 4
'max_features': 'auto'
```

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

