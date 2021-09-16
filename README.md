# Optimal-RegressionModel-HyperParameters-Flask-Azure

## Objective
Monitor and seek the optimal regression model **(DecisionTreeRegressor, RandomForestRegressor, XGBRegressor, SVR, KNeighborsRegressor)** and optimize each model's hyper-parameters using Tree-structured Parzen Estimator Approach (TPE) by iterating 1,000 trials. Evaluated the model's performance based on RMSE given different approaches by feature engineering (One-Hot Encoding, Target Encoding, etc) for the house prediction dataset. 

## Optimal Regression Model and Optimal Hyper-Parameters

**RandomForest Regressor: TRAINING_CLEAN**
```
Performance: 118098.32337847917. RMSE
Best hyperparameters: {'model_type': 'RandomForestRegressor', 'n_estimators': 618, 'max_depth': 12, 'min_samples_split': 19, 'min_samples_leaf': 2, 'max_features': 'auto'}
```

## Docker
```bash
 docker build -t optimalregressionapi .
 docker run -ti optimalregressionapi  
 source activate ml 
 
 # Train Optimal Regression Model with Optuna
 cd src/src
 python train_no_scaling.py
 python train_scaling.py
 python optimal_model.py
 
 # Deploy Model using Flask 
 cd .. 
 python app.py
```

## Deployment with Flask/Azure
<p align="center">
  <img width="600" height="800" src="https://github.com/jf20541/Optimal-RegressionModel-HyperParameters-Flask-Azure/blob/main/inputs/Screen%20Shot%202021-09-14%20at%2012.11.13%20PM.png?raw=true">
  https://randomforestregressorhomeprediction.azurewebsites.net
</p>

## Metric
![](https://latex.codecogs.com/svg.latex?%5Cfn_phv%20%5Clarge%20RMSE%20%3D%20%5Csqrt%7B%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7D%28Predicted_%7Bi%7D%20-%20Actual_%7Bi%7D%5E%7B%7D%29%5E%7B2%7D%7D%7BN%7D%7D)


## Tree-structured Parzen Estimator Approach (TPE)
**Objective**\
Build a probability model of the objective function and use it to select the optimal regression model and its hyper-parameters to evaluate its performance by minimizing RMSE on the testing set. With Bayesian approach, it keeps track of the past results which it uses to form a probabilistic model mapping hyper-parameters to a probability of a score on the objective function.

TPE is a model that applies the Bayes Rule given the equation below, where the probability of (y) RMSE score on the objective function given a set of hyper-parameters (x)

![](https://latex.codecogs.com/svg.latex?%5Cfn_phv%20%5Clarge%20p%28y%7Cx%29%20%3D%20%5Cfrac%7Bp%28x%7Cy%29*p%28y%29%7D%7Bp%28x%29%7D)


With **TWO** distributions for hyper-parameters
1. **l(x):** the value of the objective function is less than the threshold
2. **g(x):** the value of the objective function is greater than the threshold

![](https://latex.codecogs.com/svg.latex?%5Cfn_phv%20%5Clarge%20p%28x%7Cy%29%20%3D%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20%5Cl%28x%29%20%26%20if%5C%3B%20y%20%3C%20y*%5C%5C%20g%28x%29%20%26%20if%5C%3B%20y%20%5Cgeq%20y*%20%5Cend%7Bmatrix%7D%5Cright.)

**Expected Improvement (EI)** objective is to maximize the ratio below. The objective function records the results and its hyper-parameters, forming a history. TPE works by iterating pairs to form l(x), evaluates each ratios l(x)/g(x), and returns the highest Expected Improvement.

![](https://latex.codecogs.com/svg.latex?%5Cfn_phv%20%5Clarge%20EI_%7By*%7D%20%28x%29%20%3D%20arg%20max_%7Bx%5Cepsilon%20%5Cchi%20%7D%20%5Cfrac%7Bl%28x%29%7D%7Bg%28x%29%7D)


## Repository File Structure
    ├── src          
    │   ├── optimal_model.py        # Extract optimal Regression Model with its optimal hyper-parameter for deployment
    │   ├── train_no_scaling.py     # Optimal (RandomForest, DecisionTree, XGBRegressor) without scaled data and optimized hyper-parameters
    │   ├── train_scaling.py        # Optimal (KNearestNeighbor & SVM) with scaled data and optimized hyper-parameters.
    │   └── config.py               # Define path as global variable
    ├── inputs
    │   ├── train.csv               # Training dataset from Kaggle
    │   ├── train_no_scale.csv      # No scaled dataset (featured engineered, and feature selection)
    │   ├── train_scale.csv         # Scaled dataset (featured engineered, and feature selection)
    │   └── train_clean.csv         # Cleaned data, featured engineered, scaled
    ├── templates
    │   └── home.html               # HTML Code for front end deployment
    ├── statics
    │   └── css
    │       └── style.css           # Apply a unique style to a HTML elements
    ├── notebooks
    │   └── house_price_eda.ipynb   # EDA, Feature Engineering, Feature Selection
    ├── requierments.txt            # Packages used for project
    ├── sources.txt                 # Sources
    ├── Dockerfile                  # Dockerize Flask Application 
    └── README.md

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
