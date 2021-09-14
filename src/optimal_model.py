import pandas as pd
import numpy as np
import config
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import pickle


df = pd.read_csv(config.TRAINING_NO_SCALE)
targets = df[['price']].values
features = df[['price','bedrooms', 'bathrooms', 'sqft_living', 'floors', 'condition', 'yr_built', 'yr_renovated']].values

x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.25)
# x_train = x_train.reshape(x_train.shape[0], -1)
# x_test = x_test.reshape(x_test.shape[0], -1)
# print(y_train.shape)
def train():
    model = XGBRegressor(
        eta=0.1,
        gamma=0.5,
        max_depth=9,
        min_child_weight=1,
        subsample=1.0,
    )

    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"RMSE for RandomForest Regressor {rmse:0.2f}")
    # pickle.dump(model, open('../models/model.pkl', 'wb'))
    

if __name__ == "__main__":
    train()
