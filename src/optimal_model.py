import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import config
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv(config.TRAINING_NO_SCALE)
targets = df[["price"]]
features = df.drop("price", axis=1)
features = df[
    ["bedrooms", "bathrooms", "sqft_living", "floors", "condition", "yr_built"]
]
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.25)


def train():
    model = RandomForestRegressor(
        n_estimators=618,
        max_depth=12,
        min_samples_split=19,
        min_samples_leaf=2,
        max_features="auto",
    )

    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"RMSE for RandomForest Regressor {rmse:0.2f}")
    pickle.dump(model, open(config.MODEL_PATH, "wb"))

    feat_imp = pd.Series(model.feature_importances_, index=features.columns)
    feat_imp.plot(kind="barh", figsize=(10, 6))
    plt.show()


if __name__ == "__main__":
    train()
