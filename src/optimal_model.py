import pandas as pd
import config
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

df = pd.read_csv(config.CLEAN_FILE)
targets = df.price.values
features = df.drop("price", axis=1).values

# scaler = StandardScaler()
# targets = scaler.fit_transform(targets)
# features = scaler.fit_transform(features)

x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.25)


print(y_train)

regr = RandomForestRegressor(
    n_estimators=849,
    max_depth=15,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="log2",
)
regr.fit(x_train, y_train)
pred = regr.predict(x_test)
mse = mean_squared_error(y_test, pred)
print(mse)


# x_train = scaler.inverse_transform(x_train)
# y_train = scaler.inverse_transform(y_train)
# y_test = scaler.inverse_transform(y_test)

# inverse_transform


# Best hyperparameters:
# {'model_type': 'RandomForestRegressor', 'n_estimators': 849, 'max_depth': 15, 'min_samples_split': 4, 'min_samples_leaf': 2, 'max_features': 'log2'}
