import pandas as pd
import config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

df = pd.read_csv(config.CLEAN_FILE)
targets = df.price.values.reshape(-1, 1)
features = df.drop("price", axis=1).values

scaler = StandardScaler()
targets = scaler.fit_transform(targets)
features = scaler.fit_transform(features)
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.25)

regr = SVR(kernel="linear", C=2.911547949756579, degree=4)
regr.fit(x_train, y_train)
pred = regr.predict(x_test)
mse = mean_squared_error(y_test, pred)

print(mse)
