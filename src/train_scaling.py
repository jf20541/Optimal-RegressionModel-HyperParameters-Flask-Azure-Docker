import pandas as pd
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import optuna
import config


df = pd.read_csv(config.CLEAN_FILE)
targets = df.price.values.reshape(-1, 1)
features = df.drop("price", axis=1).values

scaler = MinMaxScaler()
targets = scaler.fit_transform(targets)
features = scaler.fit_transform(features)


x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.25)


def create_model(trial):
    model_type = trial.suggest_categorical(
        "model_type",
        [
            "SVR",
            "KNeighborsRegressor",
        ],
    )

    if model_type == "SVR":
        kernel = trial.suggest_categorical(
            "kernel", ["linear", "poly", "rbf", "sigmoid"]
        )
        regularization = trial.suggest_uniform("svm-regularization", 0.01, 10)
        degree = trial.suggest_discrete_uniform("degree", 1, 5, 1)
        model = SVR(kernel=kernel, C=regularization, degree=degree)

    if model_type == "KNeighborsRegressor":
        n_neighbors = trial.suggest_int("n_neighbors", 3, 10)
        weights = trial.suggest_categorical(
            "weights", ["uniform", "distance", "uniform"]
        )
        p = trial.suggest_int("p", 2, 5)

        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p)

    if trial.should_prune():
        raise optuna.TrialPruned()

    return model


def model_performance(model, x_test, y_test):
    pred = model.predict(x_test)
    return sqrt(mean_squared_error(y_test, pred))


def objective(trial):
    model = create_model(trial)
    model.fit(x_train, y_train.ravel())
    return model_performance(model, x_test, y_test)


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    # get optimal model and its hyper-parameters
    best_model = create_model(study.best_trial)
    best_model.fit(x_train, y_train.ravel())
    trial = study.best_trial
    print(f"Performance: {model_performance(best_model, x_test, y_test)}")
    print(f"Best hyperparameters: {trial.params}")


# Best hyperparameters: {'model_type': 'RandomForestRegressor', 'n_estimators': 849, 'max_depth': 15, 'min_samples_split': 4, 'min_samples_leaf': 2, 'max_features': 'log2'}
