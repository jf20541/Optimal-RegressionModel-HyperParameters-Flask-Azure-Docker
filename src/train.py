import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import optuna
import config


df = pd.read_csv(config.CLEAN_FILE)
targets = df.price.values.reshape(-1, 1)
features = df.drop("price", axis=1).values

scaler = StandardScaler()
targets = scaler.fit_transform(targets)
features = scaler.fit_transform(features)

x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.25)


def create_model(trial):
    model_type = trial.suggest_categorical(
        "model_type",
        [
            "decision-tree",
            "svr",
            "XGBRegressor",
            "RandomForestRegressor",
            "KNeighborsRegressor",
        ],
    )

    if model_type == "svr":
        kernel = trial.suggest_categorical(
            "kernel", ["linear", "poly", "rbf", "sigmoid"]
        )
        regularization = trial.suggest_uniform("svm-regularization", 0.01, 10)
        degree = trial.suggest_discrete_uniform("degree", 1, 5, 1)
        model = SVR(kernel=kernel, C=regularization, degree=degree)

    if model_type == "decision-tree":
        max_depth = trial.suggest_int("max_depth", 5, 10)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 20)

        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )

    if model_type == "XGBRegressor":
        eta = trial.suggest_discrete_uniform("eta", 0.1, 1.0, 0.1)
        gamma = trial.suggest_discrete_uniform("gamma", 0.1, 1.0, 0.1)
        max_depth = trial.suggest_int("max_depth", 5, 10)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 5)
        subsample = trial.suggest_discrete_uniform("subsample", 0.1, 1.0, 0.1)

        model = XGBRegressor(
            eta=eta,
            gamma=gamma,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
        )

    if model_type == "RandomForestRegressor":
        n_estimators = trial.suggest_int("n_estimators", 100, 1000)
        max_depth = trial.suggest_int("max_depth", 5, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 20)
        max_features = trial.suggest_categorical(
            "max_features", ["auto", "sqrt", "log2"]
        )

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )

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


def model_performance(model, X=x_test, y=y_test):
    y_pred = model.predict(X)
    return round(mean_squared_error(y_pred, y), 3)


def objective(trial):
    model = create_model(trial)
    model.fit(x_train, y_train.ravel())
    return model_performance(model)


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    # get optimal model and its hyper-parameters
    best_model = create_model(study.best_trial)
    best_model.fit(x_train, y_train.ravel())
    print("Performance: ", model_performance(best_model))
