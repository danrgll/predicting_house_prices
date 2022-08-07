import data_preparation as dp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from encoder import transform_categorial_into_numeric, transform_categorial_into_target


def grid_search(x_train, y_train, estimator, param_distributions, cv=5, n_jobs=-1, seed=42):
    """
    Grid search for random_forest
    :param x_train:
    :param y_train:
    :param seed:
    :param n_estimators: Anzahl Estimators
    :param max_features:
    :param max_depth:
    :param bootstrap:
    :param criterion:
    :return:
    """
    GSCV = GridSearchCV(estimator=estimator, param_grid=param_distributions, cv=cv, n_jobs=n_jobs)
    GSCV.fit(x_train, y_train.values.ravel())
    return GSCV.best_params_


def random_search(x_train, y_train, estimator, param_distributions, n_iter=100, cv=5, verbose=2, n_jobs=-1, seed=42):
    """

    :param estimator: der Schätzer/das Model für das gute Hyperparameter gesucht werden sollen
    :param param_distributions: dict mit den Hyperparameter als key in Form eines Strings und als
    Value eine Liste mit den auszuprobierenden Parametern
    :param n_iter: Number of parameter settings that are sampled
    :param cv: Determines the cross-validation splitting strategy. None, to use the default 5-fold cross validation,
    integer, to specify the number of folds in a (Stratified)KFold,
    :param verbose: Controls the verbosity: the higher, the more messages.
    :param n_jobs: Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
    :return: best_paramter for estimator
    """
    rf_random = RandomizedSearchCV(estimator=estimator, param_distributions=param_distributions, n_iter=n_iter, cv=cv, verbose=verbose,
                                   random_state=seed, n_jobs=n_jobs)
    rf_random.fit(x_train, y_train.values.ravel())
    return rf_random.best_params_


def search_hyperparameter_random_forest(x_train, y_train, search):
    """
    search for good hyperparameter configs with random or grid search
    :param search: "grid" or "random"
    :return: best_parameters
    """
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=10, stop=600, num=5)]
    # Number of features to consider at every split
    max_features = ['sqrt']
    # Maximum number of levels in tree
    max_depth = [20, 30, 40]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 3]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1]
    # Method of selecting samples for training each tree
    bootstrap = [False]
    search_para = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    rf = RandomForestRegressor()
    if search == "random":
        n_iter = 200  # number of samples
        best_para = random_search(x_train, y_train, rf, search_para, n_iter=n_iter, cv=10)
    elif search == "grid":
        best_para = grid_search(x_train, y_train, rf, search_para, cv=5)
    else:
        raise Exception("This search is not reachable")
    print("Best parameter found at search")
    print(best_para)
    return best_para


def search_hyperparamter_gradientboosting(x_train, y_train, search):
    # loss function die genutzt werden soll um den Fehler zu berechnen
    loss = ["squared_error", "absolute_error", "huber"]
    # Lernrate
    learningrate = [0.08, 0.1, 0.2, 0.3]
    # Anzahl schwacher Lerner die aufeinander aufbauen. The number of Boosting stages to perform.
    n_estimators = [60, 80, 100, 150, 200, 250, 300, 400, 500, 700, 1000]
    # Anzahl Datenpunkte in % die fürs Fitten der einzelnen Lerner verwendet werden soll,
    # weniger als 100% nennt sich STochhastic Gradient boosting
    subsample = [1.0, 0.97, 0.9, 0.95]
    min_samples_split = [2, 3, 4]
    min_samples_leaf = [1, 2, 3]
    max_depth = [2, 3, 5, 7, 9, None]
    # The number of features to consider when looking for the best split:
    max_features = ["auto", "sqrt", "log2", "None"]
    search_para = {"loss": loss,
                   "learning_rate": learningrate,
                   "n_estimators": n_estimators,
                   "subsample": subsample,
                   "min_samples_split": min_samples_split,
                   "min_samples_leaf": min_samples_leaf,
                   "max_depth": max_depth,
                   "max_features": max_features}
    grdb = GradientBoostingRegressor()
    if search == "random":
        n_iter = 300  # number of samples
        best_para = random_search(x_train, y_train, grdb, search_para, n_iter=n_iter, cv=10)
    elif search == "grid":
        best_para = grid_search(x_train, y_train, grdb, search_para, cv=10)
    else:
        raise Exception("This search is not reachable")
    print("Best parameter found at search")
    print(best_para)
    return best_para
















