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
    :param param_distributions:
    :return:
    """
    GSCV = GridSearchCV(estimator=estimator, param_grid=param_distributions, cv=cv, n_jobs=n_jobs)
    GSCV.fit(x_train, y_train.values.ravel())
    results = GSCV.cv_results_
    # for key, value in param_distributions.items():
    plot_grid_search(results, param_distributions["n_estimators"], param_distributions["max_depth"], 'N Estimators', 'Max Features')
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
    results = rf_random.cv_results_
    return rf_random.best_params_


def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label=name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    plt.savefig('grid_search')


def search_hyperparameter_random_forest(x_train, y_train, search, search_para):
    """
    search for good hyperparameter configs with random or grid search
    :param search: "grid" or "random"
    :return: best_parameters
    """
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


def search_hyperparamter_gradientboosting(x_train, y_train, search, search_para):
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


