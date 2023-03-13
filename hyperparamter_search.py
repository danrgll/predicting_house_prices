import data_preprocessing as dp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from data_encoding import transform_categorial_into_numeric, transform_categorial_into_target


class GridSearch:
    def __init__(self, x_train, y_train, estimator, param_distributions, cv=5, n_jobs=-1, seed=42):
        self.GSCV = GridSearchCV(estimator=estimator, param_grid=param_distributions, cv=cv, n_jobs=n_jobs,
                            return_train_score=True)
        self.GSCV.fit(x_train, y_train.values.ravel())
        self.results = self.GSCV.cv_results_

    def return_best_parameters(self):
        return self.GSCV.best_params_

    def return_best_estimator(self):
        return self.GSCV.best_estimator_

    def plot_grid_search(self, grid_param_1, grid_param_2, name_param_1, name_param_2, file_name,
                         plot_training=False):
        """
        :param grid_param_1:
        :param grid_param_2:
        :param name_param_1:
        :param name_param_2:
        :param file_name:
        :param plot_training:
        :return:
        """
        # Plot Grid search scores
        _, ax = plt.subplots(1, 1)
        if plot_training is True:
            scores_mean_training = self.results['mean_train_score']
            scores_mean_training = np.array(scores_mean_training).reshape(len(grid_param_2), len(grid_param_1))
            for idx, val in enumerate(grid_param_2):
                ax.plot(grid_param_1, scores_mean_training[idx, :], '-o', label=name_param_2 + ': ' + str(val))

        scores_mean = self.results['mean_test_score']
        scores_mean = np.array(scores_mean).reshape(len(grid_param_2), len(grid_param_1))

        scores_sd = self.results['std_test_score']
        scores_sd = np.array(scores_sd).reshape(len(grid_param_2), len(grid_param_1))
        for idx, val in enumerate(grid_param_2):
            ax.plot(grid_param_1, scores_mean[idx, :], '-o', label=name_param_2 + ': ' + str(val))

        ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
        ax.set_xlabel(name_param_1, fontsize=16)
        ax.set_ylabel('CV Average Score', fontsize=16)
        ax.legend(loc="best", fontsize=15)
        ax.grid('on')
        plt.savefig(file_name)


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
                                   random_state=seed, n_jobs=n_jobs, return_train_score=True)
    rf_random.fit(x_train, y_train.values.ravel())
    results = rf_random.cv_results_
    return rf_random.best_params_


def search_hyperparameter_random_forest(x_train, y_train, search, search_para, plot_training):
    """
    search for good hyperparameter configs with random or grid search
    :param search: "grid" or "random"
    :return: gridsearch object
    """
    rf = RandomForestRegressor()
    if search == "random":
        n_iter = 200  # number of samples
        best_para = random_search(x_train, y_train, rf, search_para, n_iter=n_iter, cv=10)
    elif search == "grid":
        gridsearch = GridSearch(x_train, y_train, rf, search_para, cv=5)
        gridsearch.plot_grid_search(search_para["n_estimators"], search_para["max_depth"], "n Estimators",
                                    "max depth", "estimators_max depth_test", plot_training=plot_training)
        # gridsearch.plot_grid_search(search_para["n_estimators"], search_para[""])
    else:
        raise Exception("This search is not reachable")
    return gridsearch


def search_hyperparamter_gradientboosting(x_train, y_train, search, search_para, plot_training):
    grdb = GradientBoostingRegressor()
    if search == "random":
        n_iter = 300  # number of samples
        best_para = random_search(x_train, y_train, grdb, search_para, n_iter=n_iter, cv=10)
    elif search == "grid":
        gridsearch = GridSearch(x_train, y_train, grdb, search_para, cv=5)
        # gridsearch.plot_grid_search(search_para["n_estimators"], search_para["learning_rate"], "n Estimators",
        #                           "lr_rate", "gdb_estimators_lr_test", plot_training=plot_training)
        gridsearch.plot_grid_search(search_para["n_estimators"], search_para["subsample"], "n Estimators",
                                    "subsample", "gdb_estimators_subsample_test", plot_training=plot_training)
    else:
        raise Exception("This search is not reachable")
    return gridsearch


