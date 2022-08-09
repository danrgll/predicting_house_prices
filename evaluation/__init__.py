import data_preparation as dp
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn import metrics
from matplotlib import pyplot as plt


# ToDo: eine datei ausgeben mit den wichtigsten daten des Models als Validierungsprozess

def evaluate(model, test_features, test_labels):
    """evaluate regression model with test data"""
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy


def fit_and_val_random_forest(x_train, y_train, para, save=True):
    """
    :param save: save model (True, False)
    :param para: parameter from model in a dict
    :return:
    """
    best_model = RandomForestRegressor(**para)
    x_train, x_val, y_train, y_val = dp.split_data(x_train, y_train, test_size=0.1, shuffle=True)
    best_model.fit(x_train, y_train.values.ravel())
    if save is True:
        filename = "rf_model_val.sav"
        pickle.dump(best_model, open(filename, "wb"))
    print("METRICS")
    print(best_model.feature_importances_)
    zipped = zip(best_model.feature_importances_, x_train.columns)
    sorted_zip = sorted(zipped, key=lambda x: x[0])
    zipped_list = list(sorted_zip)
    print(zipped_list)
    columns = list(zip(*zipped_list))
    sort_feature_imp = list(columns[0])
    x_features = list(columns[1])
    print(columns)
    # print("feature importance:", zipped_list)
    plt.barh(x_features, sort_feature_imp)
    plt.show(block=True)
    prediction = best_model.predict(x_val)
    print("Test")
    y_true = y_val["SalePrice"].tolist()
    print(y_true)
    print(prediction)
    print(f"MEA: {metrics.mean_absolute_error(y_val, prediction)}")
    validation = y_val.assign(Prediction=prediction)
    validation["Abweichung"] = abs(validation["SalePrice"] - validation["Prediction"])
    print(validation.info())
    df_val = validation.sort_values("Abweichung")
    df_val.to_csv("validation_random_forest.csv", encoding="utf-8")
    print(df_val)
    evaluate(best_model, x_val, y_true)


def fit_and_val_gradient_boosting(x_train, y_train, para, save=True):
    """
    :param save: save model(True,False)
    :param para: parameter from model in a dict
    :return:
    """
    best_model = GradientBoostingRegressor(**para)
    """    best_model = GradientBoostingRegressor(n_estimators=200, criterion='mse',
             learning_rate=0.03, loss='ls', max_depth=12,
             max_features=None, max_leaf_nodes=None,
             min_samples_leaf=16, min_samples_split=16,
             subsample=1.0)
             """
    x_train, x_val, y_train, y_val = dp.split_data(x_train, y_train, test_size=0.1, shuffle=True)
    best_model.fit(x_train, y_train.values.ravel())
    if save is True:
        filename = "gbrt_model_val.pkl"
        pickle.dump(best_model, open(filename, "wb"))
    print("METRICS")
    print(best_model.feature_importances_)
    zipped = zip(best_model.feature_importances_, x_train.columns)
    sorted_zip = sorted(zipped, key=lambda x: x[0])
    zipped_list = list(sorted_zip)
    print(zipped_list)
    columns = list(zip(*zipped_list))
    sort_feature_imp = list(columns[0])
    x_features = list(columns[1])
    print(columns)
    # print("feature importance:", zipped_list)
    plt.barh(x_features, sort_feature_imp)
    plt.show(block=True)
    prediction = best_model.predict(x_val)
    print("Test")
    y_true = y_val["SalePrice"].tolist()
    print(y_true)
    print(prediction)
    print(f"MEA: {metrics.mean_absolute_error(y_val, prediction)}")
    validation = y_val.assign(Prediction=prediction)
    validation["Abweichung"] = abs(validation["SalePrice"] - validation["Prediction"])
    print(validation.info())
    df_val = validation.sort_values("Abweichung")
    df_val.to_csv("validation_grdb.csv", encoding="utf-8")
    print(df_val)
    evaluate(best_model, x_val, y_true)


def fit_val_neurol_network():
    pass


def fit_and_val_ensemble_model(x_train, y_train, para_rf, para_gdb):
    rf = RandomForestRegressor(**para_rf)
    gdb = GradientBoostingRegressor(**para_gdb)
    ensemble_model = VotingRegressor([("rf", rf), ("grdb", gdb)], n_jobs=-1)
    x_train, x_val, y_train, y_val = dp.split_data(x_train, y_train, test_size=0.1, shuffle=True)
    ensemble_model.fit(x_train, y_train.values.ravel())
    print("METRICS")
    prediction = ensemble_model.predict(x_val)
    print("Test")
    y_true = y_val["SalePrice"].tolist()
    print(y_true)
    print(prediction)
    print(f"MEA: {metrics.mean_absolute_error(y_val, prediction)}")
    validation = y_val.assign(Prediction=prediction)
    validation["Abweichung"] = abs(validation["SalePrice"] - validation["Prediction"])
    print(validation.info())
    df_val = validation.sort_values("Abweichung")
    df_val.to_csv("validation_ensemble.csv", encoding="utf-8")
    print(df_val)
    evaluate(ensemble_model, x_val, y_true)