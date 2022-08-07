import data_preparation as dp
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn import metrics
from matplotlib import pyplot as plt
from encoder import transform_categorial_into_numeric, transform_categorial_into_target
from hyperparamter_search import search_hyperparameter_random_forest, search_hyperparamter_gradientboosting


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


"""
def evaluate_random_forest(x_train, y_train, n_estimators, max_features, max_depth):
    rnd_forest = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, n_jobs=-1,
                                       bootstrap=True, oob_score=True)
    rnd_forest.fit(x_train, y_train.values.ravel())
    print("METRICS")
    zipped = zip(rnd_forest.feature_importances_, x_train.columns)
    sorted_zip = sorted(zipped, key=lambda x: x[0])
    zipped_list = list(sorted_zip)
    print("feature importance:", zipped_list)
    plt.barh(list(x_train.columns), rnd_forest.feature_importances_)
    plt.show()
    print("Oob_score:", rnd_forest.oob_score_)
    print("Median HousePrices:", y_train.median())
    print("Min HousePrice", y_train.min())
    print("Max HousePrice", y_train.max())
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, rnd_forest.oob_prediction_))
    print('Mean Squared Error:', metrics.mean_squared_error(y_train, rnd_forest.oob_prediction_))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, rnd_forest.oob_prediction_)))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, rnd_forest.oob_prediction_)))
    return rnd_forest


def evaluate_gradient_boosting(x_train, y_train, x_val, y_val, max_depth=2, subsample=0.8, learning_rate=0.1):
    gbrt = GradientBoostingRegressor(max_depth=max_depth, n_estimators=1000, subsample=subsample, learning_rate=learning_rate)
    gbrt.fit(x_train, y_train.values.ravel())
    print("METRICS")
    errors = [metrics.mean_squared_error(y_val, y_pred)
              for y_pred in gbrt.staged_predict(x_val)]
    bst_n_estimators = np.argmin(errors) + 1
    print(bst_n_estimators)
    gbrt_best = GradientBoostingRegressor(max_depth=max_depth, n_estimators=bst_n_estimators, subsample=subsample, learning_rate=learning_rate)
    gbrt_best.fit(x_train, y_train.values.ravel())
    y_predict = gbrt_best.predict(x_val)
    print("Mean squared error")
    val_error = metrics.mean_squared_error(y_val, y_predict)
    print("Mean squared error", val_error)
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_val, y_predict)))
    x_ax = range(len(y_val))
    plt.scatter(x_ax, y_val, s=5, color="blue", label="original")
    plt.plot(x_ax, y_predict, lw=0.8, color="red", label="predicted")
    plt.legend()
    plt.show()

"""


def clean_house_prices_data():
    df_train = dp.load_data("train.csv", encoding="utf-8")
    df_train.drop(
        labels=[17, 90, 102, 156, 182, 259, 342, 362, 371, 392, 520, 532, 533, 552, 646, 705, 736, 749, 778, 868,
                894,
                897, 984, 1000, 1011, 1035, 1045, 1048, 1049, 1090, 1179, 1216, 1218, 1230, 1321, 1412, 553,
                1232, 39, 948, 332, 1379], axis=0)
    df_train.to_csv("check5.csv", encoding="utf-8")
    # target_data = df_train.iloc[:, 80:81]
    # del df_train["SalePrice"]
    df_test = dp.load_data("test.csv", encoding="utf-8")
    df_test.insert(2, "SalePrice", [1] * df_test.shape[0], True)
    df = pd.concat([df_train, df_test], keys=[0, 1])
    # CLEAN DATA
    # drop out columns that have more than 50% null objects over all data points or correlate with other features
    dp.delete_features(df,
                       ["Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature", "Utilities", "Street",
                        "LowQualFinSF",
                        "GarageYrBlt", "Id"])
    average = dp.replace_null_with_average_number(df, ["LotFrontage"])
    df["MasVnrType"].replace(to_replace=np.nan, value="None", inplace=True)
    df["MasVnrArea"].replace(to_replace=np.nan, value=0.0, inplace=True)
    df[['GarageQual', 'GarageCond', "GarageType", "GarageFinish"]] = df[
        ["GarageQual", "GarageCond", "GarageType", "GarageFinish"]].fillna('NI')
    # drop rows which have missing values in choosen columns
    df = df.dropna(axis=0, subset=["MSZoning", "Exterior1st", "Exterior2nd", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
                                   "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "KitchenQual", "Functional", "GarageCars",
                                   "GarageArea", "SaleType", "BsmtQual", "BsmtExposure"])
    return df


def encoding_data(df):
    # transform categorial variables to a target(Encoding) for all possibilieties
    df_train, df_test = df.xs(0), df.xs(1)
    df_train["SalePrice"].to_csv("check4", encoding="utf-8")
    dp.delete_features(df_test, ["SalePrice"])
    # df.insert(-1, "SalePrice", target_data, True)
    features_transformed = ["MSZoning",
                            "LotShape", "LandContour", "LotConfig",
                            "LandSlope", "Neighborhood", "Condition1",
                            "Condition2",
                            "BldgType", "HouseStyle", "RoofStyle",
                            "RoofMatl",
                            "Exterior1st", "Exterior2nd", "MasVnrType",
                            "ExterQual",
                            "ExterCond", "Foundation", "BsmtQual",
                            "BsmtCond",
                            "BsmtExposure", "BsmtFinType1",
                            "BsmtFinType2",
                            "Heating", "HeatingQC", "CentralAir",
                            "Electrical",
                            "KitchenQual", "Functional", "GarageType",
                            "GarageFinish", "GarageQual", "GarageCond",
                            "PavedDrive", "SaleType", "SaleCondition"]
    df_train, categorial_encoders = transform_categorial_into_target(df_train, features_transformed)
    for feature in features_transformed:
        df_test[feature] = categorial_encoders[feature].transform(df_test[feature])
    df_train.to_csv("clean_train_data.csv", encoding="utf-8")
    df_test.to_csv("clean_test_data.csv", encoding="utf-8")
    return df_train, df_test


# ToDo: eine datei ausgeben mit den wichtigsten daten des Models als Validierungsprozess

def fit_and_val_random_forest(x_train, y_train, save=True):
    """
    :param save: save model (True, False)
    :return:
    """
    best_model = RandomForestRegressor(n_estimators=600, min_samples_split=3, min_samples_leaf=1, max_features="sqrt",
                                       max_depth=30, bootstrap=False)
    x_train, x_val, y_train, y_val = dp.split_data(x_train, y_train, test_size=0.1, shuffle=True)
    best_model.fit(x_train, y_train.values.ravel())
    if save is True:
        filename = "rf_model.sav"
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
    plt.show()
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


def fit_and_val_gradient_boosting(x_train, y_train, save=True):
    """
    :param save: save model(True,False)
    :return:
    """
    para = {'subsample': 0.9, 'n_estimators': 400, 'min_samples_split': 3, 'min_samples_leaf': 2, 'max_features': 'auto',
            'max_depth': 2, 'loss': 'squared_error', 'learning_rate': 0.08}
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
        filename = "gbrt_model.pkl"
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
    plt.show()
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


def fit_val_neurol_network():
    pass


def fit_and_val_ensemble_model(x_train, y_train):
    rf = RandomForestRegressor()
    grdb = GradientBoostingRegressor()
    ensemble_model = VotingRegressor([("rf", rf), ("grdb", grdb)], n_jobs=-1)
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


if __name__ == '__main__':
    df = clean_house_prices_data()
    df_train, df_test = encoding_data(df)
    print(df_train.info())
    x_train = df_train.iloc[:, 0:70]
    y_train = df_train.iloc[:, 70:71]
    # para = search_hyperparamter_gradientboosting(x_train, y_train, "random")
    # para =search_hyper_para_random_forest(x_train, y_train, "grid")
    # print(para)
    # {'bootstrap': False, 'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 600}
    # mit cv=3, {'n_estimators': 673, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 40, 'bootstrap': False}
    # mit cv=10 {'n_estimators': 231, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': False}
    # best_model = RandomForestRegressor(n_estimators=673, min_samples_split=2, min_samples_leaf=1, max_features="sqrt", max_depth=40, bootstrap=False)
    print(y_train)
    # fit_and_val_random_forest(x_train, y_train)
    fit_and_val_gradient_boosting(x_train, y_train)
    # MEA: 20847.467290285473
