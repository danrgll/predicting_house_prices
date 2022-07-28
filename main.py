import sklearn
import data_preparation as dp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV

import encoder
from encoder import Encoder
from encoder import transform_categorial_into_numeric

from numpy import nan


def grid_search(x_train, y_train, seed, n_estimators, max_features, max_depth, bootstrap, criterion):
    rnd_forest = RandomForestRegressor(random_state=seed, n_jobs=-1)
    param_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'bootstrap' : bootstrap,
        'max_depth': max_depth,
        'criterion': criterion
    }
    GSCV = GridSearchCV(estimator=rnd_forest, param_grid=param_grid, cv=5)
    GSCV.fit(x_train, y_train.values.ravel())
    return GSCV.best_params_


def evaluate_random_forest(x_train, y_train):
    rnd_forest = RandomForestRegressor(random_state=42, n_estimators=700, max_features=0.4, max_depth=40, n_jobs=-1,
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


def old_version():
    df = dp.load_data("train.csv", encoding="utf-8")
    # CLEAN DATA
    # drop out columns that have more than 50% null objects over all data points or correlate with other features
    dp.delete_features(df, ["Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature", "Utilities", "Street",
                            "LowQualFinSF", "GarageYrBlt"])
    average = dp.replace_null_with_average_number(df, ["LotFrontage"])
    print(type(df["MasVnrArea"][0]))
    df["MasVnrType"].replace(to_replace=np.nan, value="None", inplace=True)
    df["MasVnrArea"].replace(to_replace=np.nan, value=0.0, inplace=True)
    # drop rows with index because of missing values...
    df = df.drop(
        labels=[17, 90, 102, 156, 182, 259, 342, 362, 371, 392, 520, 532, 533, 552, 646, 705, 736, 749, 778, 868,
                894,
                897, 984, 1000, 1011, 1035, 1045, 1048, 1049, 1090, 1179, 1216, 1218, 1230, 1321, 1412, 553,
                1232, 39, 948, 332, 1379], axis=0)
    null_data = df[df.isnull().any(axis=1)]
    # drop all rows with nan values
    df = df.dropna()
    df.to_csv("clean_train_data.csv", encoding="utf-8")
    enge = dp.features_with_null_objects(df)
    print(enge)
    df, categorial_encoders = transform_categorial_into_numeric(df, ["MSZoning",
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
                                                                        "PavedDrive", "SaleType", "SaleCondition"])
    print(df.info())
    print(categorial_encoders["MSZoning"].classes_)
    # PIPELINE
    df = dp.delete_outliers(df, 0, 400000)
    features_data = df.iloc[:, 1:71]
    target_data = df.iloc[:, 71:72]
    # find_outliers(target_data)
    print("FEATURES_DATA")
    print(features_data)
    print("TARGET_DATA")
    print(target_data)
    x_train, y_train = dp.split_data(features_data, target_data, test_size=0)
    # best_para = grid_search(x_train, y_train, 42, [400, 500, 600, 700], [0.4, 0.5, 0.6], [30, 40, 50, 60, 70], [True, False], ["absolute_error", "squared_error"])
    # print(best_para)
    best_rnd_forest = evaluate_random_forest(x_train, y_train)
    x_test = dp.load_data("test.csv", encoding="uft-8")
    dp.delete_features(x_test,
                       ["Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature", "Utilities", "Street",
                        "LowQualFinSF", "GarageYrBlt"])
    x_test["LotFrontage"].replace(to_replace=np.nan, value=average, inplace=True)
    print(categorial_encoders)
    print("Feature with missing values:")
    for ele in dp.features_with_null_objects(x_test):
        print(ele, " \n")
    x_test.to_csv("clean_test_data.csv", encoding="utf-8")
    manual_test_data = dp.load_data("clean_test_data.csv", encoding="utf-8")

    """
    print(manual_test_data)
    print(manual_test_data.info())
    x_test = transform_categorial_into_numeric(manual_test_data, ["MSZoning",
                                                                     "LotShape", "LandContour", "LotConfig",
                                                                     "LandSlope", "Neighborhood", "Condition1", "Condition2",
                                                                     "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
                                                                     "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual",
                                                                     "ExterCond", "Foundation", "BsmtQual", "BsmtCond",
                                                                     "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
                                                                     "Heating", "HeatingQC", "CentralAir", "Electrical",
                                                                     "KitchenQual", "Functional", "GarageType",
                                                                     "GarageFinish", "GarageQual", "GarageCond",
                                                          "PavedDrive", "SaleType", "SaleCondition"], categorial_encoders)
    selected_rows = x_test[x_test.isnull().any(axis=1)]
    s = x_test.stack(dropna=False)
    L = [list(x) for x in s.index[s.isna()]]
    print(L)
    print("NaN Values")
    print(selected_rows)
    y_pred = best_rnd_forest.predict(x_test)
    print(y_pred)



    #x_train, x_val, y_train, y_val = split_data(features_data, target_data, shuffle=True)
    #evaluate_gradient_boosting(x_train, y_train, x_val, y_val)
    """


def clean_house_prices_data():
    df_train = dp.load_data("train.csv", encoding="utf-8")
    df_train = df_train.drop(
        labels=[17, 90, 102, 156, 182, 259, 342, 362, 371, 392, 520, 532, 533, 552, 646, 705, 736, 749, 778, 868,
                894,
                897, 984, 1000, 1011, 1035, 1045, 1048, 1049, 1090, 1179, 1216, 1218, 1230, 1321, 1412, 553,
                1232, 39, 948, 332, 1379], axis=0)
    target_data = df_train.iloc[:, 80:81]
    del df_train["SalePrice"]
    df_test = dp.load_data("test.csv", encoding="utf-8")
    df = pd.concat([df_train, df_test], keys=[0, 1])
    # CLEAN DATA
    # drop out columns that have more than 50% null objects over all data points or correlate with other features
    dp.delete_features(df,
                       ["Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature", "Utilities", "Street",
                        "LowQualFinSF",
                        "GarageYrBlt"])
    average = dp.replace_null_with_average_number(df, ["LotFrontage"])
    df["MasVnrType"].replace(to_replace=np.nan, value="None", inplace=True)
    df["MasVnrArea"].replace(to_replace=np.nan, value=0.0, inplace=True)
    df[['GarageQual', 'GarageCond', "GarageType", "GarageFinish"]] = df[
        ["GarageQual", "GarageCond", "GarageType", "GarageFinish"]].fillna('NI')
    # Setze manuell die Zahlen der Kategorien
    set = [["MSZoning", ["A", "C(all)", "C", "FV", "I", "RH", "RL", "RP", "RM", "NI"]],
           ["LotShape", ["Reg", "IR1", "IR2", "IR3"]],
           ["LandContour", ["Low", "HLS", "Bnk", "Lvl"]],
           
           ]
    dict_encoders = encoder.set_manual_encoder(set)
    dict_encoder["MSZoning"] = create_fit_encoder(["A", "C(all)", "C", "FV", "I", "RH", "RL", "RP", "RM"])
    dict_encoder["LotShape"] = create_fit_encoder(["Reg", "IR1", "IR2", "IR3"])
    dict_encoder["LandContour"] = create_fit_encoder(["Lvl", "Bnk", "HLS", "Low"])
    dict_encoder["LotConfig"] = create_fit_encoder(["Inside", "Corner", "CulDSac", "FR2", "FR3"])
    dict_encoder["LandSlope"] = create_fit_encoder(["Gtl", "Mod", "Sev"])
    dict_encoder[""] = create_fit_encoder([])
    dict_encoder[""] = create_fit_encoder([])
    dict_encoder[""] = create_fit_encoder([])
    dict_encoder[""] = create_fit_encoder([])
    dict_encoder[""] = create_fit_encoder([])
    dict_encoder[""] = create_fit_encoder([])
    df.to_csv("clean_data.csv", encoding="utf-8")
    return df


if __name__ == '__main__':
    df = clean_house_prices_data()
    # drop all rows with nan values
    # df = df.dropna()
    # drop rows with index because of missing values...
    enge, null_data = dp.features_with_null_objects(df)
    print(f"features:{enge}")
    #print(null_data.info())
    # drop all rows with nan values
    print(f"Info DF: {df.info()}")
    print(f"DF:{df}")
    # transform categorial variables to a one hot vector for all possibilieties
    df = pd.get_dummies(df, columns=["MSZoning",
                                      "LotShape", "LandContour", "LotConfig",
                                      "LandSlope", "Neighborhood", "Condition1", "Condition2",
                                      "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
                                      "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual",
                                      "ExterCond", "Foundation", "BsmtQual", "BsmtCond",
                                      "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
                                      "Heating", "HeatingQC", "CentralAir", "Electrical",
                                      "KitchenQual", "Functional", "GarageType",
                                      "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "SaleType",
                                      "SaleCondition"])
    # Selecting data from multi index
    df_train, df_test = df.xs(0), df.xs(1)
    print(df_train.info())
    print(df_test.info())
    # null_data = df_train[df_train.isnull().any(axis=1)]
    # df.to_csv("clean_train_data.csv", encoding="utf-8")
    # enge = dp.features_with_null_objects(df)
    # print(enge)
    # dp.location_of_null_objects()
"""

    # CLEAN DATA
    # drop out columns that have more than 50% null objects over all data points or correlate with other features
    dp.delete_features(df, ["Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature", "Utilities", "Street", "LowQualFinSF", "GarageYrBlt"])
    average = dp.replace_null_with_average_number(df, ["LotFrontage"])
    print(type(df["MasVnrArea"][0]))
    df["MasVnrType"].replace(to_replace=np.nan, value="None", inplace=True)
    df["MasVnrArea"].replace(to_replace=np.nan, value=0.0, inplace=True)
    # drop rows with index because of missing values...
    df = df.drop(labels=[17, 90, 102, 156, 182, 259, 342, 362, 371, 392, 520, 532, 533, 552, 646, 705, 736, 749, 778, 868, 894,
                         897, 984, 1000, 1011, 1035, 1045, 1048, 1049, 1090, 1179, 1216, 1218, 1230, 1321, 1412, 553,
                         1232, 39, 948, 332, 1379], axis=0)
    null_data = df[df.isnull().any(axis=1)]
    # drop all rows with nan values
    df = df.dropna()
    df.to_csv("clean_train_data.csv", encoding="utf-8")
    enge = dp.features_with_null_objects(df)
    print(enge)
    df, categorial_encoders = dp.transform_categorial_into_numeric(df, ["MSZoning",
                                                                     "LotShape", "LandContour", "LotConfig",
                                                                     "LandSlope", "Neighborhood", "Condition1", "Condition2",
                                                                     "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
                                                                     "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual",
                                                                     "ExterCond", "Foundation", "BsmtQual", "BsmtCond",
                                                                     "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
                                                                     "Heating", "HeatingQC", "CentralAir", "Electrical",
                                                                     "KitchenQual", "Functional", "GarageType",
                                                                     "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "SaleType", "SaleCondition"])
    print(df.info())
    print(categorial_encoders["MSZoning"].classes_)
    # PIPELINE
    df = dp.delete_outliers(df, 0, 400000)
    features_data = df.iloc[:, 1:71]
    target_data = df.iloc[:, 71:72]
    #find_outliers(target_data)
    print("FEATURES_DATA")
    print(features_data)
    print("TARGET_DATA")
    print(target_data)
    x_train, y_train = dp.split_data(features_data, target_data, test_size=0)
    #best_para = grid_search(x_train, y_train, 42, [400, 500, 600, 700], [0.4, 0.5, 0.6], [30, 40, 50, 60, 70], [True, False], ["absolute_error", "squared_error"])
    #print(best_para)
    best_rnd_forest = evaluate_random_forest(x_train, y_train)
    x_test = dp.load_data("test.csv", encoding="uft-8")
    dp.delete_features(x_test,
                    ["Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature", "Utilities", "Street", "LowQualFinSF", "GarageYrBlt"])
    x_test["LotFrontage"].replace(to_replace=np.nan, value=average, inplace=True)
    print(categorial_encoders)
    print("Feature with missing values:")
    for ele in dp.features_with_null_objects(x_test):
        print(ele, " \n")
    x_test.to_csv("clean_test_data.csv", encoding="utf-8")
    manual_test_data = dp.load_data("clean_test_data.csv", encoding="utf-8")

    """
"""
    print(manual_test_data)
    print(manual_test_data.info())
    x_test = transform_categorial_into_numeric(manual_test_data, ["MSZoning",
                                                                     "LotShape", "LandContour", "LotConfig",
                                                                     "LandSlope", "Neighborhood", "Condition1", "Condition2",
                                                                     "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
                                                                     "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual",
                                                                     "ExterCond", "Foundation", "BsmtQual", "BsmtCond",
                                                                     "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
                                                                     "Heating", "HeatingQC", "CentralAir", "Electrical",
                                                                     "KitchenQual", "Functional", "GarageType",
                                                                     "GarageFinish", "GarageQual", "GarageCond",
                                                          "PavedDrive", "SaleType", "SaleCondition"], categorial_encoders)
    selected_rows = x_test[x_test.isnull().any(axis=1)]
    s = x_test.stack(dropna=False)
    L = [list(x) for x in s.index[s.isna()]]
    print(L)
    print("NaN Values")
    print(selected_rows)
    y_pred = best_rnd_forest.predict(x_test)
    print(y_pred)



    #x_train, x_val, y_train, y_val = split_data(features_data, target_data, shuffle=True)
    #evaluate_gradient_boosting(x_train, y_train, x_val, y_val)
    """






