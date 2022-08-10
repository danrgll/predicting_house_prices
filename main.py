import data_preparation as dp
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from encoder import transform_categorial_into_numeric, transform_categorial_into_target
from hyperparamter_search import search_hyperparameter_random_forest, search_hyperparamter_gradientboosting
from evaluation import fit_and_val_gradient_boosting, fit_and_val_random_forest, fit_and_val_ensemble_model
from settings import RF_PARA_GRID, GDB_PARA_GRID


def clean_house_prices_data():
    df_train = dp.load_data("train.csv", encoding="utf-8")
    df_train.drop(
        labels=[17, 90, 102, 156, 182, 259, 342, 362, 371, 392, 520, 532, 533, 552, 646, 705, 736, 749, 778, 868,
                894,
                897, 984, 1000, 1011, 1035, 1045, 1048, 1049, 1090, 1179, 1216, 1218, 1230, 1321, 1412, 553,
                1232, 39, 948, 332, 1379], axis=0)
    # target_data = df_train.iloc[:, 80:81]
    # del df_train["SalePrice"]
    df_test = dp.load_data("test.csv", encoding="utf-8")
    print(df_test.info())
    df_test.insert(2, "SalePrice", [1] * df_test.shape[0], True)
    df = pd.concat([df_train, df_test], keys=[0, 1])
    # CLEAN DATA
    # drop out columns that have more than 50% null objects over all data points or correlate with other features
    dp.delete_features(df,
                       ["Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature", "Utilities", "Street",
                        "LowQualFinSF",
                        "GarageYrBlt", "BsmtFinType2"])
    average = dp.replace_null_with_average_number(df, ["LotFrontage"])
    df["MasVnrType"].replace(to_replace=np.nan, value="None", inplace=True)
    df["MasVnrArea"].replace(to_replace=np.nan, value=0.0, inplace=True)
    df["BsmtFullBath"].replace(to_replace=np.nan, value=0.0, inplace=True)
    df["BsmtHalfBath"].replace(to_replace=np.nan, value=0.0, inplace=True)
    df["BsmtFinSF1"].replace(to_replace=np.nan, value=0.0, inplace=True)
    df["BsmtFinSF2"].replace(to_replace=np.nan, value=0.0, inplace=True)
    df["BsmtUnfSF"].replace(to_replace=np.nan, value=0.0, inplace=True)
    df["TotalBsmtSF"].replace(to_replace=np.nan, value=0.0, inplace=True)
    df[['GarageQual', 'GarageCond', "GarageType", "GarageFinish", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1"]] = df[
        ["GarageQual", "GarageCond", "GarageType", "GarageFinish", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1"]].fillna('NI')
    # drop rows which have missing values in choosen columns
    # ToDo Missing Values aus Test Set werden weggeworfen shit
    df_train, df_test = df.xs(0), df.xs(1)
    df_test.to_csv("modified_test.csv", encoding="utf-8")
    # feature, data = dp.features_with_null_objects(df_test)
    # data.to_csv("missing_data.csv", encoding="utf-8")
    # data = dp.load_data("clean_test.csv", encoding="utf-8")
    # print(data.info())
    manual_changed_data = dp.load_data("manual_data.csv", encoding="utf-8", seperator=";")
    dp.delete_features(manual_changed_data, ["Unnamed: 0"])
    print(manual_changed_data.info())
    manual_changed_data.set_index('Id')
    df_test.set_index('Id')
    print("test:")
    print(df_test.info())
    df_test.loc[manual_changed_data.index, :] = manual_changed_data[:] # ändert die Zeilen die manuel angepasst wurden aus missing_data.csv
    print("Wichtig Wichtig")
    print(df_test.info())
    df_test = dp.load_data("final.csv", encoding="utf-8", seperator=";")
    dp.delete_features(df_test, ["Unnamed: 0"])
    df = pd.concat([df_train, df_test], keys=[0, 1])
    # print(feature)
    # print(data.info())
    df = df.dropna(axis=0, subset=["MSZoning", "Exterior1st", "Exterior2nd", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
                                   "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "KitchenQual", "Functional", "GarageCars",
                                   "GarageArea", "SaleType", "BsmtQual", "BsmtExposure"])
    return df


def encoding_data(df):
    # transform categorial variables to a target(Encoding) for all possibilieties
    df_train, df_test = df.xs(0), df.xs(1)
    # df_train["SalePrice"].to_csv("check4", encoding="utf-8")
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


def save_models(x_train, y_train, para_rf, para_gdb):
    # save final model fit
    rf = RandomForestRegressor(**para_rf)
    rf.fit(x_train, y_train.values.ravel())
    pickle.dump(rf, open("rf_model.sav", "wb"))

    gdb = GradientBoostingRegressor(**para_gdb)
    gdb.fit(x_train, y_train.values.ravel())
    pickle.dump(gdb, open("gdb_model.pkl", "wb"))

    rf = RandomForestRegressor(**para_rf)
    gdb = GradientBoostingRegressor(**para_gdb)
    ensemble_model = VotingRegressor([("rf", rf), ("grdb", gdb)], n_jobs=-1)
    ensemble_model.fit(x_train, y_train.values.ravel())
    pickle.dump(ensemble_model, open("ensemble_model.pkl", "wb"))


def final_predict(df_test):
    print(df_test)
    ensemble_model = pickle.load(open("ensemble_model.pkl", 'rb'))
    id = df_test["Id"]
    dp.delete_features(df_test, ["Id"])
    print(df_test.info())
    prediction = ensemble_model.predict(df_test)
    # id = list(range(1461, 2920))
    print(len(id))
    print(len(prediction))
    d = {'Id': id, 'SalePrice': prediction}
    df = pd.DataFrame(data=d)
    df.to_csv("submit.csv", index=False)


if __name__ == '__main__':
    df = clean_house_prices_data()
    df_train, df_test = encoding_data(df)
    print("TestTest")
    print(df_test.info())
    dp.delete_features(df_train,
                       ["Id"])
    print("InfoInfo")
    print(df_train.info())
    x_train = df_train.iloc[:, 0:69]
    print("kddkdkddk")
    print(x_train.info())
    y_train = df_train.iloc[:, 69:70]
    # para = search_hyperparamter_gradientboosting(x_train, y_train, "grid", GDB_PARA_GRID, plot_training=False)
    # rf_model = search_hyperparameter_random_forest(x_train, y_train, "grid", RF_PARA_GRID, plot_training=True)
    # print(para)
    print(y_train)
    # Parameter der Modelle die am vielversprechensten waren
    para_rf = {"n_estimators": 50, "min_samples_split": 3, "min_samples_leaf": 1, "max_features": "sqrt",
            "max_depth": None, "bootstrap": False}
    para_gdb = {'subsample': 0.95, 'n_estimators': 500, 'min_samples_split': 3, 'min_samples_leaf': 2,
            'max_features': 'auto',
            'max_depth': 2, 'loss': 'squared_error', 'learning_rate': 0.05}
    # fit_and_val_random_forest(x_train, y_train, para_rf)
    # fit_and_val_gradient_boosting(x_train, y_train, para_gdb)
    # fit_and_val_ensemble_model(x_train, y_train, para_rf, para_gdb)

    # final model fit
    # save_models(x_train, y_train, para_rf, para_gdb)
    final_predict(df_test)

