import data_preparation as dp
import pickle
import numpy as np
import pandas as pd
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


if __name__ == '__main__':
    df = clean_house_prices_data()
    df_train, df_test = encoding_data(df)
    print(df_train.info())
    x_train = df_train.iloc[:, 0:70]
    y_train = df_train.iloc[:, 70:71]
    # para = search_hyperparamter_gradientboosting(x_train, y_train, "random", GDB_PARA_GRID)
    para = search_hyperparameter_random_forest(x_train, y_train, "grid", RF_PARA_GRID)
    # print(para)
    print(y_train)
    # Parameter der Modelle die am vielversprechensten waren
    para_rf = {"n_estimators": 600, "min_samples_split": 3, "min_samples_leaf": 1, "max_features": "sqrt",
            "max_depth": 30, "bootstrap": False}
    para_gdb = {'subsample': 0.9, 'n_estimators': 400, 'min_samples_split': 3, 'min_samples_leaf': 2,
            'max_features': 'auto',
            'max_depth': 2, 'loss': 'squared_error', 'learning_rate': 0.08}
    # fit_and_val_random_forest(x_train, y_train, para_rf)
    # fit_and_val_gradient_boosting(x_train, y_train, para_gdb)
    # fit_and_val_ensemble_model(x_train, y_train)
    # MEA: 20847.467290285473
