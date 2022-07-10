import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from numpy import nan


def load_data(path):
    return pd.read_csv(path)


def split_data(x_data, y_data, test_size=0.2, shuffle=False):
    if test_size == 0:
        x_train, y_train = x_data, y_data
        return x_train, y_train
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=test_size, shuffle=shuffle)
    return x_train, x_val, y_train, y_val


def transform_categorial_into_numeric(df, transform_features, encoders=None):
    """transform data and safe each label encoder
    >>> df = load_data("train.csv")
    >>> df, categorial_encoders = transform_categorial_into_numeric(df, ["MSSubClass", "MSZoning", "Street", "Alley"])
    >>> df["Street"][0]
    1
    """
    if encoders is None:
        dict_encoders = {}
        for feature in transform_features:
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature])
            dict_encoders[feature] = le
        return df, dict_encoders
    else:
        for feature in transform_features:
            print(feature)
            le = encoders[feature]
            df[feature] = le.transform(df[feature])
        return df


def delete_features(df, features: list):
    for feature in features:
        del df[feature]


def features_with_null_objects(df):
    null_features = []
    for column in df:
        for element in df[column]:
            if str(element) == "nan":
                if column not in null_features:
                    null_features.append(column)
    return null_features


def replace_null_with_average_number(df, features):
    for feature in features:
        sum = 0
        counter = 0
        for element in df[feature]:
            if str(element) != "nan":
                sum += float(element)
                counter += 1
        average = sum // counter
        df[feature].replace(to_replace=np.nan, value=average, inplace=True)
    return average


def find_outliers(y_data):
    x_data = [*range(0, len(y_data), 1)]
    print(y_data, (len(y_data)), x_data, len(x_data))
    plt.scatter(x_data, y_data, 5)
    plt.show()


def delete_outliers(df, min_border, max_border):
    df = df[df.SalePrice < max_border]
    df = df[df.SalePrice > min_border]
    return df


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


if __name__ == '__main__':
    df = load_data("train.csv")
    # CLEAN DATA
    # drop out columns that have more than 50% null objects over all data points
    delete_features(df, ["Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature", "Utilities", "Street", "LowQualFinSF"])
    average = replace_null_with_average_number(df, ["LotFrontage"])
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
    df.to_csv("clean_train_data.csv", sep='\t')
    enge = features_with_null_objects(df)
    print(enge)
    df, categorial_encoders = transform_categorial_into_numeric(df, ["MSZoning",
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
    # PIPELINE
    df = delete_outliers(df, 0, 400000)
    features_data = df.iloc[:, 1:72]
    target_data = df.iloc[:, 72:73]
    #find_outliers(target_data)
    print("FEATURES_DATA")
    print(features_data)
    print("TARGET_DATA")
    print(target_data)
    x_train, y_train = split_data(features_data, target_data, test_size=0)
    #best_para = grid_search(x_train, y_train, 42, [400, 500, 600, 700], [0.4, 0.5, 0.6], [30, 40, 50, 60, 70], [True, False], ["absolute_error", "squared_error"])
    #print(best_para)
    best_rnd_forest = evaluate_random_forest(x_train, y_train)
    x_test = load_data("test.csv")
    delete_features(x_test,
                    ["Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature", "Utilities", "Street", "LowQualFinSF"])
    df["LotFrontage"].replace(to_replace=np.nan, value=average, inplace=True)
    df.to_csv("clean_test_data.csv", sep='\t')
    print(categorial_encoders)
    x_test = transform_categorial_into_numeric(x_test, ["MSZoning",
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
    df.to_csv("clean_test_data.csv", sep='\t')
    y_pred = best_rnd_forest.predict(x_test)
    print(y_pred)


    #x_train, x_val, y_train, y_val = split_data(features_data, target_data, shuffle=True)
    #evaluate_gradient_boosting(x_train, y_train, x_val, y_val)








