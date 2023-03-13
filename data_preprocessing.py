import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt


def load_data(path, encoding=None, seperator=None):
    if encoding is None:
        if seperator is None:
            return pd.read_csv(path)
        else:
            return pd.read_csv(path, sep=seperator)
    else:
        if seperator is None:
            return pd.read_csv(path, encoding=encoding)
        else:
            return pd.read_csv(path, encoding=encoding, sep=seperator)



def split_data(x_data, y_data, test_size=0.2, shuffle=False):
    if test_size == 0:
        x_train, y_train = x_data, y_data
        return x_train, y_train
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=test_size, shuffle=shuffle)
    return x_train, x_val, y_train, y_val


def delete_features(df, features: list):
    for feature in features:
        del df[feature]


def features_with_null_objects(df):
    """returns list with columns of the null objects"""
    null_features = []
    null_data = df[df.isnull().any(axis=1)]
    # null_data.to_csv('null_objects.csv')
    for column in df:
        for element in df[column]:
            if str(element) == "nan":
                if column not in null_features:
                    null_features.append(column)
    return null_features, null_data

# ToDo: identifiziere NullObjekte


def location_of_null_objects(df):
    null_data = df[df.isnull().any(axis=1)]
    print(null_data)
    locations = []
    for row in null_data:
        print("hello")
        print(row)
        index = row.iloc[0]["Id"]
        for column in row:
            if row[column] == "nan":
                locations.append([row[index], column])
    return locations


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

