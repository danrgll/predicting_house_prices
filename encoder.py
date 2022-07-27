from sklearn.preprocessing import LabelEncoder
import numpy as np
from data_preparation import load_data
import pandas as pd
from sklearn.utils import column_or_1d


class Encoder(LabelEncoder):
    """Erweiterung von LabelEncoder um die Funktion fit_more, welche die Funktion fit ergänzt durch ein weiteres
     hinzufügen von Klassen ermöglicht nachdem fit schon ausgeführt wurde, was über die Funktion
     fit der Klasse LabelEncoder nicht möglich ist."""

    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self

    def fit_more(self, more_data):
        """fit further classes for new labels"""
        le = LabelEncoder()
        le.fit(more_data)
        for element in le.classes_:
            print(le.classes_)
            if element not in self.classes_:
                self.classes_ = np.append(self.classes_, [element])
                print(self.classes_)


def create_fit_encoder(para):
    """Erstellt einen Encoder und fitted die Features in liste para"""
    le = Encoder()
    le.fit(para)
    return le


def set_manual_encoder(list):
    """
    ermöglicht manuelle Erstellung von Encodern. Nützlich wenn Reihenfolge der Werte relevant sind.
    :param list: [["Key", [features]], [...], ]
    :return: encoders zusammengefasst als dict und über ihre Keys ansprechbar
    >>> encoders = set_manual_encoder([["MSZoning", ["A", "C(all)", "C", "FV", "I", "RH", "RL", "RP", "RM", "NI"]]])
    >>> encoder = encoders["MSZoning"]
    >>> d = {'MSZoning': ["A", "C", "NI"]}
    >>> df = pd.DataFrame(data=d)
    >>> df["MSZoning"] = encoder.transform(df["MSZoning"])
    >>> values = df['MSZoning'].values.tolist()
    >>> print(values)
    [0, 2, 9]
    """
    dict_encoder = dict()
    for feature in list:
        dict_encoder[feature[0]] = create_fit_encoder(feature[1])
    return dict_encoder


def transform_categorial_into_numeric(df, transform_features, encoders=None):
    """transform data and safe each label encoder with a simple LabelEncoder
    >>> df = load_data("train.csv")
    >>> df, categorial_encoders = transform_categorial_into_numeric(df, ["MSSubClass", "MSZoning", "Street", "Alley"])
    >>> df["Street"][0]
    1
    """
    if encoders is None:
        dict_encoders = {}
        for feature in transform_features:
            le = Encoder()
            df[feature] = le.fit_transform(df[feature])
            dict_encoders[feature] = le
        return df, dict_encoders
    else:
        for feature in transform_features:
            print(feature)
            for (key, element) in encoders.items():
                print(f"{key}: {element}")
            le = encoders[feature]
            print(encoders[feature])
            le.fit_more(df[feature])
            encoders[feature] = le
            le = encoders[feature]
            df[feature] = le.transform(df[feature])
        return df



if __name__ == '__main__':
    #le = Encoder()
    #le.fit([0, 1, 2, 3])
    #le.fit_more([4])
    #print(le.classes_)
    a = set_manual_encoder([["MSZoning", ["A", "C(all)", "C", "FV", "I", "RH", "RL", "RP", "RM", "NI"]]])
    print(a)
    # a["MSZoning"]

