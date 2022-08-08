import numpy as np
# Parameter Search Room from RandomForest Model Grid

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=10, stop=600, num=10)]
# Number of features to consider at every split
max_features = ['sqrt']
# Maximum number of levels in tree
max_depth = [3, 10, 15, 20, 30, 40]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [3]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1]
# Method of selecting samples for training each tree
bootstrap = [False]
RF_PARA_GRID = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Parameter Search Room from Gradient Boosting Grid
# loss function die genutzt werden soll um den Fehler zu berechnen
loss = ["squared_error", "absolute_error", "huber"]
# Lernrate
learningrate = [0.08, 0.1, 0.2, 0.3]
# Anzahl schwacher Lerner die aufeinander aufbauen. The number of Boosting stages to perform.
n_estimators = [60, 80, 100, 150, 200, 250, 300, 400, 500, 700, 1000]
# Anzahl Datenpunkte in % die fürs Fitten der einzelnen Lerner verwendet werden soll,
# weniger als 100% nennt sich STochhastic Gradient boosting
subsample = [1.0, 0.97, 0.9, 0.95]
min_samples_split = [2, 3, 4]
min_samples_leaf = [1, 2, 3]
max_depth = [2, 3, 5, 7, 9, None]
# The number of features to consider when looking for the best split:
max_features = ["auto", "sqrt", "log2", "None"]
GDB_PARA_GRID = {"loss": loss,
               "learning_rate": learningrate,
               "n_estimators": n_estimators,
               "subsample": subsample,
               "min_samples_split": min_samples_split,
               "min_samples_leaf": min_samples_leaf,
               "max_depth": max_depth,
               "max_features": max_features}