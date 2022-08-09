import numpy as np
# Parameter Search Room from RandomForest Model Grid

# Number of trees in random forest
# lin_space return evenly spaced numbers over a specified interval.
n_estimators = [int(x) for x in np.linspace(start=1, stop=75, num=50)]
# Number of features to consider at every split
max_features = ['sqrt']
# Maximum number of levels in tree
max_depth = [100]
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
loss = ["squared_error"]
# Lernrate
learningrate = [0.05]
# Anzahl schwacher Lerner die aufeinander aufbauen. The number of Boosting stages to perform.
n_estimators = [int(x) for x in np.linspace(start=10, stop=1000, num=50)]
# Anzahl Datenpunkte in % die fürs Fitten der einzelnen Lerner verwendet werden soll,
# weniger als 100% nennt sich STochhastic Gradient boosting
subsample = [1, 0.9, 0.8, 0.95]
min_samples_split = [3]
min_samples_leaf = [2]
max_depth = [2]
# The number of features to consider when looking for the best split:
max_features = ["auto"]
GDB_PARA_GRID = {"loss": loss,
               "learning_rate": learningrate,
               "n_estimators": n_estimators,
               "subsample": subsample,
               "min_samples_split": min_samples_split,
               "min_samples_leaf": min_samples_leaf,
               "max_depth": max_depth,
               "max_features": max_features}



