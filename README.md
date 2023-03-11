# Ensemble Regression Model using GradientBoosting and RandomForest 

This project is about analyzing and predicting the sales prices of residential homes using the Ames Housing Dataset, which is a popular dataset in the field of machine learning and data science. The goal of this project is to develop a regression model that accurately predicts the sale price of homes based on various features such as the number of bedrooms, bathrooms, square footage, etc.

## Dataset Overview

The Ames Housing Dataset contains a total of 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa. The dataset was compiled by Dean De Cock for use in data science education and research, and it has become a popular choice for practicing data scientists due to its rich set of features and complex relationships between them.
Data Cleaning and Preprocessing

## Data Cleaning and Preprocessing

Before building the regression model, the dataset needed to be cleaned and preprocessed to remove missing values, handle categorical variables, and engineer new features. This involved performing exploratory data analysis (EDA) to identify patterns and trends in the data, imputing missing values using appropriate techniques, and encoding categorical variables using one-hot encoding or label encoding.



## Model Development
For this project, we will be using two popular ensemble regression techniques, Gradient Boosting and Random Forest, to develop our regression model. Both of these algorithms have shown to be effective in handling complex data with non-linear relationships.

### Gradient Boosting
Gradient Boosting is a type of boosting algorithm that builds an ensemble of weak decision trees and sequentially improves upon them by minimizing the loss function. It is known for its ability to handle high-dimensional data and produce accurate predictions.

### Random Forest
Random Forest is a type of bagging algorithm that also builds an ensemble of decision trees, but in this case, each tree is trained on a random subset of the features. This helps to reduce the variance and improve the accuracy of the predictions.


### Ensemble
After building and also fine-tuning the Hyperparameter of the individual Gradient Boosting and Random Forest models, we decided to combine them using the VotingRegressor ensemble technique. The VotingRegressor combines the predictions of multiple regression models to produce a final prediction that is more accurate than any of the individual models.

In our ensemble, we assigned equal weights to the Gradient Boosting and Random Forest models, and the final prediction was based on the average of their predicted values. This technique often leads to better performance and increased stability, as it takes advantage of the strengths of both models and reduces the impact of their weaknesses.

## Evaluation


