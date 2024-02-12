# CUSTOMER-CHURN-PREDICTION
 The code imports data, examines its structure, engineers features, cross-validates models like Logistic Regression, TensorFlow, XGBoost, LightGBM, CatBoost, and evaluates their performance based on ROC AUC score.



Importing Libraries and Data
The initial section of the code imports necessary libraries and sets up the environment for accessing data from Kaggle. It downloads the required dataset files and prepares the working directories.

Introduction
Here, the code provides an introduction to the problem statement of the competition, which is binary classification with the Bank Churn Dataset. It outlines the columns present in the dataset, including the target variable ('Exited'), and briefly describes their meanings.

Loading Libraries and Datasets
In this section, the code imports essential libraries for data manipulation, visualization, and modeling, such as pandas, numpy, matplotlib, seaborn, TensorFlow, Optuna, etc. It then loads the training and test datasets into pandas DataFrames for further analysis and model training.

Descriptive Statistics
The code performs exploratory data analysis by displaying the first few rows of the training and test datasets and generating descriptive statistics for each column. This helps in understanding the structure and characteristics of the data, including data types, missing values, unique values, and summary statistics.

Preparation
In this part, the code prepares for feature engineering and model building by defining helper functions and setting up variables like X (features) and y (target) for model training. It also initializes parameters such as random seeds and cross-validation strategies for consistent and reliable model evaluation.

Feature Engineering
Feature engineering is crucial for improving model performance. Here, the code defines several functions and transformers for generating new features, transforming existing ones, and preparing the data for modeling. Techniques like nullifying, rounding, encoding categorical variables, and creating new features are applied to enhance the predictive power of the model.

Model Cross Validation
This section contains functions for cross-validating different machine learning models. It defines pipelines for each model, including preprocessing steps, model instantiation, and evaluation using cross-validation. The models evaluated include Logistic Regression, TensorFlow neural network, XGBoost, LightGBM, and CatBoost. Performance metrics like ROC AUC score are used for evaluation.

Logistic Regression
The code defines a logistic regression model pipeline, including preprocessing steps like rounding, feature generation, encoding, and standardization. It then evaluates the model's performance using cross-validation and prints the validation score.

TensorFlow
A custom TensorFlow neural network model is defined using the Keras API. The model architecture includes several dense layers with batch normalization and LeakyReLU activation functions. The pipeline preprocesses the data and then trains the TensorFlow model using cross-validation.

XGBoost
An XGBoost classifier pipeline is created with preprocessing steps like rounding, feature generation, vectorization, and encoding. Hyperparameters for XGBoost are optimized using Optuna's hyperparameter search. The model is then evaluated using cross-validation, and the performance is reported.

LightGBM
Similarly, a LightGBM classifier pipeline is defined with preprocessing steps like rounding, feature generation, vectorization, and encoding. Hyperparameters for LightGBM are optimized using Optuna, and the model is evaluated using cross-validation.

CatBoost
Two CatBoost classifier pipelines are defined, one with a traditional bootstrapping method and another with a Bayesian bootstrapping method. Both pipelines preprocess the data, including rounding, feature generation, vectorization, and encoding. The models are evaluated using cross-validation.

Each section contributes to the overall process of loading data, preparing it for modeling, engineering useful features, building and evaluating machine learning models, and optimizing hyperparameters for better performance. The code demonstrates a systematic approach to solving a binary classification problem using various machine learning techniques.







![image](https://github.com/hannu-shaik/CUSTOMER-CHURN-PREDICTION/assets/140539636/1429ee39-90b4-42a0-b994-6a3a2c71b5da)





