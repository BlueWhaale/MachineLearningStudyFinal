import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os
import time
import warnings
import joblib

import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import math
import random
import pickle

import shap
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance, plot_tree

from sklearn.metrics import explained_variance_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import dump_svmlight_file
from pandas.plotting import scatter_matrix
from sklearn.manifold import TSNE

import scipy as sp
import scipy.stats

def multi_model_predict(multi_model, x):
    n_model = len(multi_model)
    n_data = x.shape[0]

    mu = np.zeros(n_data)
    std = np.zeros(n_data)

    y_vec = np.zeros((n_data, n_model))

    for i in range(n_model):
        #y_temp = multi_model[i].predict(x)
        #y_vec[:,i] = y_temp

        y_temp = multi_model[i].predict_proba(x) # Class 대신 Probability를 얻을 수도 있다.
        y_vec[:,i] = y_temp[:,1]

    for k in range(n_data):
        mu[k] = np.mean(y_vec[k,:])
        std[k] = np.std(y_vec[k,:])

    mu = np.array(np.round(mu), dtype=np.int32)

    return mu, std

file_name = "heart_final_train.csv"
# "age", "sex", "cp", "trtbps", "chol", "fbs", "restecg", "thalachh", "exng", "oldpeak", "slp", "caa", "thall", "output",
df = pd.read_csv(file_name, engine="python", encoding="utf-8-sig")

df.dropna(axis=0, inplace=True)
categorical_features = ["cp", "restecg", "slp", "thall"]
df_encoded = pd.get_dummies(df, columns=categorical_features, prefix=categorical_features)
boolean_columns = df_encoded.select_dtypes(include='bool').columns
df_encoded[boolean_columns] = df_encoded[boolean_columns].astype(int)

df_numerical = df_encoded.select_dtypes(include=["number"])
df_numerical.corr()["output"].sort_values(ascending=False)

n_ensemble = 7

list_model = []
list_X_train = []
list_y_train = []
list_fi_xgboost = []
list_fi_shap = []

df_X  = df_encoded.loc[:, ["age", "sex", "trtbps", "chol", "fbs", "thalachh", "exng", "oldpeak", "caa", "cp_0", "cp_1", "cp_2", "cp_3", "restecg_0", "restecg_1", "restecg_2", "slp_0", "slp_1", "slp_2", "thall_0", "thall_1", "thall_2", "thall_3"]]
df_y= df_encoded.loc[:, "output"]

X_train_init, X_test, y_train_init, y_test = train_test_split(df_X, df_y, test_size=0.3)

indices = list(range(len(X_train_init)))
random.shuffle(indices)
split_indices = np.array_split(indices, n_ensemble)

for i in range(n_ensemble):
    test_indices = split_indices[i]
    train_indices = []
    for j in range(n_ensemble):
        if j != i:
            train_indices.extend(split_indices[j])

    X_test_temp = X_train_init.iloc[test_indices]
    y_test_temp = y_train_init.iloc[test_indices]

    X_train = X_train_init.iloc[train_indices]
    y_train = y_train_init.iloc[train_indices]

    print("* train_test_split:", i)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    list_X_train.append(X_train)
    list_y_train.append(y_train)

start = time.time()

for i in range(n_ensemble):
    print("* model.fit:", i)

    model = XGBClassifier(
            learning_rate =0.1,
            n_estimators=1000,
            max_depth=4,
            min_child_weight=6+2,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.005,
            objective= 'binary:logistic',
            nthread=4,
            scale_pos_weight=1)

    model.fit(list_X_train[i], list_y_train[i])

    list_model.append(copy.deepcopy(model))

end = time.time()
print(f"학습에 걸린 시간 : {end - start:.5f} sec")

list_train_accuracy = []

for i in range(n_ensemble):

    model = list_model[i]

    X_train = list_X_train[i]
    y_train = list_y_train[i]

    y_est = model.predict(X_train)
    n_train = len(y_train)
    n_correct = np.size(np.where((y_train - y_est) == 0))

    list_train_accuracy.append(n_correct/n_train)

    print(f'<< Training Results: {i} >>')
    print('accuracy=%g' % (n_correct/n_train))

    # confusion_mat = confusion_matrix(y_train, y_est)
    # print('confusion matrix=')
    # print(confusion_mat)
    # plt.figure(figsize=(7,5))
    # #sns.heatmap(confusion_mat, annot=True, cmap='Blues')
    # sns.heatmap(confusion_mat, annot=True, fmt="d", cmap='Blues')
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # plt.show()

list_test_accuracy = []

for i in range(n_ensemble):

    model = list_model[i]

    X_test = X_test
    y_test = y_test

    y_est = model.predict(X_test)
    n_test = len(y_test)
    n_correct = np.size(np.where((y_test - y_est) == 0))

    list_test_accuracy.append(n_correct/n_test)

    print(f'<< Test Results: {i} >>')
    print('accuracy=%g' % (n_correct/n_test))

    confusion_mat = confusion_matrix(y_test, y_est)
    print('confusion matrix=')
    # print(confusion_mat)

    plt.figure(figsize=(7,5))
    #sns.heatmap(confusion_mat, annot=True, cmap='Blues')
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

for i in range(n_ensemble):
    model = list_model[i]

    # Get feature importance from the trained model
    feature_importances = model.feature_importances_

    list_fi_xgboost.append(feature_importances)

    # Create a DataFrame for visualization
    importance_df = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    print(f'<< Feature Importance form XGBoost: {i} >>')

    # Plot the feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.title('Feature Importance from XGBoost')
    plt.gca().invert_yaxis()  # Highest importance on top
    plt.show()


# SHAP value analysis
# explainer = shap.Explainer(model, X_train)
# shap_values_train = explainer.shap_values(X_train) # 전체 Data 분석에 활용 됨
# explanation_object_train = explainer(X_train) # 개별 Data 분석에 활용 됨

# Feature importance from SHAP values (average absolute SHAP value)

list_explanation_object = []

for i in range(n_ensemble):
    model = list_model[i]
    #X_train = list_X_train[i]

    # SHAP Explainer
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer.shap_values(X_test)
    explanation_object = explainer(X_test)

    list_explanation_object.append(explanation_object)

    # Feature importance from SHAP values (average absolute SHAP value)
    feature_importance_shap = np.abs(shap_values).mean(axis=0)

    list_fi_shap.append(feature_importance_shap)

    # Create a DataFrame for visualization
    importance_df_shap = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importance_shap
    }).sort_values(by='Importance', ascending=False)

    print(f'<< Feature Importance from SHAP: {i} >>')

    # Plot the feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df_shap['Feature'], importance_df_shap['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.title('Feature Importance from SHAP')
    plt.gca().invert_yaxis()  # Highest importance on top
    plt.show()

# y_est = model.predict(X_test)
y_est, y_est_std = multi_model_predict(list_model, X_test)

n_test = len(y_test)
n_correct = np.size(np.where((y_test - y_est) == 0))

print(f'<< Test Results from Ensemble Model >>')
print('accuracy(ensemble)=%g' % (n_correct/n_test))
print('accuracy(individual)=',list_test_accuracy)

confusion_mat = confusion_matrix(y_test, y_est)
print('confusion matrix=')
# print(confusion_mat)

plt.figure(figsize=(7,5))
#sns.heatmap(confusion_mat, annot=True, cmap='Blues')
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

plt.figure(figsize=(7, 5))
sns.histplot(y_est_std, bins=25)
plt.title('Histogram of Prediction Standard Deviation (y_est_std)')
plt.xlabel('Standard Deviation')
plt.ylabel('Frequency')
plt.show()