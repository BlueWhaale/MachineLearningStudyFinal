import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import sys
import time
import warnings
import joblib

import copy

import math
import random
import pickle

import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance, plot_tree
from xgboost.callback import EarlyStopping

from sklearn.metrics import explained_variance_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import dump_svmlight_file
from pandas.plotting import scatter_matrix
from sklearn.manifold import TSNE

import scipy as sp
import scipy.stats


def multi_model_predict_result(multi_model, x):
    n_model = len(multi_model)
    n_data = x.shape[0]

    mu = np.zeros(n_data)
    std = np.zeros(n_data)

    mu_vec = np.zeros((n_data, n_model))

    for i in range(n_model):
        mu_temp = multi_model[i].predict_proba(x)
        mu_vec[:,i] = mu_temp[:,1]

    for k in range(n_data):
        mu[k] = np.mean(mu_vec[k,:])
        std[k] = np.std(mu_vec[k,:])

    mu = np.array(np.round(mu), dtype=np.int32)

    return mu, std, mu_vec


def multi_model_predict(multi_model, x):
    n_model = len(multi_model)
    n_data = x.shape[0]

    mu = np.zeros(n_data)
    std = np.zeros(n_data)

    y_vec = np.zeros((n_data, n_model))

    for i in range(n_model):
        y_temp = multi_model[i].predict_proba(x)
        y_vec[:, i] = y_temp[:, 1]

    for k in range(n_data):
        mu[k] = np.mean(y_vec[k, :])
        std[k] = np.std(y_vec[k, :])

    mu = np.array(np.round(mu), dtype=np.int32)

    return mu, std


# NOTE: for debugging
is_verbos = False
is_show_graph = True

file_name = "heart_final_train.csv"
# "age", "sex", "cp", "trtbps", "chol", "fbs", "restecg", "thalachh", "exng", "oldpeak", "slp", "caa", "thall", "output"
df = pd.read_csv(file_name, engine="python", encoding="utf-8-sig")

df.dropna(axis=0, inplace=True)
categorical_features = ["cp", "restecg", "slp", "thall"]
df_encoded = pd.get_dummies(df, columns=categorical_features, prefix=categorical_features)
boolean_columns = df_encoded.select_dtypes(include='bool').columns
df_encoded[boolean_columns] = df_encoded[boolean_columns].astype(int)

data_cnt = 40
n_ensemble = 7
df_init_40 = df_encoded.groupby('output', group_keys=False).apply(lambda x: x.sample(min(len(x), data_cnt // 2)))
df_else = df_encoded.drop(df_init_40.index)
df_add = pd.DataFrame(columns=df_encoded.columns).astype(df_encoded.dtypes)

all_accuracy = []

train_start_time = time.time()

while True:
    list_model = []
    list_X_train = []
    list_y_train = []
    list_fi_xgboost = []
    list_fi_shap = []

    df_data = pd.concat([df_init_40, df_add])

    if len(df_data) > 120:
        break

    if is_verbos:
        print(f"data count: {len(df_data)}")

    df_X = df_data.loc[:, ["age", "sex", "trtbps", "chol", "fbs", "thalachh", "exng", "oldpeak", "caa",
                           "cp_0", "cp_1", "cp_2", "cp_3", "restecg_0", "restecg_1", "restecg_2",
                           "slp_0", "slp_1", "slp_2", "thall_0", "thall_1", "thall_2", "thall_3"]]
    df_y = df_data.loc[:, "output"]

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

        X_train = X_train_init.iloc[train_indices]
        y_train = y_train_init.iloc[train_indices]

        list_X_train.append(X_train)
        list_y_train.append(y_train)

    start = time.time()

    for i in range(n_ensemble):
        X_train = list_X_train[i]
        y_train = list_y_train[i]

        model = XGBClassifier(
            learning_rate=0.03,
            n_estimators=1000,
            max_depth=6,
            min_child_weight=3,
            gamma=0.2,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.05,
            reg_lambda=0.1,
            objective='binary:logistic',
            nthread=4,
            scale_pos_weight=1
        )

        # Split X_train and y_train into training and validation sets
        X_train_model, X_val, y_train_model, y_val = train_test_split(
            X_train, y_train, test_size=0.1, stratify=y_train
        )

        model.fit(
            X_train_model,
            y_train_model,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        list_model.append(copy.deepcopy(model))

    end = time.time()
    if is_verbos:
        print(f"model train time : {end - start:.5f} sec")

    list_train_accuracy = []

    for i in range(n_ensemble):
        model = list_model[i]

        X_train = list_X_train[i]
        y_train = list_y_train[i]

        y_est = model.predict(X_train)
        n_train = len(y_train)
        n_correct = np.size(np.where((y_train - y_est) == 0))

        list_train_accuracy.append(n_correct / n_train)

    list_test_accuracy = []

    for i in range(n_ensemble):
        model = list_model[i]
        
        y_est = model.predict(X_test)
        n_test = len(y_test)
        n_correct = np.size(np.where((y_test - y_est) == 0))

        list_test_accuracy.append(n_correct / n_test)

    df_X_test = df_else.loc[:, ["age", "sex", "trtbps", "chol", "fbs", "thalachh", "exng", "oldpeak", "caa",
                                "cp_0", "cp_1", "cp_2", "cp_3", "restecg_0", "restecg_1", "restecg_2",
                                "slp_0", "slp_1", "slp_2", "thall_0", "thall_1", "thall_2", "thall_3"]]
    df_y_test = df_else.loc[:, "output"]

    y_est, y_est_std, y_est_vec = multi_model_predict_result(list_model, df_X_test)

    df_result = pd.DataFrame({'true': df_y_test, 'est': y_est})
    for i in range(y_est_vec.shape[1]):
        df_result[f'est_{i}'] = y_est_vec[:, i]

    df_result['est_std'] = y_est_std

    threshold_cnt = 5
    df_high_std = df_result.sort_values(by='est_std', ascending=False).head(threshold_cnt)
    df_add = pd.concat([df_add, df_encoded.loc[df_high_std.index]])

    df_else = df_else.drop(index=df_high_std.index)

    n_test = len(df_y_test)
    n_correct = np.size(np.where((df_y_test - y_est) == 0))

    all_accuracy.append(n_correct / n_test)

    if is_verbos:
        print(f'<< Test Results from Ensemble Model >>')
        print('accuracy(individual)=', list_test_accuracy)
        print('accuracy(ensemble)=%g\n' % (n_correct / n_test))

train_end_time = time.time()

print(f"total train time : {train_end_time - train_start_time:.5f} sec")
print(f"final test accuracy : {all_accuracy[-1]}")

if is_show_graph:
    plt.plot(np.arange(40, 40 + 5 * len(all_accuracy), 5), all_accuracy, marker='o')
    plt.xlabel("data count")
    plt.ylabel("accuracy")
    plt.title("ensemble accuracy with sampling")
    plt.grid(True)
    plt.show()
