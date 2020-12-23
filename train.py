# -*- coding:utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import catboost as cb
import pandas as pd
import numpy as np
import dataset as ds
import joblib
import time


def show_acc(a, b, tip):
    acc = a.ravel() == b.ravel()
    print(tip + "Acc： {:.6f}%".format(float(acc.sum()) / a.size * 100))


@ds.time_cost
def train():
    x, y = ds.load_data()

    print('Dividing Dataset ...')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2020)
    print('Done.')

    print('Training Model ...')
    model = cb.CatBoostClassifier(
        iterations=20000,
        od_type='Iter',
        depth=5,
        learning_rate=1e-3,
        l2_leaf_reg=5,
        loss_function='Logloss',
        logging_level='Verbose',
        subsample=0.80,
        random_seed=2020,
        thread_count=-1
    )
    # model.fit(x_train, y_train, cat_features=["age_range", "gender"])
    model.fit(x_train, y_train)
    print('Done.')
    # fea_name = [column for column in x_train]
    # fea_importance = model.get_feature_importance()
    # plt.bar(fea_name, fea_importance)
    # plt.xticks(fea_name, fea_name, rotation=45)
    # plt.show()

    y_pred = model.predict_proba(x_test)
    # show_acc(y_test, y_pred[:, 1], 'CatBoost')
    print(roc_auc_score(y_test, y_pred[:, 1]))

    # print('Saving Model File ...')
    # model_name = 'model_' + time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # model.save_model('./model/' + model_name + '.model')
    # print('Done.')


train()
