# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time

FILENAME = {
    "train": "./data/train_format1.csv",
    "user_log": "./data/user_log_format1.csv",
    "user_info": "./data/user_info_format1.csv",
}
TESTNAME = './data/test_format1.csv'


def time_cost(func):
    def wrapper(*args, **kw):
        start = time.time()
        res = func(*args, **kw)
        end = time.time()
        print(
            "Function: {}, Cost: {:.3f}sec".format(
                func.__name__,
                (end - start)))
        return res
    return wrapper


def data_clean(data, fea, sigma=3):
    data_mean = np.mean(data[fea])
    data_std = np.std(data[fea], ddof=1)
    delta = sigma * data_std
    lower_thr = data_mean - delta
    upper_thr = data_mean + delta
    data[fea + '_outlier'] = data[fea].apply(
        lambda x: str('T') if x > upper_thr or x < lower_thr else str('F'))
    return data


@time_cost
def load_data(filename=FILENAME):
    if filename is None:
        filename = FILENAME

    print('Loading Samples ...')
    train = pd.read_csv(filename["train"])
    user_info = pd.read_csv(filename["user_info"])
    user_log = pd.read_csv(filename["user_log"])
    user_log = user_log.drop(columns=["brand_id"])
    print('Done.')
    print('*' * 20)

    print('Filling NaN Items ...')
    user_info["age_range"] = user_info["age_range"].fillna(
        user_info["age_range"].mode())
    user_info["gender"] = user_info["gender"].fillna(2)
    print('Done.')
    print('*' * 20)

    print('Merging Train Dataset ...')
    # train = pd.merge(train, user_info, on="user_id", how="left")
    # total_logs
    total_logs_temp = user_log.groupby([user_log["user_id"], user_log["seller_id"]]).count(
    ).reset_index()[["user_id", "seller_id", "item_id"]]
    total_logs_temp.rename(
        columns={
            "seller_id": "merchant_id",
            "item_id": "total_logs"},
        inplace=True)
    train = pd.merge(
        train, total_logs_temp, on=[
            "user_id", "merchant_id"], how="left")
    # item_count
    item_count_temp = user_log.groupby([user_log["user_id"],
                                        user_log["seller_id"],
                                        user_log["item_id"]]).count().reset_index()[["user_id",
                                                                                     "seller_id",
                                                                                     "item_id"]]
    item_count_temp = item_count_temp.groupby(
        [item_count_temp["user_id"], item_count_temp["seller_id"]]).count().reset_index()
    item_count_temp.rename(
        columns={
            "seller_id": "merchant_id",
            "item_id": "item_count"},
        inplace=True)
    train = pd.merge(
        train, item_count_temp, on=[
            "user_id", "merchant_id"], how="left")
    # cat_count
    cat_count_temp = user_log.groupby([user_log["user_id"],
                                       user_log["seller_id"],
                                       user_log["cat_id"]]).count().reset_index()[["user_id",
                                                                                   "seller_id",
                                                                                   "cat_id"]]
    cat_count_temp = cat_count_temp.groupby(
        [cat_count_temp["user_id"], cat_count_temp["seller_id"]]).count().reset_index()
    cat_count_temp.rename(
        columns={
            "seller_id": "merchant_id",
            "cat_id": "cat_count"},
        inplace=True)
    train = pd.merge(
        train, cat_count_temp, on=[
            "user_id", "merchant_id"], how="left")
    # click_on, add_cart, buy_up, mark_down
    action_log_temp = pd.get_dummies(user_log, columns=["action_type"])
    action_log_temp = action_log_temp.groupby([user_log["user_id"], user_log["seller_id"]]).agg(
        {"action_type_0": sum, "action_type_1": sum, "action_type_2": sum, "action_type_3": sum})
    action_log_temp = action_log_temp.reset_index()[["user_id",
                                                     "seller_id",
                                                     "action_type_0",
                                                     "action_type_1",
                                                     "action_type_2",
                                                     "action_type_3"]]
    action_log_temp.rename(
        columns={
            "seller_id": "merchant_id",
            "action_type_0": "click_on",
            "action_type_1": "add_cart",
            "action_type_2": "buy_up",
            "action_type_3": "mark_down"},
        inplace=True)
    train = pd.merge(
        train, action_log_temp, on=[
            "user_id", "merchant_id"], how="left")
    # browse_days
    browse_days_temp = user_log.groupby([user_log["user_id"],
                                         user_log["seller_id"],
                                         user_log["time_stamp"]]).count().reset_index()[["user_id",
                                                                                         "seller_id",
                                                                                         "time_stamp"]]
    browse_days_temp = browse_days_temp.groupby([browse_days_temp["user_id"],
                                                 browse_days_temp["seller_id"]]).count().reset_index()
    browse_days_temp.rename(
        columns={
            "seller_id": "merchant_id",
            "time_stamp": "browse_days"},
        inplace=True)
    train = pd.merge(train, browse_days_temp, on=["user_id", "merchant_id"], how="left")
    # bought_rate
    bought_rate_temp = pd.get_dummies(user_log, columns=["action_type"])
    bought_rate_temp = bought_rate_temp.groupby([user_log["user_id"]]).agg(
        {"action_type_0": sum, "action_type_1": sum, "action_type_2": sum, "action_type_3": sum})
    bought_rate_temp = bought_rate_temp.reset_index()[["user_id",
                                                       "action_type_0",
                                                       "action_type_1",
                                                       "action_type_2",
                                                       "action_type_3"]]
    bought_rate_temp["bought_rate"] = bought_rate_temp["action_type_2"] / (bought_rate_temp["action_type_0"]
                                                                           + bought_rate_temp["action_type_1"]
                                                                           + bought_rate_temp["action_type_2"]
                                                                           + bought_rate_temp["action_type_3"])
    bought_rate_temp = bought_rate_temp.drop(columns=["action_type_0",
                                                      "action_type_1",
                                                      "action_type_2",
                                                      "action_type_3"])
    train = pd.merge(train, bought_rate_temp, on="user_id", how="left")
    # sold_rate
    sold_rate_temp = pd.get_dummies(user_log, columns=["action_type"])
    sold_rate_temp = sold_rate_temp.groupby([user_log["seller_id"]]).agg(
        {"action_type_0": sum, "action_type_1": sum, "action_type_2": sum, "action_type_3": sum})
    sold_rate_temp = sold_rate_temp.reset_index()[["seller_id",
                                                   "action_type_0",
                                                   "action_type_1",
                                                   "action_type_2",
                                                   "action_type_3"]]
    sold_rate_temp["sold_rate"] = sold_rate_temp["action_type_2"] / (sold_rate_temp["action_type_0"]
                                                                     + sold_rate_temp["action_type_1"]
                                                                     + sold_rate_temp["action_type_2"]
                                                                     + sold_rate_temp["action_type_3"])
    sold_rate_temp = sold_rate_temp.drop(columns=["action_type_0",
                                                  "action_type_1",
                                                  "action_type_2",
                                                  "action_type_3"])
    sold_rate_temp.rename(columns={"seller_id": "merchant_id"}, inplace=True)
    train = pd.merge(train, sold_rate_temp, on="merchant_id", how="left")
    print('Done.')
    print('*' * 20)

    label = train['label']
    train = train.drop(columns=["user_id", "merchant_id", "label"])
    # train["age_range"] = train["age_range"].astype(str)
    # train["gender"] = train["gender"].astype(str)
    feature = train

    print('Shape of Dataset: {}'.format(feature.shape))

    return feature, label


def load_test(testname=TESTNAME):
    filename = FILENAME

    print('Loading Tests ...')
    test = pd.read_csv(testname)
    user_info = pd.read_csv(filename["user_info"])
    user_log = pd.read_csv(filename["user_log"])
    user_log = user_log.drop(columns=["brand_id"])
    print('Done.')
    print('*' * 20)

    print('Filling NaN Items ...')
    user_info["age_range"] = user_info["age_range"].fillna(
        user_info["age_range"].mode())
    user_info["gender"] = user_info["gender"].fillna(2)
    print('Done.')
    print('*' * 20)

    print('Merging Test Dataset ...')
    # test = pd.merge(test, user_info, on="user_id", how="left")
    # total_logs
    total_logs_temp = user_log.groupby([user_log["user_id"], user_log["seller_id"]]).count(
    ).reset_index()[["user_id", "seller_id", "item_id"]]
    total_logs_temp.rename(
        columns={
            "seller_id": "merchant_id",
            "item_id": "total_logs"},
        inplace=True)
    test = pd.merge(
        test, total_logs_temp, on=[
            "user_id", "merchant_id"], how="left")
    # item_count
    item_count_temp = user_log.groupby([user_log["user_id"],
                                        user_log["seller_id"],
                                        user_log["item_id"]]).count().reset_index()[["user_id",
                                                                                     "seller_id",
                                                                                     "item_id"]]
    item_count_temp = item_count_temp.groupby(
        [item_count_temp["user_id"], item_count_temp["seller_id"]]).count().reset_index()
    item_count_temp.rename(
        columns={
            "seller_id": "merchant_id",
            "item_id": "item_count"},
        inplace=True)
    test = pd.merge(
        test, item_count_temp, on=[
            "user_id", "merchant_id"], how="left")
    # cat_count
    cat_count_temp = user_log.groupby([user_log["user_id"],
                                       user_log["seller_id"],
                                       user_log["cat_id"]]).count().reset_index()[["user_id",
                                                                                   "seller_id",
                                                                                   "cat_id"]]
    cat_count_temp = cat_count_temp.groupby(
        [cat_count_temp["user_id"], cat_count_temp["seller_id"]]).count().reset_index()
    cat_count_temp.rename(
        columns={
            "seller_id": "merchant_id",
            "cat_id": "cat_count"},
        inplace=True)
    test = pd.merge(
        test, cat_count_temp, on=[
            "user_id", "merchant_id"], how="left")
    # click_on, add_cart, buy_up, mark_down
    action_log_temp = pd.get_dummies(user_log, columns=["action_type"])
    action_log_temp = action_log_temp.groupby([user_log["user_id"], user_log["seller_id"]]).agg(
        {"action_type_0": sum, "action_type_1": sum, "action_type_2": sum, "action_type_3": sum})
    action_log_temp = action_log_temp.reset_index()[["user_id",
                                                     "seller_id",
                                                     "action_type_0",
                                                     "action_type_1",
                                                     "action_type_2",
                                                     "action_type_3"]]
    action_log_temp.rename(
        columns={
            "seller_id": "merchant_id",
            "action_type_0": "click_on",
            "action_type_1": "add_cart",
            "action_type_2": "buy_up",
            "action_type_3": "mark_down"},
        inplace=True)
    test = pd.merge(
        test, action_log_temp, on=[
            "user_id", "merchant_id"], how="left")
    # browse_days
    browse_days_temp = user_log.groupby([user_log["user_id"],
                                         user_log["seller_id"],
                                         user_log["time_stamp"]]).count().reset_index()[["user_id",
                                                                                         "seller_id",
                                                                                         "time_stamp"]]
    browse_days_temp = browse_days_temp.groupby([browse_days_temp["user_id"],
                                                 browse_days_temp["seller_id"]]).count().reset_index()
    browse_days_temp.rename(
        columns={
            "seller_id": "merchant_id",
            "time_stamp": "browse_days"},
        inplace=True)
    test = pd.merge(test, browse_days_temp, on=["user_id", "merchant_id"], how="left")
    # bought_rate
    bought_rate_temp = pd.get_dummies(user_log, columns=["action_type"])
    bought_rate_temp = bought_rate_temp.groupby([user_log["user_id"]]).agg(
        {"action_type_0": sum, "action_type_1": sum, "action_type_2": sum, "action_type_3": sum})
    bought_rate_temp = bought_rate_temp.reset_index()[["user_id",
                                                       "action_type_0",
                                                       "action_type_1",
                                                       "action_type_2",
                                                       "action_type_3"]]
    bought_rate_temp["bought_rate"] = bought_rate_temp["action_type_2"] / (bought_rate_temp["action_type_0"]
                                                                           + bought_rate_temp["action_type_1"]
                                                                           + bought_rate_temp["action_type_2"]
                                                                           + bought_rate_temp["action_type_3"])
    bought_rate_temp = bought_rate_temp.drop(columns=["action_type_0",
                                                      "action_type_1",
                                                      "action_type_2",
                                                      "action_type_3"])
    test = pd.merge(test, bought_rate_temp, on="user_id", how="left")
    # sold_rate
    sold_rate_temp = pd.get_dummies(user_log, columns=["action_type"])
    sold_rate_temp = sold_rate_temp.groupby([user_log["seller_id"]]).agg(
        {"action_type_0": sum, "action_type_1": sum, "action_type_2": sum, "action_type_3": sum})
    sold_rate_temp = sold_rate_temp.reset_index()[["seller_id",
                                                   "action_type_0",
                                                   "action_type_1",
                                                   "action_type_2",
                                                   "action_type_3"]]
    sold_rate_temp["sold_rate"] = sold_rate_temp["action_type_2"] / (sold_rate_temp["action_type_0"]
                                                                     + sold_rate_temp["action_type_1"]
                                                                     + sold_rate_temp["action_type_2"]
                                                                     + sold_rate_temp["action_type_3"])
    sold_rate_temp = sold_rate_temp.drop(columns=["action_type_0",
                                                  "action_type_1",
                                                  "action_type_2",
                                                  "action_type_3"])
    sold_rate_temp.rename(columns={"seller_id": "merchant_id"}, inplace=True)
    test = pd.merge(test, sold_rate_temp, on="merchant_id", how="left")
    print('Done.')
    print('*' * 20)

    test["user_id"] = test["user_id"].astype(str)
    test["merchant_id"] = test["merchant_id"].astype(str)
    info = np.asarray(test[["user_id", "merchant_id"]])
    test = test.drop(columns=["user_id", "merchant_id", "prob"])
    # test["age_range"] = test["age_range"].astype(str)
    # test["gender"] = test["gender"].astype(str)
    feature = test

    print('Shape of Dataset: {}'.format(feature.shape))

    return feature, info
