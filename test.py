# -*- coding:utf-8 -*-

from catboost import CatBoostClassifier
import dataset as ds
import pandas as pd
import numpy as np
import time


test_data, info_list = ds.load_test()
print('Loading Model ...')
model = CatBoostClassifier()
model.load_model('./model/model_20201217_162334.model')
print('Done.')
print('*' * 20)

print('Predicting ...')
y_pred = model.predict_proba(test_data)
ans = y_pred[:, 1].reshape(-1, 1)
print('Done.')
print('*' * 20)

print('Generating Answer File ...')
answer = np.concatenate((info_list, ans), axis=1)
answer = pd.DataFrame(answer)
answer.columns = ["user_id", "merchant_id", "prob"]
ans_name = 'ans_' + time.strftime('%Y%m%d_%H%M%S', time.localtime())

print('Saving File ...')
answer.to_csv('./result/' + ans_name + '.csv', sep=',', header=True, index=False)
print('Done.')
