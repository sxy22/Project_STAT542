#!/usr/bin/env python
# coding: utf-8

# import time
from sklearn.ensemble import RandomForestRegressor
import pandas as pd 
import numpy as np 
from datetime import datetime
from dateutil.relativedelta import relativedelta

def get_weeknum(date):
    res = date.isocalendar()[1]
    return res  

def data_process(data):
    data['weeknum'] = data['Date'].map(get_weeknum)
    data['year'] = data['Date'].map(lambda x: x.year)
    data['IsHoliday'] = data['IsHoliday'].map(lambda x: 5 if x else 1)
    # data.drop(['Date'], axis=1, inplace=True)

def mypredict(train, test, next_fold, t):
    fearutes = ['Store', 'Dept', 'IsHoliday', 'weeknum', 'year']
    if next_fold is None:
        data_process(train)
        data_process(test)
    else:
        data_process(next_fold)
        train = pd.concat([train, next_fold])
    train_x, train_y = train[fearutes], train['Weekly_Sales']
    use_weight = True
    train_weight = None
    if use_weight:
        train_weight = train_x['IsHoliday'].values
        
    start_date = datetime(2011, 3, 1) + relativedelta(months=2 * (t - 1))
    end_date = datetime(2011, 5, 1) + relativedelta(months=2 * (t - 1))
    
    ind = test[(test['Date'] >= start_date) & (test['Date'] < end_date)].index
    test_pred = test.loc[ind,].copy()
    test.drop(ind, inplace=True)
    
    test_x = test_pred[fearutes]
    pred_y = None
    # start = time.time()
    RF = RandomForestRegressor(n_estimators=150, max_depth=40,
                               max_features='auto', random_state=542,
                               n_jobs=-1, bootstrap=True, oob_score=False)
    RF.fit(train_x, train_y, sample_weight=train_weight)
    pred_y = RF.predict(test_x)
    # end = time.time()
    # run_time = np.round(end - start, 2)
    # print(t, run_time)
    test_pred['Weekly_Pred'] = pred_y
    
    return train, test_pred
