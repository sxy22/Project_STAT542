#!/usr/bin/env python
# coding: utf-8
import numpy as np 
import pandas as pd
import xgboost as xgb 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# feature interaction
def interact(data, inter_feature):
    for sub in inter_feature:
        for i in range(len(sub)):
            for j in range(i + 1, len(sub)):
                f1 = sub[i]
                f2 = sub[j]
                inter = f1 + '_x_' + f2
                data_inter = data[f1] * data[f2]
                data_inter.name = inter
                data = pd.concat((data, data_inter), axis=1)
    return data 

# read train
train = pd.read_csv('./train.csv')
# processing train data
removed_features = ['Street', 'Utilities', 'Condition_2', 'Roof_Matl', 'Heating', 
                'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 
                'Longitude','Latitude', 'Bsmt_Full_Bath', 'Bsmt_Half_Bath']
train.index = pd.Index(range(len(train)))
train.fillna(0, inplace=True)
Y = train['Sale_Price']
train.drop(['Sale_Price', 'PID'], axis=1, inplace=True)
train.drop(removed_features, axis=1, inplace=True)
# feature
features = train.columns.values.tolist()
categ_features = []
dense_features = []
for f, dtype in train.dtypes.items():
    if dtype == object:
        categ_features.append(f)
    else:
        dense_features.append(f)
inter_feature = [['Year_Built', 'Bsmt_Unf_SF', 'Mo_Sold'],
                 ['Year_Built', 'Lot_Area'],
                 ['Year_Built', 'Garage_Area'],
                 ['Mo_Sold', 'Garage_Area']]
# regression
train_reg = train.copy()
# wins
wins_features = ["Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Total_Bsmt_SF", 
                 "Second_Flr_SF", 'First_Flr_SF', "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF", 
                 "Open_Porch_SF", 'Enclosed_Porch', 'Three_season_porch', 'Screen_Porch', "Misc_Val"] 
features_quan = []
for f in wins_features:
    col = train_reg[f]
    quan95 = np.quantile(col, 0.95)
    features_quan.append(quan95)
    train_reg[f] = np.where(col > quan95, quan95, col)
# standardize
scaler = StandardScaler()
train_reg_dense = train_reg[dense_features]
standard_train_reg = pd.DataFrame(scaler.fit_transform(train_reg_dense), columns=train_reg_dense.columns)
# interaction
standard_train_reg = interact(standard_train_reg, inter_feature)
# onehot
OH = OneHotEncoder(handle_unknown='ignore')
onehot_train_reg = OH.fit_transform(train_reg[categ_features]).toarray()
onehot_train_reg = pd.DataFrame(onehot_train_reg, columns=OH.get_feature_names())
# concat
train_reg = pd.concat([onehot_train_reg, standard_train_reg], axis=1)
Y = np.log(Y)

# fit reg model
ridge_alphas = np.linspace(4, 15, 100) 
ridgecv = RidgeCV(alphas=ridge_alphas, normalize=False, cv=10)
ridgecv.fit(train_reg, Y)

# fit xgb model
xgb_model = xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                      verbosity=0, subsample=0.5, random_state=2021)

xgb_model.fit(train_reg, Y)


# read test
test = pd.read_csv('./test.csv')
# processing test
test.index = pd.Index(range(len(test)))
test.fillna(0, inplace=True)
test.drop(removed_features, axis=1, inplace=True)
test_PID = test['PID']
test.drop(['PID'], axis=1, inplace=True)
# reg test 
test_reg = test.copy()
# wins
for i in range(len(wins_features)):
    quan95 = features_quan[i]
    f = wins_features[i]
    test_reg[f] = np.where(test_reg[f] > quan95, quan95, test_reg[f])
# standardize
test_reg_dense = test_reg[dense_features]
standard_test_reg = pd.DataFrame(scaler.transform(test_reg_dense), columns=test_reg_dense.columns)
# interaction
standard_test_reg = interact(standard_test_reg, inter_feature)
onehot_test_reg = OH.transform(test_reg[categ_features]).toarray() 
onehot_test_reg = pd.DataFrame(onehot_test_reg, columns=OH.get_feature_names())
# concat
test_reg = pd.concat([onehot_test_reg, standard_test_reg], axis=1)
# predict using reg model
test_pred_reg = ridgecv.predict(test_reg)
# predict using xgb model
test_pred_xgb = xgb_model.predict(test_reg)

test_pred_reg = np.exp(test_pred_reg)
test_pred_xgb = np.exp(test_pred_xgb)

mysubmission1 = pd.DataFrame({'PID': test_PID, 'Sale_Price': test_pred_reg})
mysubmission2 = pd.DataFrame({'PID': test_PID, 'Sale_Price': test_pred_xgb})
mysubmission1.to_csv('./mysubmission1.txt', sep=',', index=False)
mysubmission2.to_csv('./mysubmission2.txt', sep=',', index=False)

