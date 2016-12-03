__author__ = 'devil'

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import AdaBoostRegressor as AdaboostR
from pandas import scatter_matrix
from sklearn import linear_model
from calculatingMSF import MSF
from common import common, RMSe
import numpy as np

raw_df = pd.read_hdf('data.h5', 'raw')

feature_df = pd.DataFrame()
for subject_id, subject_df in raw_df.groupby(raw_df.subject):
    for test_id, test_df in subject_df.groupby(subject_df.test):
        feature_df = feature_df.append(common.compute_features(test_df), ignore_index=True)
    feature_df.reset_index(inplace=True, drop=True)

h = feature_df.Test_time.apply(lambda x: x.hour)
m = feature_df.Test_time.apply(lambda x: x.minute)
date = feature_df.Test_time.apply(lambda x: x.date())
feature_df['time_of_day'] = h + (m / 60.0)
feature_df['date'] = date
computation = MSF.compute_MSF(feature_df)
labels_df = pd.read_hdf('data.h5', 'labels')

y_set = pd.DataFrame()
print("Enter the model to use :")
print("1. MSFsc")
print("2. MSF")
print("3. MSW")
ch = input("Please enter something: ")
if ch == '1':
    y_set = computation.MSF_final
if ch == '2':
    y_set = computation.MSF
if ch == '3':
    y_set = computation.MSW
    y_set.iloc[4] = 0

# #Removing the Alcohol data
# alcoholFreeTime = labels_df.loc[labels_df.Alcohol == 0].Time.apply(lambda x: x.to_pydatetime().replace(minute=0))
# Test_time_without_sec = feature_df.Test_time.apply(lambda x: x.replace(minute=0, second=0))
# feature_df['Test_time'] = Test_time_without_sec
# feature_df = feature_df.loc[feature_df.Test_time.isin(alcoholFreeTime.values)]

# selecting the feature set
x_set = [[y] for y in feature_df.groupby(feature_df.Subject).q50_mean.mean().values]

# dividing into test and train set
y_train_set = y_set[:-2]
y_test_set = y_set[-2:]
x_train_set = x_set[:-2]
x_test_set = x_set[-2:]

# train regression
tree = DTR(max_depth=None)
tree.fit(x_train_set, y_train_set)
rng = np.random.RandomState(1)
ada = AdaboostR(DTR(max_depth=None), n_estimators=300, random_state=rng)
ada.fit(x_train_set, y_train_set)
predict = ada.predict(x_test_set)
error = RMSe(y_test_set, predict)
print("RMSe value is:", error / 3600)

# plotting
plt.scatter(x_test_set, y_test_set, color='black')
plt.plot(x_test_set, ada.predict(x_test_set), color='blue', linewidth=3)
plt.show()
