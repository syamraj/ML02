__author__ = 'devil'

import matplotlib.pyplot as plt
import pandas as pd
from pandas import scatter_matrix
from sklearn import linear_model
from calculatingMSF import MSF
from common import common, RMSe
from sklearn.svm import SVR
import numpy as np

raw_df = pd.read_hdf('data.h5', 'raw')
labels_df = pd.read_hdf('data.h5', 'labels')

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


alcoholFreeTime = labels_df.loc[labels_df.Alcohol == 0].Time.apply(lambda x: x.to_pydatetime().replace(minute=0))
Test_time_without_sec = feature_df.Test_time.apply(lambda x: x.replace(minute=0, second=0))
feature_df['Test_time'] = Test_time_without_sec
feature_df = feature_df.loc[feature_df.Test_time.isin(alcoholFreeTime.values)]


# x_set = [[y] for y in feature_df.groupby(feature_df.Subject).q50_mean.mean().values]
# x_set = [[y] for y in feature_df.groupby(feature_df.Subject).q95_mean.mean().values]
# x_set = [[y] for y in feature_df.groupby(feature_df.Subject).q75_mean.mean().values]

x_set = [[y] for y in feature_df.groupby(feature_df.Subject).q50_mean.mean().values]




y_train_set = y_set[:-2]
print("y_train_set", y_train_set)
y_test_set = y_set[-2:]
print("y_test_set", y_test_set)
x_train_set = x_set[:-2]
print("x_train_set", x_train_set)
x_test_set = x_set[-2:]
print("y_test_set", y_test_set)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=.1)

svr_lin = SVR(kernel='linear', C=1e3)

svr_poly = SVR(kernel='poly', C=1e3, degree=2)

svr_rbf.fit(x_train_set, y_train_set)
print('svr_rbf')
svr_lin.fit(x_train_set, y_train_set)
print('svr_lin')

regr = linear_model.LinearRegression()
regr.fit(x_train_set, y_train_set)

error_svr_lin = RMSe(y_test_set, svr_lin.predict(x_test_set))
error_svr_rbf = RMSe(y_test_set, svr_rbf.predict(x_test_set))
#error_svr_poly = RMSe(y_test_set, svr_poly.predict(x_test_set))

print("RMSe for SVR_lin is:", error_svr_lin/3600)
print("RMSe for SVR_rbf is:", error_svr_rbf/3600)
#print("RMSe for SVR_poly is:", error_svr_poly)
#poly = svr_poly.fit(x_train_set, y_train_set)
#print('svr_poly')
#predict = svr_lin.predict(x_test_set)
#print(svr_poly.score(x_test_set, y_test_set))
#print(svr_poly.score(x_train_set, y_train_set))



plt.scatter(x_test_set, y_test_set, color='darkorange', label='data')
#plt.scatter(x_test_set, y_test_set, color='black', label='test data')
plt.hold('on')
plt.plot(x_test_set, svr_rbf.predict(x_test_set), color='navy', label='RBF model')
plt.plot(x_test_set, svr_lin.predict(x_test_set), color='c', label='Linear model')
plt.plot(x_test_set, regr.predict(x_test_set), color='blue',label = 'regr')
#plt.plot(x_test_set, poly.predict(x_test_set), color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('SVC and linear ')
plt.legend()
plt.show()