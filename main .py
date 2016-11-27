import matplotlib.pyplot as plt
import pandas as pd
from pandas import scatter_matrix
from sklearn import linear_model
from calculatingMSF import MSF
from common import common
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
y_set = MSF.compute_MSF(feature_df)

#playing ith different features to find the best one######################################################################################
# x_set = np.zeros((7,1))
# x_set[:,0] = feature_df.groupby(feature_df.Subject).q50_mean.mean().values

# x_set = [[y] for y in feature_df.groupby(feature_df.Subject).q50_mean.mean().values]#0.34
# x_set = [[y] for y in feature_df.groupby(feature_df.Subject).q95_mean.mean().values]#0.096
# x_set = [[y] for y in feature_df.groupby(feature_df.Subject).q75_mean.mean().values]#0.63

#trying to remove some noisy values from feature_df
#remove the test data those belong to drunk time,
labels_df = pd.read_hdf('data.h5', 'labels')

alcoholFreeTime = labels_df.loc[labels_df.Alcohol == 0].Time.apply(lambda x: x.to_pydatetime().replace(minute=0))
Test_time_without_sec = feature_df.Test_time.apply(lambda x: x.replace(minute=0,second=0))

feature_df['Test_time'] = Test_time_without_sec
feature_df = feature_df.loc[feature_df.Test_time.isin(alcoholFreeTime.values)]

x_set = [[y] for y in feature_df.groupby(feature_df.Subject).q50_mean.mean().values]#0.55

#give weights to some measurement times###################################################################################################




y_train_set = y_set[:-2]
y_test_set = y_set[-2:]

x_train_set = x_set[:-2]
x_test_set = x_set[-2:]
regr = linear_model.LinearRegression()
regr.fit(x_train_set, y_train_set)
predict = regr.predict(x_test_set)
print(regr.score(x_test_set, y_test_set))

# Plot Analysis
# plt.scatter(feature_df.groupby(feature_df.Subject).q50_mean.mean().values, y_set,  color='black')
# plt.scatter(feature_df.groupby(feature_df.Subject).q75_mean.mean().values, y_set,  color='blue')
# plt.scatter(feature_df.groupby(feature_df.Subject).q95_mean.mean().values, y_set,  color='red')

# feature_df.q95_mean.plot(kind='density', color ="blue")
# feature_df.q50_mean.plot(kind='density', color ="red")
# feature_df.q75_mean.plot(kind='density', color ="green")
#
# plt.show()
#########################################################################
# Plot outputs
# plt.scatter(x_test_set, y_test_set,  color='black')
# plt.plot(x_test_set, regr.predict(x_test_set), color='blue',linewidth=3)
#
# plt.xticks(())
# plt.yticks(())
#
# plt.show()

# last step: comare the result of the model(e=circadian phase base on reaction time) and r=MSF circadian phase#################
