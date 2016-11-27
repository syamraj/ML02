import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from calculatingMSF import MSF
from common import common

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


x_set = [[y] for y in feature_df.groupby(feature_df.Subject).q95_mean.mean().values]
y_set = MSF.compute_MSF(feature_df)

y_train_set = y_set[:-2]
y_test_set = y_set[-2:]

x_train_set = x_set[:-2]
x_test_set = x_set[-2:]

regr = linear_model.LinearRegression()
regr.fit(x_train_set, y_train_set)
predict = regr.predict(x_test_set)
print(regr.score(x_test_set, y_test_set))

# Plot outputs
# plt.scatter(x_test_set, y_test_set,  color='black')
# plt.plot(x_test_set, regr.predict(x_test_set), color='blue',linewidth=3)
#
# plt.xticks(())
# plt.yticks(())
#
# plt.show()

# last step: comare the result of the model(e=circadian phase base on reaction time) and r=MSF circadian phase#################
