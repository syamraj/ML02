import pandas as pd
from sklearn import linear_model
import datetime
from calculatingMSF import MSF
from common import common

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

date = labels_df.Time.apply(lambda x: x.date())
labels_df['date'] = date

participant_work_list = pd.DataFrame()
for subject_id, subject_df in labels_df.groupby(labels_df.Participant_ID):
    for date_dy, date_df in subject_df.groupby(subject_df.date):
        participant_work_list = participant_work_list.append(pd.Series(
            {
                # normally for determining free or work date subject_ID dosnt matte,
                # but sometimes the work day and free day of different people differ
                'Subject_id': subject_id,
                'Date': date_dy,
                'Workday': date_df.Workday.iloc[0],

            }), ignore_index=True)

MSF_participant_sleep_list = pd.DataFrame()
for subject_id, subject_df in feature_df.groupby(feature_df.Subject):
    for date_day, date_df in subject_df.groupby(subject_df.date):

        SD = 0
        Workday = 0

        if (len(MSF_participant_sleep_list) > 0 and
                    len(MSF_participant_sleep_list.loc[MSF_participant_sleep_list.Subject_id == subject_id]) > 0):

            yesterday_SO = MSF_participant_sleep_list.loc[MSF_participant_sleep_list.Subject_id == subject_id].SO.iloc[
                -1]
            today_wakeup_time = min(date_df.Test_time)

            if (yesterday_SO.date() == today_wakeup_time.date() - datetime.timedelta(days=1)):
                dif = today_wakeup_time - yesterday_SO
                SD = dif.seconds / (60)

        workdayRow = participant_work_list.loc[(participant_work_list.Date == date_day) &
                                               (participant_work_list.Subject_id == subject_id)]

        if (len(workdayRow.Workday.values) > 0):
            Workday = workdayRow.Workday.values[0]

        i = max(date_df.Test_time).to_pydatetime()
        MSF_participant_sleep_list = MSF_participant_sleep_list.append(pd.Series(
            {
                'Subject_id': subject_id,
                'SO': max(date_df.Test_time),
                'SO_Sec': int((i.hour * 3600 + i.minute * 60 + i.second)),  # seconds
                'SD': SD,  # seconds
                'Workday': Workday
            }), ignore_index=True)

MSF.compute_MSF(MSF_participant_sleep_list, verbose=False)

# regr = linear_model.LinearRegression()
# regr.fit(featture_train_set, y_train_set)
# last step: comare the result of the model(e=circadian phase base on reaction time) and r=MSF circadian phase#################
