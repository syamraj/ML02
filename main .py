import pandas as pd
import numpy as np
from sklearn import linear_model
import datetime

#we have two groups : data & labels

#data contains this information in each row :
#timestamp,foreperiod_start,foreperiod_end,response_received,subject,trial,test,requested_foreperiod,actual_foreperiod,response_time,note,tag,garbage_collection,

#label contains this information in each row:
#Alarmclock,Alcohol,Caffeine,Food,Medication,Nicotine,Participant_ID,Sleep,Sports,Time,Workday,
from sklearn.model_selection import train_test_split

raw_df = pd.read_hdf('data.h5', 'raw')
labels_df = pd.read_hdf('data.h5', 'labels')

feature_df = pd.DataFrame()
# y_train
# x_train
# dataset_train, dataset_test, label_train, label_test = train_test_split(raw_df, labels_df, test_size=0.1)

def RMSe(e, r):
    return np.sqrt(np.sum((((e - r)**2)))/len(e))

def get_quantile(data, q):
    """Takes series of values and returns quantile limit as well as the mean of the values above the quantile.
  data: Data as pandas Series.
  q: Quantile (0.75 -> 75%)
  returns: quantile limit, mean value of elements above quantile limit
  """

    quantile_limit = data.quantile(q=q)
    quantile_mean = data[data >= quantile_limit].mean()
    return quantile_limit, quantile_mean

def compute_features(test_df, verbose=False):
    """ Takes PVT test results and returns feature vector as a result.
    test_df: Dataframe containing PVT test results.
    Returns: Series containing the feature vector.
    """
    test_time = test_df.timestamp.iloc[0]
    n = test_df.shape[0]
    positive_data = test_df[test_df.response_time > 0] # drop all "too early samples"
    n_positive = positive_data.shape[0]
    positive_mean = positive_data.response_time.mean()
    positive_median = positive_data.response_time.median()
    positive_std = positive_data.response_time.std()
    q50_lim, q50_mean = get_quantile(positive_data.response_time, 0.50)
    q75_lim, q75_mean = get_quantile(positive_data.response_time, 0.75)
    q90_lim, q90_mean = get_quantile(positive_data.response_time, 0.90)
    q95_lim, q95_mean = get_quantile(positive_data.response_time, 0.95)
    features = pd.Series({
        'Test_time': test_time,
        'Subject' : test_df.subject.iloc[0],
        'Test_nr': test_df.test.iloc[0],
        'n_total': n,
        'n_positive': n_positive,
        'positive_mean': positive_mean,
        'positive_median': positive_median,
        'positive_std' : positive_std,
        'q50_lim': q50_lim,
        'q75_lim': q75_lim,
        'q90_lim': q90_lim,
        'q95_lim': q95_lim,
        'q50_mean': q50_mean,
        'q75_mean': q75_mean,
        'q90_mean': q90_mean,
        'q95_mean': q95_mean})
    if verbose:
        print(features)
    return features



for subject_id, subject_df in raw_df.groupby(raw_df.subject):
    for test_id, test_df in subject_df.groupby(subject_df.test):
        feature_df = feature_df.append(compute_features(test_df), ignore_index=True)
    feature_df.reset_index(inplace=True, drop=True)

# Compute the time of day as a float
h = feature_df.Test_time.apply(lambda x: x.hour)
m = feature_df.Test_time.apply(lambda x: x.minute)
date = feature_df.Test_time.apply(lambda x: x.date())
feature_df['time_of_day'] = h + (m/60.0)
feature_df['date'] = date

date = labels_df.Time.apply(lambda x: x.date())
labels_df['date'] = date
#This will result in a DataFrame with one entry per test and the following columns:

#################################################################################################################
#first step : calculate circadian baseline for each participant throught MSF formula#############################
    #tip : not necessariliy consecuence
# baraye har subjecty avalin time test va akharie dishabesh modeheme => toole khab ro darim. va shorooe khab ro
# hala chetori befahmim ke dishab boode! tarikh ham bayad jaee bashe
# age mal dishabsho nadashte bashim amalan bi asare oon roozemoon


#feature_df.sort(lambda x: x.Test_time)
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
        # labels_df = pd.read_hdf('data.h5', 'labels', where='Time='+min(date_df.Test_time))
        # print(labels_df[0])
        SD = 0

        if (len(MSF_participant_sleep_list) > 0 and
                    len(MSF_participant_sleep_list.loc[MSF_participant_sleep_list.Subject_id == subject_id])>0 ):

            yesterday_SO = MSF_participant_sleep_list.loc[MSF_participant_sleep_list.Subject_id == subject_id].SO.iloc[-1]
            today_wakeup_time = min(date_df.Test_time)

            if(yesterday_SO.date() == today_wakeup_time.date() - datetime.timedelta(days=1)):
                dif = today_wakeup_time - yesterday_SO
                SD = dif.seconds / (60*60)
                # print("participant ", subject_id , " slept at " , yesterday_SO, " and woke up at ", today_wakeup_time)
                # print("duration: " , SD)


        workdayRow = participant_work_list.loc[(participant_work_list.Date == date_day) &
                                               (participant_work_list.Subject_id == subject_id)]
        Workday = 0
        if(len(workdayRow.Workday.values) > 0):
            Workday = workdayRow.Workday.values[0]

        MSF_participant_sleep_list = MSF_participant_sleep_list.append(pd.Series(
        {
            'Subject_id': subject_id,
            'SO' : max(date_df.Test_time),
            'SD' : SD,
            'Workday' : Workday
        }),ignore_index=True)

    # avg_SO_free = MSF_participant_sleep_list["SO"].mean()
    avg_SD_free = MSF_participant_sleep_list.loc[(MSF_participant_sleep_list.Subject_id == subject_id) &
                                                 (MSF_participant_sleep_list.Workday == 1.0)].SD.mean()

    avg_SD_work = MSF_participant_sleep_list.loc[(MSF_participant_sleep_list.Subject_id == subject_id) &
                                                 (MSF_participant_sleep_list.Workday == 0.0)].SD.mean()

    work_days = len(MSF_participant_sleep_list.loc[(MSF_participant_sleep_list.Subject_id == subject_id) &
                                                 (MSF_participant_sleep_list.Workday == 1.0)].SD)

    free_days = len(MSF_participant_sleep_list.loc[(MSF_participant_sleep_list.Subject_id == subject_id) &
                                                 (MSF_participant_sleep_list.Workday == 0.0)].SD)

    # print("participate " , subject_id, "mean SD free" , avg_SD_free)
    # print("participate ", subject_id, "mean SD work", avg_SD_work)

    # print("participate " , subject_id, " free days:" , free_days)
    # print("participate ", subject_id, " work days:", work_days)





#################################################################################################################
#step two : preparing feature####################################################################################

###################################################################################################################
#step three : building model base on feture########################################################################
# regr = linear_model.LinearRegression()
# regr.fit(featture_train_set, y_train_set)
##################################################################################################################
#last step: comare the result of the model(e=circadian phase base on reaction time) and r=MSF circadian phase#################



