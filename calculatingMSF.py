import pandas as pd
import datetime

class MSF():

    def compute_MSF(feature_df,verbose=False, cache=True):

        result_list = pd.DataFrame()
        MSF_final = 0

        if cache:
            return [78095.657143,80223.283333,77042.860714,60128.900000,80620.792593,47766.539583,67473.637500]

        labels_df = pd.read_hdf('data.h5', 'labels')
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
                            len(MSF_participant_sleep_list.loc[
                                        MSF_participant_sleep_list.Subject_id == subject_id]) > 0):

                    yesterday_SO = \
                    MSF_participant_sleep_list.loc[MSF_participant_sleep_list.Subject_id == subject_id].SO.iloc[
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

        for subject_id, subject_df in MSF_participant_sleep_list.groupby(MSF_participant_sleep_list.Subject_id):

            avg_SD_free = MSF_participant_sleep_list.loc[(MSF_participant_sleep_list.Subject_id == subject_id) &
                                                         (MSF_participant_sleep_list.Workday == 0.0)].SD.mean()

            avg_SO_free = MSF_participant_sleep_list.loc[(MSF_participant_sleep_list.Subject_id == subject_id) &
                                                         (MSF_participant_sleep_list.Workday == 0.0)].SO_Sec.mean()

            MSF = avg_SO_free + avg_SD_free / 2
            SD_sum = MSF_participant_sleep_list.loc[(MSF_participant_sleep_list.Subject_id == subject_id)
                                                    & MSF_participant_sleep_list.SD != 0].SD.sum()
            SD_days = len(MSF_participant_sleep_list.loc[(MSF_participant_sleep_list.Subject_id == subject_id)
                                                         & MSF_participant_sleep_list.SD != 0].SD)

            SD_week_avg = SD_sum / SD_days

            if (avg_SD_free > SD_week_avg):
                MSF_final = MSF - (avg_SD_free - SD_week_avg) / 2
            else:
                MSF_final = MSF

            result_list = result_list.append(pd.Series(
                {
                    "Subject_id": subject_id,
                    "MSF": MSF,
                    "SD_week": SD_week_avg,
                    "MSF_final": MSF_final
                }), ignore_index=True);

            if (verbose):
                print("Subject_id ", subject_id,
                      " MSF ", MSF,
                      " SD_week ", SD_week_avg,
                      " MSF_final ", MSF_final
                      )

        print(result_list)
        return result_list.MSF_final
