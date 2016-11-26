import pandas as pd

class MSF():

    def compute_MSF(MSF_participant_sleep_list, verbose=False):

        result_list = pd.DataFrame()
        MSF_final = 0
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
