import pandas as pd
import numpy as np


def RMSe(e, r):
    return np.sqrt(np.sum((((e - r) ** 2))) / len(e))


def get_quantile(data, q):
    quantile_limit = data.quantile(q=q)
    quantile_mean = data[data >= quantile_limit].mean()
    return quantile_limit, quantile_mean


class common():

    def compute_features(test_df, verbose=False):
        test_time = test_df.timestamp.iloc[0]
        n = test_df.shape[0]
        positive_data = test_df[test_df.response_time > 0]  # drop all "too early samples"
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
            'Subject': test_df.subject.iloc[0],
            'Test_nr': test_df.test.iloc[0],
            'n_total': n,
            'n_positive': n_positive,
            'positive_mean': positive_mean,
            'positive_median': positive_median,
            'positive_std': positive_std,
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

