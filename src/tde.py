import numpy as np
import pandas as pd


def MultivariateTDE(data: pd.DataFrame,
                    k: int,
                    horizon: int,
                    target_col: str, drop_na: bool = True):
    """
    time delay embedding for mv time series

    :param data: multivariate time series as pd.DF
    :param k: embedding dimension (applied to all cols)
    :param horizon: forecasting horizon
    :param target_col: string denoting the target column

    :return: trainable data set
    """

    iter_over_k = list(range(k, 0, -1))

    X_cols = []
    for col in data.columns:
        # input sequence (t-n, ... t-1)
        X, col_iter = [], []
        for i in iter_over_k:
            X.append(data[col].shift(i))

        X = pd.concat(X, axis=1)
        X.columns = [f'{col}-{j-1}' for j in iter_over_k]
        X_cols.append(X)

    X_cols = pd.concat(X_cols, axis=1)

    # forecast sequence (t, t+1, ... t+n)
    y = []
    for i in range(0, horizon):
        y.append(data[target_col].shift(-i))

    y = pd.concat(y, axis=1)
    y.columns = [f'{target_col}+{i}' for i in range(1, horizon + 1)]

    data_set = pd.concat([X_cols, y], axis=1)

    if drop_na:
        data_set = data_set.dropna()

    return data_set


def UnivariateTDE(data: pd.Series, k: int, horizon: int, drop_na: bool = True):
    """
    time delay embedding for mv time series

    :param data: multivariate time series as pd.DF
    :param k: embedding dimension (applied to all cols)
    :param horizon: forecasting horizon
    :param target_col: string denoting the target column

    :return: trainable data set
    """

    s = pd.DataFrame({'t': data})
    df = MultivariateTDE(data=s, k=k, horizon=horizon, target_col='t')
    if drop_na:
        df = df.dropna().reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


def time_delay_embedding(series: pd.Series, n_lags: int, horizon: int):
    """
    Time delay embedding
    Time series for supervised learning

    :param series: time series as pd.Series
    :param n_lags: number of past values to used as explanatory variables
    :param horizon: how many values to forecast
    :return: pd.DataFrame with reconstructed time series
    """
    n_lags_iter = list(range(n_lags, -horizon, -1))

    X = [series.shift(i) for i in n_lags_iter]
    X = pd.concat(X, axis=1)
    X.columns = [f't-{j - 1}' if j > 0 else f't+{np.abs(j) + 1}'
                 for j in n_lags_iter]

    return X
