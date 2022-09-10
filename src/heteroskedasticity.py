import pandas as pd
import statsmodels.stats.api as sms
from statsmodels.formula.api import ols

from patsy import dmatrices


def het_tests(series: pd.Series, test: str) -> float:
    """
    Testing for heteroskedasticity using the White or Breusch-Pagan test
    :param series: Univariate time series as pd.Series
    :param test: String denoting the test. One of 'white' or 'breuschpagan'

    :return: p-value as a float.
    If the p-value is high, we accept the null hypothesis that there is no heteroscedastisticity
    """
    formula = 'value ~ time'
    assert test in ['white', 'breuschpagan'], 'Unknown test'

    series = series.reset_index(drop=True).reset_index()
    series.columns = ['time', 'value']
    series['time'] += 1

    olsr = ols(formula, series).fit()

    y, X = dmatrices(formula, series, return_type='dataframe')

    if test == 'white':
        _, p_value, _, _ = sms.het_white(olsr.resid, X)
    else:
        _, p_value, _, _ = sms.het_breuschpagan(olsr.resid, X)

    return p_value
