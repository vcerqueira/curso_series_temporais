from datetime import datetime

import numpy as np
import pandas as pd
from sklego.preprocessing import RepeatingBasisFunction


class RBFTerms:
    AVAILABLE_PERIODS = [
        '.month',
        # '.quarter',
        '.day',
        '.hour',
        '.dayofweek',
        # '.weekofyear',
        # '.dayofyear'
    ]

    PERIOD_RANGES = {
        '.month': (1, 12),
        # '.quarter',
        '.day': (1, 31),
        '.hour': (0, 23),
        '.dayofweek': (0, 6),
        # '.weekofyear',
        # '.dayofyear'
    }

    def __init__(self, n_terms: int, period: str, prefix: str = ''):
        assert period in self.AVAILABLE_PERIODS, 'period not in av'

        self.period = period
        self.n_terms = n_terms
        self.prefix = prefix
        self.model = RepeatingBasisFunction(n_periods=n_terms,
                                            column='index',
                                            input_range=self.PERIOD_RANGES[self.period],
                                            remainder='drop')

    def fit(self, index: pd.DatetimeIndex):
        index_df = self.prepare_index_df(index)

        self.model.fit(index_df)

    def transform(self, index: pd.DatetimeIndex):
        index_df = self.prepare_index_df(index)

        feats = self.model.transform(index_df)

        col_names = [f'{self.prefix}RBT{i}' for i in range(feats.shape[1])]

        feats = pd.DataFrame(feats, columns=col_names, index=index)

        return feats

    def prepare_index_df(self, index: pd.DatetimeIndex):
        assert isinstance(index, pd.DatetimeIndex), 'asdsa'

        df = pd.DataFrame({'index': eval(f'index{self.period}')})

        return df


class FourierTerms:

    def __init__(self, period: float, n_terms: int, prefix=''):
        self.period = period
        self.n_terms = n_terms
        self.prefix = prefix

    def transform(self, index: pd.DatetimeIndex, use_as_index: bool = True):
        t = np.array(
            (index - datetime(1970, 1, 1)).total_seconds().astype(float)
        ) / (3600 * 24.)

        fourier_x = np.column_stack([
            fun((2.0 * (i + 1) * np.pi * t / self.period))
            for i in range(self.n_terms)
            for fun in (np.sin, np.cos)
        ])

        col_names = [
            f'{self.prefix}{fun.__name__[0].upper()}{i}'
            for i in range(self.n_terms)
            for fun in (np.sin, np.cos)
        ]

        fourier_df = pd.DataFrame(fourier_x, columns=col_names)

        if use_as_index:
            fourier_df.index = index

        return fourier_df
