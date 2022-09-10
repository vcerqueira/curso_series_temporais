import pandas as pd

from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae


def optimize_prophet(series: pd.Series, grid):
    test_size = 48

    train, test = train_test_split(series, test_size=test_size, shuffle=False)

    opt_results = {'loss': [], 'params': []}

    n = len(grid)

    for i, params in enumerate(grid):
        print(f'Configuration {i + 1}/{n}')
        model = Prophet(seasonality_mode=params['seasonality_mode'],
                        growth=params['growth'],
                        weekly_seasonality=True,
                        daily_seasonality=True,
                        yearly_seasonality=True,
                        n_changepoints=params['n_changepoints'],
                        changepoint_prior_scale=params['changepoint_prior_scale'])

        model = model.add_seasonality(name='monthly', period=24 * 30, fourier_order=10)

        train_df = train.reset_index()
        train_df.columns = ['ds', 'y']

        model.fit(train_df)

        forecast = model.make_future_dataframe(periods=test.shape[0],
                                               include_history=False,
                                               freq='H')
        forecast = model.predict(forecast)
        forecast = forecast.filter(items=['ds', 'yhat'])

        loss = mae(test, forecast['yhat'].values)

        opt_results['loss'].append(loss)
        opt_results['params'].append(params)

    idx_params = pd.Series(opt_results['loss']).argmin()
    params = opt_results['params'][idx_params]

    return params
