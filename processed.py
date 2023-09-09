import pandas as pd
import numpy as np
import ydata_profiling
from sklearn.preprocessing import MinMaxScaler

def load():
    date_vars = ['DatetimeBegin','DatetimeEnd']

    agg_ts = pd.read_csv('data/BE_1_2013-2015_aggregated_timeseries.csv', sep='\t', dtype={'DatetimeBegin': 'object', 'DatetimeEnd': 'object'})
    agg_ts['DatetimeBegin'] = pd.to_datetime(agg_ts['DatetimeBegin'])
    agg_ts['DatetimeEnd'] = pd.to_datetime(agg_ts['DatetimeEnd'])

    print('aggregated timeseries shape:{}'.format(agg_ts.shape))

    ydata_profiling.ProfileReport(agg_ts)

    ser_avail_days = agg_ts.groupby('SamplingPoint').nunique()['DatetimeBegin']

    df = agg_ts.loc[agg_ts.DataAggregationProcess=='P1D', :] 
    df = df.loc[df.UnitOfAirPollutionLevel!='count', :]
    df = df.loc[df.SamplingPoint.isin(ser_avail_days[ser_avail_days.values >= 1000].index), :]
    vars_to_drop = ['AirPollutant','AirPollutantCode','Countrycode','Namespace','TimeCoverage','Validity','Verification','AirQualityStation',
                'AirQualityStationEoICode','DataAggregationProcess','UnitOfAirPollutionLevel', 'DatetimeEnd', 'AirQualityNetwork',
                'DataCapture', 'DataCoverage']
    df.drop(columns=vars_to_drop, axis='columns', inplace=True)

    dates = list(pd.period_range(min(df.DatetimeBegin), max(df.DatetimeBegin), freq='D').values)
    samplingpoints = list(df.SamplingPoint.unique())

    new_idx = []
    for sp in samplingpoints:
        for d in dates:
            new_idx.append((sp, np.datetime64(d)))

    df.set_index(keys=['SamplingPoint', 'DatetimeBegin'], inplace=True)
    df.sort_index(inplace=True)
    df = df.reindex(new_idx)

    df['AirPollutionLevel'] = df.groupby(level=0).AirPollutionLevel.bfill().fillna(0)
    print('{} missing values'.format(df.isnull().sum().sum()))

    df = df.loc['SPO-BETR223_00001_100',:]

    train = df.query('DatetimeBegin < "2014-07-01"')
    valid = df.query('DatetimeBegin >= "2014-07-01" and DatetimeBegin < "2015-01-01"')
    test = df.query('DatetimeBegin >= "2015-01-01"')

    # Save column names and indices to use when storing as csv
    cols = train.columns
    train_idx = train.index
    valid_idx = valid.index
    test_idx = test.index

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    train = scaler.fit_transform(train)
    valid = scaler.transform(valid)
    test = scaler.transform(test)

    train = pd.DataFrame(train, columns=cols, index=train_idx)
    valid = pd.DataFrame(valid, columns=cols, index=valid_idx)
    test = pd.DataFrame(test, columns=cols, index=test_idx)

    #train.to_csv('data/processed/train.csv')
    #valid.to_csv('data/processed/valid.csv')
    #test.to_csv('data/processed/test.csv')

    return train, valid, test