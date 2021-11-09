#!/usr/bin/python
"""
Collection of functions for adding features into dataset
"""
#Libraries Import
import time
import numpy as np
import pandas as pd


def alphas_exceptions(input_params):
    """
    Function checks if input data are in desired format.
    :return: Raise exception if necessary
    """
    # Exception check of the column label data type
    if type(input_params['price']) not in [str]:
        raise TypeError('Price label must be a string.')

    if type(input_params['rolling_window']) not in [int]:
        raise TypeError('Rolling Window label must be a integer only.')
    
def compute_returns(df):
    """
    Calculates returns based on close price movement.
    :return: Updated df about returns for OHLC.
    """
    # Returns computation
    df['returns'] = np.log(df['Close'] / df['Close'].shift(1))

    df.dropna(inplace=True)

    return df

def compute_simple_moving_average(df, input_params):
    """
    Function calculates simple moving average based on selected price.
    :return: Updated df about simple moving average.
    """
    # Exception checks of the input data
    alphas_exceptions(input_params)

    # SMA computation
    df['sma'] = df[input_params['price']].rolling(input_params['rolling_window']).mean()

    return df

def compute_rolling_minimum(df, input_params):
    """
    Function calculates rolling minimum based on selected price.
    :return: Updated df about rolling minimum.
    """
    # Exception checks of the input data
    alphas_exceptions(input_params)

    df['roll_min'] = df[input_params['price']].rolling(input_params['rolling_window']).min()

    return df

def compute_rolling_maximum(df, input_params):
    """
    Function calculates rolling maximum based on selected price.
    :return: Updated df about rolling minimum.
    """
    # Exception checks of the input data
    alphas_exceptions(input_params)

    df['roll_max'] = df[input_params['price']].rolling(input_params['rolling_window']).max()

    return df

def compute_rolling_volatility(df, input_params):
    """
    Function calculates rolling volatility based on log returns.
    :return: Updated df about rolling volatility.
    """

    # Exception check of the column label data type
    if type(input_params['rolling_window']) not in [int]:
        raise TypeError('Rolling Window label must be a integer only.')

    df['roll_vol'] = df['returns'].rolling(input_params['rolling_window']).std()

    return df

def compute_momentum(df, input_params):
    """
    Function calculates rolling momentum based on log returns.
    :return: Updated df about rolling momentum.
    """
    # Exception check of the column label data type
    if type(input_params['rolling_window']) not in [int]:
        raise TypeError('Rolling Window label must be a integer only.')

    df['roll_mom'] = df['returns'].rolling(input_params['rolling_window']).mean()

    return df

def create_lags(df, input_params):
    """
    Function creates lagged data of selected features by nr. of desired rows.
    :return: Updated financial data df
    """
    # Exception check of the required nr. of lags data type
    if type(input_params['lags']) not in [int]:
        raise TypeError('Nr. of lags parameter must be a int only.')

    cols = []
    for f in df.columns.tolist():
        for lag in range(1, input_params['lags'] + 1):
            col = f'{f}_lag_{lag}'  # Lagged feature column name
            df[col] = df[f].shift(lag)  # Create shift to apply lag
            cols.append(col)

    df.dropna(inplace=True)

    return df

if __name__ == "__main__":
    # Runtime initiation
    run_start = time.time()
    print('Calculating features...')
    
    # General Parameters
    input_params = {
        'signed_flow_volume': 'Volume',
        'price': 'Close',
        'rolling_window': 100,
        'lags': 5,  # Nr. of required lags
    }
    
    # Raw data import
    sp500 = pd.read_pickle('data/sp500_fill.pickle')

    # Features calculation
    sp500 = compute_returns(sp500)  # Returns computation
    sp500 = compute_simple_moving_average(sp500, input_params)  # SMA computation
    sp500 = compute_rolling_minimum(sp500, input_params)  # Rolling minimum computation
    sp500 = compute_rolling_maximum(sp500, input_params)  # Rolling maximum computation
    sp500 = compute_rolling_volatility(sp500, input_params)  # Rolling volatility computation
    sp500 = compute_momentum(sp500, input_params)  # Rolling volatility computation

    # Drop NA rows
    sp500.dropna(inplace=True)

    # Create lags
    sp500 = create_lags(sp500, input_params)

    # Save dataset
    sp500.to_pickle('data/sp500_features.pickle')

    # Evaluate runtime
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print('Features calculation finished in:', '%d:%02d:%02d'%(h, m, s))
