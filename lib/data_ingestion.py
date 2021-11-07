#!/usr/bin/python
"""
Collection of functions for data ingestion and datasets modification.
"""
#Libraries Import
import time
import dateparser
import pandas as pd


if __name__ == "__main__":
    # Runtime initiation
    run_start = time.time()
    print('Processing raw data...')

    # Raw data import
    # raw_dates = pd.read_csv('data/Days_scrape.csv')
    # raw_events = pd.read_csv('data/Events_scrape.csv')
    raw_sp500 = pd.read_csv('data/SP500_scrape.csv')
    raw_events= pd.read_excel('data/Events_full.xlsx')

    # S&P500 DF modifications
    sp500 = raw_sp500.copy()
    sp500['Date'] = raw_sp500['Date'].apply(lambda x: dateparser.parse(x).date())  # Parse to datetime
    sp500.sort_values(by='Date', ascending=True, inplace=True)
    sp500.reset_index(inplace=True, drop=True)

    # Events DF modifications
    events = raw_events.copy()
    events['select'] = raw_events['Time'].str[:]
    events['select_2'] = events['select'].fillna(method="ffill")
    events.dropna(axis=0, subset=['Event', 'Time'], inplace=True)
    to_drop = events.loc[events['select_2'] == 'All Day']
    events.drop(index=to_drop.index, inplace=True)
    to_drop_2 = events[events['select_2'].str.len() < 5]
    events.drop(index=to_drop_2.index, inplace=True)
    to_drop_3 = events.loc[events['select_2'] == 'Tentative']
    events.drop(index=to_drop_3.index, inplace=True)
    events['Event'] = events['Event'].str.split('(').str[0]
    events['Event'] = events['Event'].str[:-1]
    events['Date'] = events['select_2'].apply(lambda x: dateparser.parse(x).date())  # Parse to datetime
    events.drop(columns=['Cur.', 'Imp.', 'Actual', 'Forecast', 'Previous', 'select', 'select_2'], inplace=True)
    events = events[['Date', 'Time', 'Event']]
    events.reset_index(inplace=True, drop=True)
    a=1

    # Events DF modifications
    # raw_events.replace('Tentative', '23:00', inplace=True)  # Replace 'tentative' string
    # raw_events.replace(':00', '', inplace=True)  # Replace zeros
    # raw_events['Time_int'] = raw_events['Time'].str.split(':').str[0].astype(int)  # Convert to integer

    # raw_events['dT'] = raw_events['Time_int'] - raw_events['Time_int'].shift(1)

    # print(raw_events['dT'].where(raw_events['dT'] < 0).count())

    # raw_events['Time'] = raw_events['Time'].apply(lambda x: dateparser.parse(x).time())  # Parse to datetime
    # raw_events['Time'] = raw_events['Time'].astype(float)  # Parse to datetime
    # raw_events['Seconds'] = raw_events['Time'].dt.total_seconds()

    # raw_events['dT'] = raw_events.Time - raw_events.Time.shift()
    # raw_events['dT'] = raw_events['Time'] - raw_events['Time'].shift(1)

    # Time difference
    # raw_events['dT'] = raw_events.Time.diff()
    # raw_events['dT'] = raw_events.Time - raw_events.Time.shift()
    # raw_events['Difference'] = np.where(raw_events.Time == raw_events.Time.shift(), raw_events.DateTime - raw_events.DateTime.shift(), np.nan)

    # Evaluate runtime
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print('Dataset processing finished in:', '%d:%02d:%02d'%(h, m, s))

    a=1
