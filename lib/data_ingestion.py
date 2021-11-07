#!/usr/bin/python
"""
Collection of functions for data ingestion and datasets modification.
"""
#Libraries Import
import time
import re
import dateparser
import pandas as pd


def remove_partenthesis(row):
    regex = re.compile(".*?\((.*?)\)")
    result = re.findall(regex, row)

    ret = row
    for t in result:
        ret = row.replace(t, '')

    ret = ret.replace('(', '')
    ret = ret.replace(')', '')
    ret = ret.replace('[', '')
    ret = ret.replace(']', '')
    ret = ret.rstrip()

    return ret


def remove_MoM_QoQ(row):
    row = row.replace('MoM', '')
    row = row.replace('QoQ', '')

    row = row.rstrip()

    return row


def convert(el):
    if isinstance(el, str):
        el = float(el.replace("M", "e+06") \
                   .replace(".0B", "e+09") \
                   .replace("K", "e+03") \
                   .replace("%", "") \
                   .replace(",", ".") \
                   .replace(".00", ""))
    elif isinstance(el, int):
        el = float(el)

    return el


if __name__ == "__main__":
    # Runtime initiation
    run_start = time.time()
    print('Processing raw data...')

    # Raw data import
    raw_sp500 = pd.read_csv('data/SP500_scrape.csv')
    raw_events= pd.read_excel('data/Events_scrape.xlsx')

    # S&P500 DF modifications
    sp500 = raw_sp500.copy()
    sp500.replace(to_replace=',', value='', regex=True, inplace=True)
    sp500 = sp500[['Open', 'High', 'Low', 'Close', 'Volume']].astype('float64')
    sp500['Date'] = raw_sp500['Date'].apply(lambda x: dateparser.parse(x).date())  # Parse to datetime
    sp500.sort_values(by='Date', ascending=True, inplace=True)
    sp500.reset_index(inplace=True, drop=True)

    # Events DF modifications
    events = raw_events.copy()
    events['select'] = raw_events['Time'].str[:]
    events['select_2'] = events['select'].fillna(method="ffill")
    events.dropna(axis=0, subset=['Event', 'Time'], inplace=True)
    events['Event'] = events['Event'].apply(remove_partenthesis).apply(remove_MoM_QoQ)
    to_drop = events.loc[events['select_2'] == 'All Day']
    events.drop(index=to_drop.index, inplace=True)
    to_drop_2 = events[events['select_2'].str.len() < 5]
    events.drop(index=to_drop_2.index, inplace=True)
    to_drop_3 = events.loc[events['select_2'] == 'Tentative']
    events.drop(index=to_drop_3.index, inplace=True)
    events = events.drop(events[events['Actual'].isna()].index)
    to_drop_4 = events[events['Actual'].str.len() < 2]
    events.drop(index=to_drop_4.index, inplace=True)
    events['Event'] = events['Event'].str.split('(').str[0]
    events['Event'] = events['Event'].str[:-1]
    events['Date'] = events['select_2'].apply(lambda x: dateparser.parse(x).date())  # Parse to datetime
    events.drop(columns=['Cur.', 'Imp.', 'Forecast', 'Previous', 'select', 'select_2',], inplace=True)
    events = events[['Date', 'Event', 'Actual']]
    events.reset_index(inplace=True, drop=True)

    # Events pivot table
    events_pivot_actual = events.pivot_table(values='Actual', index=['Date'],
                                             columns=['Event'], aggfunc=lambda x: x)
    events_pivot_actual = events_pivot_actual.applymap(convert)
    to_drop_5 = events_pivot_actual[events_pivot_actual['Building Permit'].str.len() < 2]
    events_pivot_actual.drop(index=to_drop_5.index, inplace=True)
    to_drop_6 = events_pivot_actual[events_pivot_actual['New Home Sale'].str.len() < 2]
    events_pivot_actual.drop(index=to_drop_6.index, inplace=True)
    events_pivot_actual = events_pivot_actual.astype('float64')

    # Merge datasets
    sp500_calendar = sp500.merge(events_pivot_actual, how='outer', left_on='Date',
                                  right_index=True)
    sp500_calendar = sp500_calendar.set_index('Date')
    sp500_calendar = sp500_calendar.select_dtypes(include=['float64']).fillna(method='ffill')

    # Evaluate runtime
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print('Dataset processing finished in:', '%d:%02d:%02d'%(h, m, s))

    a=1
