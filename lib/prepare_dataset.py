#!/usr/bin/python
"""
Collection of functions for data ingestion and datasets modification.
"""
#Libraries Import
import time
import re
import dateparser
import pandas as pd


def remove_parenthesis(row):
    """
    Function for parentheses removal.
    Credit: https://medium.com/analytics-vidhya/how-to-find-the-correlation-between-the-s-p500-and-the-economic-calendar-using-python-7c29c4faa8ff
    """
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
    """
    Function for MoM and QoQ abbreviation removal.
    Credit: https://medium.com/analytics-vidhya/how-to-find-the-correlation-between-the-s-p500-and-the-economic-calendar-using-python-7c29c4faa8ff
    """
    row = row.replace('MoM', '')
    row = row.replace('QoQ', '')

    row = row.rstrip()

    return row


def convert(el):
    """
    Function for units conversion into scientific numcer format.
    Credit: https://medium.com/analytics-vidhya/how-to-find-the-correlation-between-the-s-p500-and-the-economic-calendar-using-python-7c29c4faa8ff
    """
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

def modify_sp500(raw):
    """
    Function cleans and modify S&P500 raw data into usable format for further
    analysis.
    :param raw: S&P500 raw data
    :return: S&P500 cleaned DF
    """
    df_mod = raw.copy()
    df_mod.replace(to_replace=',', value='', regex=True, inplace=True)
    df_mod = df_mod[['Open', 'High', 'Low', 'Close', 'Volume']].astype('float64')
    df_mod['Date'] = raw['Date'].apply(lambda x: dateparser.parse(x).date())  # Parse to datetime
    df_mod.sort_values(by='Date', ascending=True, inplace=True)
    df_mod.reset_index(inplace=True, drop=True)

    return df_mod

def modify_evens(raw):
    """
    Function cleans and modify calendar events raw data into usable format for further
    analysis.
    :param raw: Events raw data
    :return: Events cleaned DF
    """
    df_mod = raw.copy()
    df_mod['select'] = raw['Time'].str[:]
    df_mod['select_2'] = df_mod['select'].fillna(method="ffill")
    df_mod.dropna(axis=0, subset=['Event', 'Time'], inplace=True)
    df_mod['Event'] = df_mod['Event'].apply(remove_parenthesis).apply(remove_MoM_QoQ)
    to_drop = df_mod.loc[df_mod['select_2'] == 'All Day']
    df_mod.drop(index=to_drop.index, inplace=True)
    to_drop_2 = df_mod[df_mod['select_2'].str.len() < 5]
    df_mod.drop(index=to_drop_2.index, inplace=True)
    to_drop_3 = df_mod.loc[df_mod['select_2'] == 'Tentative']
    df_mod.drop(index=to_drop_3.index, inplace=True)
    df_mod = df_mod.drop(df_mod[df_mod['Actual'].isna()].index)
    to_drop_4 = df_mod[df_mod['Actual'].str.len() < 2]
    df_mod.drop(index=to_drop_4.index, inplace=True)
    df_mod['Event'] = df_mod['Event'].str.split('(').str[0]
    df_mod['Event'] = df_mod['Event'].str[:-1]
    df_mod['Date'] = df_mod['select_2'].apply(lambda x: dateparser.parse(x).date())  # Parse to datetime
    df_mod.drop(columns=['Cur.', 'Imp.', 'Forecast', 'Previous', 'select', 'select_2',], inplace=True)
    df_mod = df_mod[['Date', 'Event', 'Actual']]
    df_mod.reset_index(inplace=True, drop=True)

    return df_mod

def create_pivot(df):
    """
    Function pivot table from calendar events data
    :param df: Calendar events data
    :return: Pivoted Df
    """
    df_pivot = df.pivot_table(values='Actual', index=['Date'],
                                             columns=['Event'], aggfunc=lambda x: x)
    df_pivot = df_pivot.applymap(convert)
    to_drop_5 = df_pivot[df_pivot['Building Permit'].str.len() < 2]
    df_pivot.drop(index=to_drop_5.index, inplace=True)
    to_drop_6 = df_pivot[df_pivot['New Home Sale'].str.len() < 2]
    df_pivot.drop(index=to_drop_6.index, inplace=True)
    df_pivot = df_pivot.astype('float64')

    return df_pivot


if __name__ == "__main__":
    # Runtime initiation
    run_start = time.time()
    print('Processing raw data...')

    # Raw data import
    raw_sp500 = pd.read_csv('data/SP500_scrape.csv')
    raw_events= pd.read_excel('data/Events_scrape.xlsx')

    # S&P500 DF modifications
    sp500 = modify_sp500(raw_sp500)

    # Events DF modifications
    events = modify_evens(raw_events)

    # Events pivot table
    events_pivot_actual = create_pivot(events)

    # Merge datasets
    sp500_calendar = sp500.merge(events_pivot_actual, how='outer', left_on='Date',
                                  right_index=True)
    sp500_calendar = sp500_calendar.set_index('Date')
    sp500_calendar = sp500_calendar.select_dtypes(include=['float64']).fillna(method='ffill')

    # Save dataset
    sp500_calendar.to_pickle('data/sp500_calendar.pickle')

    # Evaluate runtime
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print('Dataset processing finished in:', '%d:%02d:%02d'%(h, m, s))
