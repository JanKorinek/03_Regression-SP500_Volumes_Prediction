#!/usr/bin/python
"""
Collection of functions for data ingestion and datasets modification.
"""
#Libraries Import
import time
import pandas as pd


if __name__ == "__main__":
    # Runtime initiation
    run_start = time.time()

    # Raw data import
    raw_dates = pd.read_csv('data/Days_scrape.csv')
    raw_events = pd.read_csv('data/Events_scrape.csv')
    raw_sp500 = pd.read_csv('data/SP500_scrape.csv')

    # Evaluate runtime
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print('Dataset processing time:', '%d:%02d:%02d'%(h, m, s))

    a=1
