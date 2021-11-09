#!/usr/bin/python
"""
Collection of functions for adding features into dataset
"""
#Libraries Import
import time
import pandas as pd



if __name__ == "__main__":
    # Runtime initiation
    run_start = time.time()
    print('Calculating features...')

    # Raw data import
    sp500 = pd.read_pickle('data/sp500_fill.pickle')


    # Save dataset
    # sp500_calendar.to_pickle('data/sp500_calendar.pickle')

    # Evaluate runtime
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print('Features calculation finished in:', '%d:%02d:%02d'%(h, m, s))

    a=1
