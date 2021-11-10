#!/usr/bin/python
"""
Collection of support functions for EDA.
"""
# Libraries import
import time
import copy
import datetime
import numpy as np
import pandas as pd
import statsmodels.tsa.api as tsa
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from scipy.stats import probplot, moment

pd.set_option("display.max.columns", None)
plt.style.use('seaborn')

SMALL_SIZE = 12
MEDIUM_SIZE = 14
LARGE_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)   # fontsize of the figure title

def print_missing_vals(df):
    """
    Function print summaries of missing values in dataset and visualize them
    in pie charts.
    :param df: Df with missing values
    :return: Prints summary and plots pie charts.
    """
    # Missing values summary
    df_work = copy.deepcopy(df)
    df_work.replace(0, np.nan, inplace=True)
    for key in df_work.columns:
        # Overall fractions calculation
        tot_val = df_work.shape[0]
        tot_val_nan = df_work[key].isna().sum(axis=0)

        print(f"{key} - Missing Value Summary:")
        print('-' * 50)
        print(tot_val_nan, '\n')

        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        labels = f'Total Values ({tot_val})', f'Total Missing Values ({tot_val_nan})'
        explode = (0, 0.1,)  # only "explode" the 2nd slice
        theme = plt.get_cmap('Paired')

        fig1, ax1 = plt.subplots()
        sizes = [(tot_val - tot_val_nan), tot_val_nan]

        ax1.set_prop_cycle("color", [theme(1. * i / len(sizes))
                                     for i in range(len(sizes))])
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax1.set_title(f"{key} - Missing Values\n", fontsize=LARGE_SIZE, fontweight='bold')

        fig1.set_facecolor('xkcd:white')
        plt.savefig(f'export/pie_missing_val_{key}.pdf', dpi=200)
    
    return df_work

def print_corr_matrix(df):
    """
    Function plots features correlation matrix from prepared dataframe.
    :param df: Df with features
    :return: Correlation matrix plot
    """
    # Clean and prepare df for correlation matrix plot
    df_cat = df.copy(deep=True)
    df_cat = df_cat.dropna()

    # Compute the correlation matrix
    corr = df_cat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(20, 18))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Plot the matrix
    sns.heatmap(corr, mask=mask, annot=True, square=True, linewidths=.5, vmin=-1, vmax=1, cmap=cmap, ax=ax)
    plt.title("S&P500 Correlation matrix", fontsize=MEDIUM_SIZE, fontweight='bold')
    plt.xlabel("Features", fontweight='bold', fontsize=SMALL_SIZE, )
    plt.ylabel("Features", fontweight='bold', fontsize=SMALL_SIZE, )
    plt.tight_layout()
    plt.savefig('export/sp500_corr_matrix.pdf', dpi=1000)
    plt.show()

    return corr

def fix_inconsistency(df):
    """
    Function fixes value falls detected by personal observation of the feature time-series.
    Therefore selection ranges are hardcoded.
    :param df: Features dataset
    :return: Fixed values Df
    """
    # Fixing values inconsistency
    from_date = datetime.date(2003, 4, 30)
    to_date = datetime.date(2006, 7, 31)
    df_fill = df.copy()
    df_fill['New Home Sale'].loc[from_date:to_date] = df['New Home Sale'].loc[from_date]

    from_date = datetime.date(2002, 9, 30)
    to_date = datetime.date(2003, 3, 31)
    df_fill['New Home Sale'].loc[from_date:to_date] = df['New Home Sale'].loc[from_date]

    from_date = datetime.date(2006, 8, 31)
    to_date = datetime.date(2006, 10, 31)
    df_fill['New Home Sale'].loc[from_date:to_date] = df['New Home Sale'].loc[from_date]

    from_date = datetime.date(2006, 12, 1)
    to_date = datetime.date(2007, 1, 31)
    df_fill['New Home Sale'].loc[from_date:to_date] = df['New Home Sale'].loc[from_date]

    from_date = datetime.date(2006, 11, 30)
    to_date = datetime.date(2007, 1, 31)
    df_fill['New Home Sale'].loc[from_date:to_date] = df['New Home Sale'].loc[from_date]

    from_date = datetime.date(2008, 10, 15)
    to_date = datetime.date(2008, 10, 22)
    df_fill['Initial Jobless Claim'].loc[from_date:to_date] = df['Initial Jobless Claim'].loc[from_date]

    from_date = datetime.date(2000, 1, 31)
    to_date = datetime.date(2000, 2, 29)
    df_fill['Unemployment Rat'].loc[from_date:to_date] = df['Unemployment Rat'].loc[from_date]

    from_date = datetime.date(2000, 5, 31)
    to_date = datetime.date(2000, 8, 31)
    df_fill['Unemployment Rat'].loc[from_date:to_date] = df['Unemployment Rat'].loc[from_date]

    from_date = datetime.date(2001, 9, 28)
    to_date = datetime.date(2001, 10, 31)
    df_fill['Unemployment Rat'].loc[from_date:to_date] = df['Unemployment Rat'].loc[from_date]

    from_date = datetime.date(2003, 4, 30)
    to_date = datetime.date(2003, 6, 30)
    df_fill['Unemployment Rat'].loc[from_date:to_date] = df['Unemployment Rat'].loc[from_date]

    from_date = datetime.date(2005, 6, 30)
    to_date = datetime.date(2005, 8, 31)
    df_fill['Unemployment Rat'].loc[from_date:to_date] = df['Unemployment Rat'].loc[from_date]

    from_date = datetime.date(2005, 10, 31)
    to_date = datetime.date(2006, 1, 31)
    df_fill['Unemployment Rat'].loc[from_date:to_date] = df['Unemployment Rat'].loc[from_date]

    from_date = datetime.date(2008, 1, 31)
    to_date = datetime.date(2008, 3, 31)
    df_fill['Unemployment Rat'].loc[from_date:to_date] = df['Unemployment Rat'].loc[from_date]

    from_date = datetime.date(2008, 4, 30)
    to_date = datetime.date(2008, 6, 30)
    df_fill['Unemployment Rat'].loc[from_date:to_date] = df['Unemployment Rat'].loc[from_date]

    return df_fill

def seasonal_decomposition(df, column, period):
    """
    Function for seasonal decomposition of the time series
    :param df: Df with time-series data
    :param column: (str) name of the desired column
    :param period: (int) period for decomposition
    :return: Time-series and seasonal decomposition plot
    """
    series = df[column]
    series.index = pd.to_datetime(series.index)
    components = tsa.seasonal_decompose(series, model='additive', period=period)

    ts = (series.to_frame('Original')
          .assign(Trend=components.trend)
          .assign(Seasonality=components.seasonal)
          .assign(Residual=components.resid))
    with sns.axes_style('white'):
        ts.plot(subplots=True, figsize=(14, 8),
                title=['Original Series', 'Trend Component', 'Seasonal Component', 'Residuals'], legend=False)
        plt.suptitle('Seasonal Decomposition', fontsize=14)
        sns.despine()
        plt.tight_layout()
        plt.subplots_adjust(top=.91)
        plt.savefig(f'export/{column}_seasonal_decomposition.pdf', dpi=600)

    return series

def stationarity_check(TS):
    """
    Checking stationarity of the time-series
    :param TS: time-series
    :return: Plot
    """
    # Calculate rolling statistics
    rolmean = TS.rolling(window = 100, center = False).mean()
    rolstd = TS.rolling(window = 18, center = False).std()

    # Perform the Dickey Fuller Test
    dftest = adfuller(TS) # change the passengers column as required

    # Plot rolling statistics:
    fig = plt.figure(figsize=(12,6))
    orig = plt.plot(TS, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Print Dickey-Fuller test results
    print ('Results of Dickey-Fuller Test:')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

    return None

def print_score(data, preffix, y_lim, color):
    """
    Function prints score comparison from multiple models.
    :param data: (dict) Score values
    :param preffix: Scoring method
    :return: Bar plot
    """
    # Extract models and score values
    models = list(data.keys())
    values = list(data.values())

    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(models, values, width=0.4, color=color)

    plt.ylabel(preffix, fontsize=SMALL_SIZE)
    plt.title(f"{preffix} Score Comparison of the Trained Models", fontsize=LARGE_SIZE)
    plt.ylim(y_lim)
    plt.savefig(f'export/{preffix}_scores.pdf', dpi=300)
    plt.show()

def concat_df(main, pred, preffix):
    """
    Function adds predicted volumes to evaluation dataframe
    :param main: Df with evaluation data
    :param pred: Predicted volume
    :return: Concatenated Df
    """
    pred_res = pred.reshape(pred.shape[0],1)
    pred_df = pd.DataFrame(data=pred_res, index=main.index, columns=[f'Volume{preffix}'])
    con = pd.concat([main, pred_df], axis=1, ignore_index=False)

    return con


def plot_correlogram(x, lags=None, title=None):
    """
    For correlogram printing
    :param x: Time-series
    :param lags: (int) Nr. of lags
    :param title:
    :return:
    """
    lags = min(10, int(len(x)/5)) if lags is None else lags
    with sns.axes_style('whitegrid'):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
        x.plot(ax=axes[0][0], title='Residuals')
        x.rolling(21).mean().plot(ax=axes[0][0], c='k', lw=1)
        q_p = np.max(q_stat(acf(x, nlags=lags), len(x))[1])
        stats = f'Q-Stat: {np.max(q_p):>8.2f}\nADF: {adfuller(x)[1]:>11.2f}'
        axes[0][0].text(x=.02, y=.85, s=stats, transform=axes[0][0].transAxes)
        probplot(x, plot=axes[0][1])
        mean, var, skew, kurtosis = moment(x, moment=[1, 2, 3, 4])
        s = f'Mean: {mean:>12.2f}\nSD: {np.sqrt(var):>16.2f}\nSkew: {skew:12.2f}\nKurtosis:{kurtosis:9.2f}'
        axes[0][1].text(x=.02, y=.75, s=s, transform=axes[0][1].transAxes)
        plot_acf(x=x, lags=lags, zero=False, ax=axes[1][0])
        plot_pacf(x, lags=lags, zero=False, ax=axes[1][1])
        axes[1][0].set_xlabel('Lag')
        axes[1][1].set_xlabel('Lag')
        fig.suptitle(title, fontsize=14)
        sns.despine()
        fig.tight_layout()
        fig.subplots_adjust(top=.9)
        plt.savefig(f'export/{title}_.pdf', dpi=600)


if __name__ == "__main__":
    # Runtime initiation
    run_start = time.time()
    print('Training all models...\n')

    # Evaluate runtime
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print('\nAll models trained in:', '%d:%02d:%02d'%(h, m, s))