# Libraries import
import time, joblib, warnings, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from learning_curve import plot_learning_curve

# Warnings turn off
warnings.simplefilter('ignore', np.RankWarning)
warnings.filterwarnings('ignore')

plt.style.use('seaborn')

## model specific variables (iterate the version and note with each change)
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "Supervised learning model for S&P500 time-series"

def split_dataset(df):
    """
    Function for splitting dataset and selecting evaluation data between
    01-01-2017 and 31-12-2018.
    """
    # Selecting training dataset
    to_date = datetime.date(2016, 12, 30)
    df_train = df.loc[:to_date]

    X_train = df_train.drop(columns=['Volume'])
    y_train = df_train[['Volume']].values.ravel()

    # Selecting evaluation dataset
    from_date = datetime.date(2017, 1, 3)
    to_date = datetime.date(2018, 12, 31)
    df_test = df.loc[from_date:to_date]

    X_test = df_test.drop(columns=['Volume'])
    y_test = df_test[['Volume']].values.ravel()

    return X_train, X_test, y_train, y_test

def train_random_forest_regr(X_train, X_test, y_train, y_test, pipe, param_grid, params_gen):
    """
    Funtion to train Random Forest Regressor
    """
    ## start timer for runtime
    time_start = time.time()

    # Defining splitter
    splitter = TimeSeriesSplit(n_splits=params_gen['n_splits'])

    rand = RandomizedSearchCV(pipe,
                              param_distributions = param_grid,
                              n_iter = params_gen['n_iter'],
                              cv = splitter,
                              verbose=2,
                              random_state=42,
                              n_jobs = -1)
    rand.fit(X_train, y_train)
    y_pred = rand.predict(X_test)

    # Evaluation metrics
    eval_rmse = round(np.sqrt(mean_squared_error(y_test,y_pred)))
    eval_rmae = round(np.sqrt(mean_absolute_error(y_test, y_pred)))
    eval = [eval_rmse, eval_rmae]

    # Save the model
    joblib.dump(rand, 'models/rfreg_model.joblib')

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    print('Random Forest Tree Model training finished in:', '%d:%02d:%02d'%(h, m, s))

    return rand, eval

def plot_model_comparison(rf_model, X_train, y_train, input_params):
    """
    Plotting comparison of learning curves for selected models.
    """
    # Define subplots
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    # Set titles
    ax1.set_title("Random Forest Tree Regression")
    ax2.set_title("ElasticNet Regression")
    ax3.set_title("Random Forest Tree Regression")
    ax4.set_title("Support Vector Regression")

    # Define axes limits
    # for ax in [ax1, ax2, ax3, ax4]:
    #     ax.set_ylim((0.1, 1.2))

    # Plot learning curves
    plot_learning_curve(rf_model, X_train, y_train, ax3, input_params)
    # plot_learning_curve(ls_model, ls_data['X'], ls_data['y'], ax=ax1)
    # plot_learning_curve(en_model, en_data['X'], en_data['y'], ax=ax2)
    # plot_learning_curve(svr_model, svr_data['X'], svr_data['y'], ax=ax4)

    plt.savefig('export/learning_curves_comparison.pdf', dpi=600)
    plt.show()

if __name__ == "__main__":
    # Runtime initiation
    run_start = time.time()
    print('Training all models...')

    # General Parameters
    input_params = {
        'n_splits': 4,
        'n_iter': 2,
    }

    # Data import
    sp500 = pd.read_pickle('data/sp500_features.pickle')

    # Prepare train-test split
    X_train, X_test, y_train, y_test = split_dataset(sp500)

    # RandomForest RandomSearch hyperparameters and pipeline
    rf = RandomForestRegressor(random_state=42)

    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    param_rand_rf = {
        'rf__bootstrap': [True, False],
        'rf__criterion': ['squared_error'],
        'rf__n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
        'rf__max_features': ['auto', 'sqrt'],
        'rf__max_depth': max_depth,
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4],
    }

    pipe_rf = Pipeline(steps=[('scaler', StandardScaler()),
                            ('rf', RandomForestRegressor())])

    # Train the model
    rf_model, rf_eval = train_random_forest_regr(X_train, X_test, y_train,
                                              y_test, pipe_rf, param_rand_rf, input_params)

    plot_model_comparison(rf_model, X_train, y_train, input_params)

    # Evaluate runtime
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print('All models trained in:', '%d:%02d:%02d'%(h, m, s))

    a=1

