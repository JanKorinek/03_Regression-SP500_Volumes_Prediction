#!/usr/bin/python
"""
Collection of functions for training multiple models with RandomizedSearch .
"""
# Libraries import
import time, joblib, warnings, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost.sklearn import XGBRegressor
from sklearn.utils.fixes import loguniform

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

def plot_pred_val(X_test, y_test, y_pred, preffix):
    """
    For plotting ground truth and predicted values.
    """
    # Plot the predicted values
    fig, ax = plt.subplots(figsize=(18, 10),)
    ax.plot(X_test.index, y_test, label='Ground Truth')
    ax.plot(X_test.index, y_pred, label='Prediction')
    ax.set_xlabel('Date', fontweight ="bold")
    ax.set_ylabel('Volume', fontweight ="bold")
    ax.set_title(f'S&P500 Volume Prediction - {preffix}_Regressor', fontweight ="bold",
                 fontsize=20)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'export/sp500_volume_prediction-{preffix}_Regressor.pdf', dpi=1000)
    plt.show()

def train_model(X_train, X_test, y_train, y_test, pipe, param_grid,
                             params_gen, preffix):
    """
    Funtion to train the model
    """
    print(f'\nTraining the {preffix} model...\n')

    # Start timer for runtime
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

    # Plot the predicted values
    plot_pred_val(X_test, y_test, y_pred, preffix)

    # Plotting additional information
    print(f'\n------ The best parameters of the {preffix} model are: ------')
    print(rand.best_estimator_)
    print('-'*50)
    print(f'The best cross-validation score: {rand.best_score_}')
    print(f'MSE: {eval_rmse}')
    print(f'MAE: {eval_rmae}')
    print('-' * 50)

    # Save the model
    joblib.dump(rand, f'models/{preffix}_model.joblib')

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    print(f'\n{preffix} model training finished in:', '%d:%02d:%02d'%(h, m, s))

    return rand, eval, y_pred


def plot_model_comparison(models, X_train, y_train, input_params):
    """
    Plotting comparison of learning curves for selected models.
    """
    print(f'\nLearning curves plotting computation in process...\n')

    # Start timer for runtime
    time_start = time.time()

    # Define subplots
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    # Set titles
    ax1.set_title("Random Forest Tree Regression")
    ax2.set_title("Ada Boost Regression")
    ax3.set_title("Gradient Boost Regression")
    ax4.set_title("XGBoost Regression")

    # Define axes limits
    # for ax in [ax1, ax2, ax3, ax4]:
    #     ax.set_ylim((0.1, 1.2))

    # Plot learning curves
    plot_learning_curve(models[0], X_train, y_train, ax1, input_params)
    plot_learning_curve(models[1], X_train, y_train, ax2, input_params)
    plot_learning_curve(models[2], X_train, y_train, ax3, input_params)
    plot_learning_curve(models[3], X_train, y_train, ax4, input_params)

    plt.savefig('export/learning_curves_comparison.pdf', dpi=600)
    plt.show()

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    print(f'\nLearning curves plotted in:', '%d:%02d:%02d'%(h, m, s))

if __name__ == "__main__":
    # Runtime initiation
    run_start = time.time()
    print('Training all models...\n')

    # General Parameters
    input_params = {
        'n_splits': 3,
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
        'rf__n_estimators': [int(x) for x in np.linspace(start = 50, stop = 1000, num = 20)],
        'rf__max_features': ['auto', 'sqrt'],
        'rf__max_depth': max_depth,
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4],
    }
    pipe_rf = Pipeline(steps=[('scaler', StandardScaler()),
                            ('rf', RandomForestRegressor())])
    # Train the model
    rf_model, rf_eval, rf_y_pred = train_model(X_train, X_test, y_train, y_test, pipe_rf,
                                    param_rand_rf, input_params, 'Random_Forest_Tree')

    # AdaBoost RandomSearch hyperparameters and pipeline
    ab = AdaBoostRegressor(random_state=42)
    param_rand_ab = {
        'ab__n_estimators': [int(x) for x in np.linspace(start = 50, stop = 1000, num = 20)],
        'ab__loss': ['linear', 'square', 'exponential'],
        'ab__learning_rate': loguniform(1e-3, 2e-1),
    }
    pipe_ab = Pipeline(steps=[('scaler', StandardScaler()),
                            ('ab', AdaBoostRegressor())])
    # Train the model
    ab_model, ab_eval, ab_y_pred = train_model(X_train, X_test, y_train, y_test, pipe_ab,
                                    param_rand_ab, input_params, 'Ada_Boost')

    # GradientBoost RandomSearch hyperparameters and pipeline
    gb = GradientBoostingRegressor(random_state=42)
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    param_rand_gb = {
        'gb__n_estimators': [int(x) for x in np.linspace(start = 50, stop = 1000, num = 20)],
        'gb__loss': ['squared_error', 'huber', 'quantile'],
        'gb__learning_rate': loguniform(1e-3, 2e-1),
        'gb__criterion': ['squared_error',],
        'gb__min_samples_split': [2, 5, 10],
        'gb__min_samples_leaf': [1, 2, 4],
        'gb__max_features': ['sqrt', 'log2'],
        'gb__max_depth': max_depth,
    }
    pipe_gb = Pipeline(steps=[('scaler', StandardScaler()),
                            ('gb', GradientBoostingRegressor())])
    # Train the model
    gb_model, gb_eval, gb_y_pred = train_model(X_train, X_test, y_train, y_test, pipe_gb,
                                    param_rand_gb, input_params, 'Gradient_Boost')

    # XGBoost RandomSearch hyperparameters and pipeline
    xgbm = XGBRegressor(random_state=42)

    param_rand_xgbm = {
        'xgbm__n_estimators': [int(x) for x in np.linspace(start = 50, stop = 1000, num = 20)],
        'xgbm__learning_rate': loguniform(1e-3, 2e-1),
        'xgbm__max_depth': range(4,8,2),
        'xgbm__objective': ['reg:squarederror'],
        'xgbm__subsample': [0.6, 0.8, 1.0],
        'xgbm__min_child_weight': [1, 2, 3, 4, 5],
        'xgbm__reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
        'xgbm__gamma': [i/10.0 for i in range(0, 5)],
        'xgbm__max_delta_step': range(1, 10, 1),
        'xgbm__booster': ['gbtree', 'gblinear'],
    }
    pipe_xgbm = Pipeline(steps=[('scaler', StandardScaler()),
                            ('xgbm', XGBRegressor())])

    # Train the model
    xgbm_model, xgbm_eval, xgbm_y_pred = train_model(X_train, X_test, y_train,
                                                     y_test, pipe_xgbm, param_rand_xgbm,
                                                     input_params, 'XGboost')
    
    # Plot model learning curves comparison
    models = [rf_model, ab_model, gb_model, xgbm_model]
    plot_model_comparison(models, X_train, y_train, input_params)

    # Evaluate runtime
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print('\nAll models trained in:', '%d:%02d:%02d'%(h, m, s))

