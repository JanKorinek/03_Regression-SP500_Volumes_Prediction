import time,os,re,csv,sys,uuid,joblib
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from numpy import arange

from logger import update_predict_log, update_train_log
from cslib import fetch_ts, engineer_features
from learning_curve import plot_learning_curve

# Warnings turn off
warnings.simplefilter('ignore', np.RankWarning)
warnings.filterwarnings('ignore')

plt.style.use('seaborn')

## model specific variables (iterate the version and note with each change)
MODEL_DIR = "models"
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "supervised learing model for time-series"

def _model_train(df,tag,pipe,param_grid,test=False):
    """
    example funtion to train model
    
    The 'test' flag when set to 'True':
        (1) subsets the data and serializes a test version
        (2) specifies that the use of the 'test' log file 

    """

    ## start timer for runtime
    time_start = time.time()
    
    X,y,dates = engineer_features(df)

    if test:
        n_samples = int(np.round(0.3 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(X.shape[0]),n_samples,
                                          replace=False).astype(int)
        mask = np.in1d(np.arange(y.size),subset_indices)
        y=y[mask]
        X=X[mask]
        dates=dates[mask]
        
    ## Perform a train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        shuffle=True, random_state=42)

    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    eval_rmse =  round(np.sqrt(mean_squared_error(y_test,y_pred)))
    
    ## retrain using all data
    grid.fit(X, y)
    model_name = re.sub("\.","_",str(MODEL_VERSION))
    if test:
        saved_model = os.path.join(MODEL_DIR,
                                   "test-{}-{}.joblib".format(tag,model_name))
        print("... saving test version of model: {}".format(saved_model))
    else:
        saved_model = os.path.join(MODEL_DIR,
                                   "sl-{}-{}.joblib".format(tag,model_name))
        print("... saving model: {}".format(saved_model))
        
    joblib.dump(grid,saved_model)

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update log
    update_train_log(tag,(str(dates[0]),str(dates[-1])),{'rmse':eval_rmse},runtime,
                     MODEL_VERSION, test=test)
  

def model_train(data_dir,pipe,param_grid,test=False):
    """
    funtion to train model given a df
    
    'mode' -  can be used to subset data essentially simulating a train
    """
    
    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if test:
        print("... test flag on")
        print("...... subseting data")
        print("...... subseting countries")
        
    ## fetch time-series formatted data
    ts_data = fetch_ts(data_dir)

    ## train a different model for each data sets
    for country,df in ts_data.items():
        
        if test and country not in ['all','united_kingdom']:
            continue
        
        _model_train(df,country,pipe,param_grid,test=test)
    
def model_load(prefix='sl',data_dir=None,training=True):
    """
    example funtion to load model
    
    The prefix allows the loading of different models
    """

    if not data_dir:
        data_dir = os.path.join("cs-train")
    
    models = [f for f in os.listdir(os.path.join(".","models")) if re.search("sl",f)]

    if len(models) == 0:
        raise Exception("Models with prefix '{}' cannot be found did you train?".format(prefix))

    all_models = {}
    for model in models:
        all_models[re.split("-",model)[1]] = joblib.load(os.path.join(".","models",model))

    ## load data
    ts_data = fetch_ts(data_dir)
    all_data = {}
    for country, df in ts_data.items():
        X,y,dates = engineer_features(df,training=training)
        dates = np.array([str(d) for d in dates])
        all_data[country] = {"X":X,"y":y,"dates": dates}
        
    return(all_data, all_models)

def model_predict(country,year,month,day,all_models=None,test=False):
    """
    example funtion to predict from model
    """

    ## start timer for runtime
    time_start = time.time()

    ## load model if needed
    if not all_models:
        all_data,all_models = model_load(training=False)
    
    ## input checks
    if country not in all_models.keys():
        raise Exception("ERROR (model_predict) - model for country '{}' could not be found".format(country))

    for d in [year,month,day]:
        if re.search("\D",d):
            raise Exception("ERROR (model_predict) - invalid year, month or day")
    
    ## load data
    model = all_models[country]
    data = all_data[country]

    ## check date
    target_date = "{}-{}-{}".format(year,str(month).zfill(2),str(day).zfill(2))
    print(target_date)

    if target_date not in data['dates']:
        raise Exception("ERROR (model_predict) - date {} not in range {}-{}".format(target_date,
                                                                                    data['dates'][0],
                                                                                    data['dates'][-1]))
    date_indx = np.where(data['dates'] == target_date)[0][0]
    query = data['X'].iloc[[date_indx]]
    
    ## sainty check
    if data['dates'].shape[0] != data['X'].shape[0]:
        raise Exception("ERROR (model_predict) - dimensions mismatch")

    ## make prediction and gather data for log entry
    y_pred = model.predict(query)
    y_proba = None
    if 'predict_proba' in dir(model) and 'probability' in dir(model):
        if model.probability == True:
            y_proba = model.predict_proba(query)


    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update predict log
    update_predict_log(country,y_pred,y_proba,target_date,
                       runtime, MODEL_VERSION, test=test)
    
    return({'y_pred':y_pred,'y_proba':y_proba})

if __name__ == "__main__":

    """
    basic test procedure for model_tune_hp.py
    """
    # Lasso hyperparameters
    param_grid_ls = {
    'ls__alpha': arange(0, 1, 0.1).tolist()
    }
    pipe_ls = Pipeline(steps=[('scaler', StandardScaler()),
                            ('ls', Lasso())])

    # ElasticNet hyperparameters
    param_grid_en = {
    'en__alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0],
    'en__l1_ratio': arange(0, 1, 0.1).tolist(),
    }
    pipe_en = Pipeline(steps=[('scaler', StandardScaler()),
                            ('en', ElasticNet())])

    # RandomForest hyperparameters
    param_grid_rf = {
    'rf__criterion': ['mse','mae'],
    'rf__n_estimators': [10,15,20,25]
    }
    pipe_rf = Pipeline(steps=[('scaler', StandardScaler()),
                            ('rf', RandomForestRegressor())])

    # SVR hyperparameters
    param_grid_svr = {
    'svr__C': [0.01, 0.1, 1, 10, 100, 1000],
    'svr__kernel': ['linear','rbf'],
    'svr__gamma': [0.001, 0.0001]
    }
    pipe_svr = Pipeline(steps=[('scaler', StandardScaler()),
                            ('svr', SVR())])

    grids = [param_grid_ls, param_grid_en, param_grid_rf, param_grid_svr]
    pipes = [pipe_ls, pipe_en, pipe_rf, pipe_svr]

    data = []
    models = []
    for param_grid, pipe in zip(grids,pipes):
        ## train the model
        print("TRAINING MODELS")
        data_dir = os.path.join("cs-train")
        model_train(data_dir,pipe,param_grid,test=False)

        ## load the model
        print("LOADING MODELS")
        all_data, all_models = model_load()
        print("... models loaded: ",",".join(all_models.keys()))

        data.append(all_data)
        models.append(all_models)

    ## plot learning curves
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    country='all'
    ls_model = models[0][country]
    ls_data = data[0][country]

    en_model = models[1][country]
    en_data = data[1][country]

    rf_model = models[2][country]
    rf_data = data[2][country]

    svr_model = models[3][country]
    svr_data = data[3][country]

    plot_learning_curve(ls_model, ls_data['X'], ls_data['y'], ax=ax1)
    plot_learning_curve(en_model, en_data['X'], en_data['y'], ax=ax2)
    plot_learning_curve(rf_model, rf_data['X'], rf_data['y'], ax=ax3)
    plot_learning_curve(svr_model, svr_data['X'], svr_data['y'], ax=ax4)

    ax1.set_title("Lasso Regression")
    ax2.set_title("ElasticNet Regression")
    ax3.set_title("Random Forest Tree Regression")
    ax4.set_title("Support Vector Regression")

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_ylim((0.1, 1.2))

    plt.show()

    # Switch for further functionality
    ## train the model
    # print("TRAINING MODELS")
    # data_dir = os.path.join("cs-train")
    # model_train(data_dir,test=False)

    ## load the model
    # print("LOADING MODELS")
    # all_data, all_models = model_load()
    # print("... models loaded: ",",".join(all_models.keys()))

    ## test predict
    # country='all'
    # year='2018'
    # month='07'
    # day='06'
    # result = model_predict(country,year,month,day)
    # print(result)
