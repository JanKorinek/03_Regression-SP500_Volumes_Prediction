# EDA - S&P500 Regression - *by Jan Kořínek*
Hello and welcome. 

In this repository you can find exploratory data analysis with trained and mutually compared models for a regression task on the S&P500 volumes.

Complete EDA with model training can be run by`EDA-S&P500_Regression.ipynb`. Project contains library of the *Python* scripts stored in *lib* folder which are responsible for the data wrangling, model training and evaluation and for plotting of various characteristics.

## Project Structure
    .
    ├── data                        # Training dataset
    ├── export                      # Sample plots
    ├── lib                         # Code library
        ├── add_features.py         # Features and lagged data creation
        ├── learning_curve.py       # Prepare and plot learning curves for all models
        ├── misc_functions.py       # Support functions
        ├── models_training.py      # All models training&evaluation pipelines
        ├── prepare_dataset.py      # Dataset preparation functions
    ├── models                      # Trained models storage
    LICENSE
    README.md
    requirements.txt
    EDA-S&P500_Regression.ipynb     # Main analysis and models training file

## Usage
* Clone the repository below:

`$ git clone https://gitfront.io/r/korinek-j/de389ca43611982d7c1ae505daac1f158aa72e5f/03-Regression-SP500-Volumes-Prediction.git`

`$ cd 03_Regression-SP500_Volumes_Prediction`

* Setup virtual environment in Anaconda, Pycharm or in IDE you're currently using.

* Install libraries in `requirements.txt`

* Run `$ jupyter-lab EDA-S&P500_Regression.ipynb`