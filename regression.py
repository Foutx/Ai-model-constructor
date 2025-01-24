from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

def random_forest(**kwargs):
    return RandomForestRegressor(kwargs)

def line_regression(**kwargs):
    return LinearRegression(kwargs)

def gradient_boosting(**kwargs):
    return GradientBoostingRegressor(kwargs)

def params_random_forest(a,b,c,d,e,f):
    return {
        'n_estimators':[a],
        'max_depth':[b],
        'min_samples_leaf':[c],
        'min_samples_split':[d],
        'max_features':[e],
        'max_samples':[f]
        }

def params_line_regression(a,b,c,d):
    return {
        'fit_intercept':[a],
        'normalize':[b],
        'copy_X':[c],
        'n_jobs':[d]
        }

def params_gradient_boosting(a,b,c,d,e,f,g,h,i,j):
    return {
        "n_estimators": [a],
        "learning_rate": [b],
        "max_depth": [c], 
        "min_samples_split": [d], 
        "min_samples_leaf": [e], 
        "max_features": [f], 
        "subsample": [g], 
        "criterion": [h], 
        "random_state": [i], 
        "verbose": [j]
    }