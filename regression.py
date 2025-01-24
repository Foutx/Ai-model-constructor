from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd

def random_forest(**kwargs):
    return RandomForestRegressor(kwargs)

def line_regression(**kwargs):
    return LinearRegression(kwargs)

def gradient_boosting(**kwargs):
    return GradientBoostingRegressor(kwargs)

def params_random_forest(a,b,c,d,e,f):
    return {
        'n_estimators':a,
        'max_depth':b,
        'min_samples_leaf':c,
        'min_samples_split':d,
        'max_features':e,
        'max_samples':f
        }

def params_line_regression(a,b,c,d):
    return {
        'fit_intercept':a,
        'normalize':b,
        'copy_X':c,
        'n_jobs':d
        }

def params_gradient_boosting(a,b,c,d,e,f,g,h,i,j):
    return {
        "n_estimators": a,
        "learning_rate": b,
        "max_depth": c, 
        "min_samples_split": d, 
        "min_samples_leaf": e, 
        "max_features": f, 
        "subsample": g, 
        "criterion": h, 
        "random_state": i, 
        "verbose": j
    }

def data_test_train_split(file_path,y_reg,percent_split,shuffle_data):
    df = pd.read_csv(file_path)
    all_colums = [x for x in df.columns]
    all_colums.remove(y_reg)
    X = df[[x for x in all_colums]]
    y = df[y_reg]

    X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=percent_split,shuffle=shuffle_data)

    return X_train, X_test, y_train, y_test

# нужно вывести ещё показатели
def regression_model(model,X_train,X_test,y_train,y_test):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_pred,y_test)
    return mse