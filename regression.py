from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

import pickle

import pandas as pd

import matplotlib.pyplot as plt


def random_forest(**kwargs):
    return RandomForestRegressor(**kwargs)

def line_regression(**kwargs):
    return LinearRegression(**kwargs)

def gradient_boosting(**kwargs):
    return GradientBoostingRegressor(**kwargs)

def params_random_forest(a,b,c,d,e,f) -> dict:
    return {
        'n_estimators':a,
        'max_depth':b,
        'min_samples_leaf':c,
        'min_samples_split':d,
        'max_features':e,
        'max_samples':f
        }

def params_line_regression(a,b,c) -> dict:
    return {
        'fit_intercept':a,
        'copy_X':b,
        'n_jobs':c
        }

def params_gradient_boosting(a,b,c,d,e,f,g) -> dict:
    return {
        "n_estimators": a,
        "max_depth": b, 
        "min_samples_leaf": c, 
        "min_samples_split": d, 
        "max_features": e,
        "learning_rate": f, 
        "subsample": g, 
    }

def params_random_forest_class(a,b,c,d,e):
    return {
    'n_estimators': a,
    'max_depth': b,
    'min_samples_leaf': c,
    'min_samples_split': d,
    'max_samples': e
}

def data_test_train_split(file_path:str,y_reg:str,percent_split:float,shuffle_data:bool):
    try:
        df = pd.read_csv(file_path)
        all_colums = [x for x in df.columns]
        all_colums.remove(y_reg)
        X = df[[x for x in all_colums]]
        y = df[y_reg]

        X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=percent_split,shuffle=shuffle_data)

        return X_train, X_test, y_train, y_test
    except:
        return None

def saving_model_data(model,b:bool,c:bool,y_test,y_pred):

    if b:
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
    if c:
        with open('metrics.txt', 'w') as f:
            f.write('MSE: {}\n'.format(mean_squared_error(y_pred,y_test)))
            f.write('MAE: {}\n'.format(mean_absolute_error(y_pred,y_test)))
            f.write('R2: {}\n'.format(r2_score(y_pred, y_pred)))
    else:
        return 

def show_metrics_regression(y_pred,y_test,mse:bool,mae:bool,r2:bool):

    if mse:
        mse = mean_squared_error(y_pred,y_test)
        plt.figure()
        plt.scatter(y_pred, y_pred)
        plt.xlabel("Фактические значения")
        plt.ylabel("Предсказанные значения")
        plt.title(f"MSE: {mse:.4f}")
        plt.show()
    if mae:
        mae = mean_absolute_error(y_pred,y_test)
        plt.figure()
        plt.scatter(y_pred, y_pred)
        plt.xlabel("Фактические значения")
        plt.ylabel("Предсказанные значения")
        plt.title(f"MAE: {mae:.4f}")
        plt.show()
    if r2:
        r2 = r2_score(y_pred, y_pred)
        plt.figure()
        plt.scatter(y_pred, y_pred)
        plt.xlabel("Фактические значения")
        plt.ylabel("Предсказанные значения")
        plt.title(f"R2: {r2:.4f}")
        plt.show()
    else:   
        return None
