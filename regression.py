from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,accuracy_score, f1_score, roc_auc_score, confusion_matrix,ConfusionMatrixDisplay, classification_report, explained_variance_score, mean_absolute_percentage_error 

import pickle

import pandas as pd

import matplotlib.pyplot as plt

from tabulate import tabulate

import numpy as np

def random_forest(**kwargs):
    return RandomForestRegressor(**kwargs)

def line_regression(**kwargs):
    return LinearRegression(**kwargs)

def gradient_boosting_regression(**kwargs):
    return GradientBoostingRegressor(**kwargs)

def random_forest_class(**kwargs):
    return RandomForestClassifier(**kwargs)

def gradient_boosting_class(**kwargs):
    return GradientBoostingClassifier(**kwargs)

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

def params_random_forest_class(a,b,c,d,e) -> dict:
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

def saving_model_data_regression(model,b:bool,c:bool,y_test,y_pred):

    if b:
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
    if c:
        try:
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred) 
            explained_variance = explained_variance_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)


            metrics = {
                'MSE': mse,
                'MAE': mae,
                'R2': r2,
                'Explained Variance': explained_variance,
                'MAPE': mape
            }


           
            metrics_table = tabulate(metrics.items(), headers=["Metric", "Value"], tablefmt="grid")

            with open('metrics.txt', 'w') as f:
                f.write("==================== Regression metrics ====================\n\n")
                f.write(metrics_table + "\n")

        except ValueError as e:
            with open('metrics.txt', 'w') as f:
                f.write(f"Error: {e}\n")
        else:
            return 

def saving_model_data_csv_class(model,b:bool,c:bool,y_test,y_pred,y_prob=None):
    if c:
        try:
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)

           
            conf_matrix_table = tabulate(conf_matrix, headers=["Predicted 0", "Predicted 1"], tablefmt="grid")

            with open('metrics.txt', 'w') as f:
                f.write("==================== Classification metrics ====================\n\n")
                f.write('Accuracy: {:.4f}\n'.format(accuracy))
                f.write('F1-score (weighted): {:.4f}\n\n'.format(f1))

                if y_prob is not None:
                    try:
                        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
                        f.write('ROC AUC (one-vs-rest): {:.4f}\n\n'.format(roc_auc))
                    except ValueError as e:
                        f.write(f"Error to calculate ROC AUC: {e}\n\n")


                f.write("Confusion Matrix:\n")
                f.write(conf_matrix_table + "\n\n")

                f.write("Classification Report:\n")
                f.write(class_report + "\n")

        except ValueError as e:
            with open('metrics.txt', 'w') as f:
                f.write(f"Error: {e}\n")
            
    if b:
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)

def show_metrics_regression(y_pred,y_test,mse:bool,mae:bool,r2:bool):

    if mse:
        mse = mean_squared_error(y_pred,y_test)
        plt.figure()
        plt.scatter(y_pred, y_pred)
        plt.xlabel("Values")
        plt.ylabel("Predicted values")
        plt.title(f"MSE: {mse:.4f}")
        plt.show()
    if mae:
        mae = mean_absolute_error(y_pred,y_test)
        plt.figure()
        plt.scatter(y_pred, y_pred)
        plt.xlabel("Values")
        plt.ylabel("Predicted values")
        plt.title(f"MAE: {mae:.4f}")
        plt.show()
    if r2:
        r2 = r2_score(y_pred, y_pred)
        plt.figure()
        plt.scatter(y_pred, y_pred)
        plt.xlabel("Values")
        plt.ylabel("Predicted values")
        plt.title(f"R2: {r2:.4f}")
        plt.show()
    else:   
        return None

def show_metrics_csv_class(y_pred,y_test,accuracy:bool,f1:bool,auc_score:bool,confusion_matrix_:bool):
    if accuracy:
        accuracy = accuracy_score(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        plt.plot(accuracy) 
        plt.xlabel("Result")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy: {accuracy:.4f}")
        plt.grid(True)
        plt.show()
    if f1:
        f1 = f1_score(y_test, y_pred, average='weighted')
        plt.figure(figsize=(6, 4))
        plt.bar(['F1-score'], [f1])
        plt.ylabel("F1-score")
        plt.ylim(0, 1)
        plt.title(f"F1-score (weighted): {f1:.4f}")
        plt.grid(True, axis='y')
        plt.show()
    if auc_score:
        try:
            roc_auc = roc_auc_score(y_test, y_pred)
        except ValueError:
            roc_auc = np.nan
        plt.figure(figsize=(6, 4))
        plt.bar(['ROC AUC'], [roc_auc])
        plt.ylabel("ROC AUC")
        plt.ylim(0, 1)
        plt.title(f"ROC AUC: {roc_auc:.4f}")
        plt.grid(True, axis='y')
        plt.show()
    if confusion_matrix_:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', values_format='d') 
        plt.title("Confusion Matrix")
        plt.show()

