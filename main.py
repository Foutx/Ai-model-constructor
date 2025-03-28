from PyQt5.QtWidgets import *
from PyQt5.QtCore import QDir

import numpy as np

import regression

if __name__ == '__main__':

    app = QApplication([])
    main = QWidget()
    line = QVBoxLayout()
    main.setFixedSize(600,400)

    app.setStyleSheet("""
    QWidget {
        background-color: #f4f4f4;
        font-size: 14px;
    }
    
    QPushButton {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 5px 8px;
        font-size: 14px;
    }
    
    QPushButton:hover {
        background-color: #45a049;
    }

    QLineEdit, QComboBox {
        background-color: white;
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 2px 5px;
        line-height: 20px;
    }

    QLabel {
        font-weight: bold;
        padding-bottom: 5px;  /* Еще выше */
    }

    QCheckBox {
        spacing: 5px;
        padding-bottom: 5px;  /* Еще выше */
    }
""")






    btn_browse = QPushButton('Select Folder')
    btn_browse2 = QPushButton('Select File')
    btn_create_regression_csv = QPushButton('Create Regression Model (CSV)')
    btn_create_classification_csv = QPushButton('Create Classification Model (CSV)')
    btn_create_classification = QPushButton('Create Classification Model')
    
    file_name_csv = QLabel('')
    lbl_data = QLabel('Data:')
    lbl_model_param = QLabel('Model:')
    lbl_metrics = QLabel('Output Graphs:')
    lbl_files = QLabel('Save Data (In main folder):')
    
    get_y_name = QLineEdit()
    get_y_name.setPlaceholderText('Target Variable Name')
    get_y_name.setToolTip('The variable you want to predict')
    
    if_shuffle  = QCheckBox(' Shuffle the dataset when splitting ')
    if_mse = QCheckBox(' MSE (Mean Squared Error)')
    if_mae = QCheckBox(' MAE (Mean Absolute Error)')
    if_r2 = QCheckBox(' R2 Score')
    
    if_f1 = QCheckBox(' F1 Score ')
    if_accuracy = QCheckBox(' Accuracy Score ')
    if_roc_auc_score = QCheckBox(' ROC AUC Score ')
    if_confusion_matrix = QCheckBox(' Confusion Matrix ')
    
    if_model_save_regression = QCheckBox(' Save Model (.pkl) ')
    if_metrics_regression = QCheckBox(' Save Metrics (.txt) ')
    if_model_save_class = QCheckBox(' Save Model (.pkl) ')
    if_metrics_class = QCheckBox(' Save Metrics (.txt) ')
    
    percent_data = QLineEdit()
    percent_data.setPlaceholderText('% of Test Data /100')
    percent_data.setToolTip('Percentage of test data in %/100')
    
    choose_task = QComboBox()
    choose_task.addItems(['Regression Task (CSV File)', 'Classification Task (CSV File)']) # 'Classification Task (Images)' (maybe in future idk <3 )
    choose_model_regression = QComboBox()
    choose_model_regression.addItems(['RandomForestRegressor', 'GradientBoostingRegressor', 'LinearRegression'])
    choose_model_classification_csv = QComboBox()
    choose_model_classification_csv.addItems(['RandomForestClassifier', 'GradientBoostingClassifier'])
    
    get_reg_forest_estimators = QLineEdit()
    get_reg_forest_estimators.setPlaceholderText('n_estimators')
    get_reg_forest_estimators.setToolTip('Number of random trees')
    get_reg_forest_max_depth = QLineEdit()
    get_reg_forest_max_depth.setPlaceholderText('max_depth')
    get_reg_forest_max_depth.setToolTip('Maximum tree depth')
    get_reg_forest_min_samples_leaf = QLineEdit()
    get_reg_forest_min_samples_leaf.setPlaceholderText('min_samples_leaf')
    get_reg_forest_min_samples_leaf.setToolTip('Minimum number of samples per leaf')
    get_reg_forest_min_samples_split = QLineEdit()
    get_reg_forest_min_samples_split.setPlaceholderText('min_samples_split')
    get_reg_forest_min_samples_split.setToolTip('Minimum number of features for a split')
    get_reg_forest_max_features = QLineEdit()
    get_reg_forest_max_features.setPlaceholderText('max_features')
    get_reg_forest_max_features.setToolTip('Maximum number of features')
    get_reg_forest_max_samples = QLineEdit()
    get_reg_forest_max_samples.setPlaceholderText('max_samples')
    get_reg_forest_max_samples.setToolTip('Maximum number of samples')
    
    get_reg_boosting_learning_rate = QLineEdit()
    get_reg_boosting_learning_rate.setPlaceholderText('learning_rate')
    get_reg_boosting_learning_rate.setToolTip('Range: [0.001; 0.1]')
    get_reg_boosting_subsample = QLineEdit()
    get_reg_boosting_subsample.setPlaceholderText('subsample')
    get_reg_boosting_subsample.setToolTip('Range: [0;1]')
    
    get_reg_linear_fit_intercept = QCheckBox('fit_intercept')
    get_reg_linear_copy_X = QCheckBox('copy_X')
    get_reg_linear_n_jobs = QLineEdit()
    get_reg_linear_n_jobs.setPlaceholderText('n_jobs')
    get_reg_linear_n_jobs.setToolTip('Number of CPU cores')


    btn_browse2.setVisible(True)
    btn_browse.setVisible(False)
    btn_create_classification_csv.setVisible(False)
    btn_create_classification.setVisible(False)

    get_reg_boosting_learning_rate.setVisible(False)
    get_reg_boosting_subsample.setVisible(False)
    get_reg_linear_fit_intercept.setVisible(False)
    get_reg_linear_copy_X.setVisible(False)
    get_reg_linear_n_jobs.setVisible(False)

    if_accuracy.setVisible(False)
    if_f1.setVisible(False)
    if_roc_auc_score.setVisible(False)
    if_confusion_matrix.setVisible(False)

    if_model_save_class.setVisible(False)
    if_metrics_class.setVisible(False)

    choose_model_classification_csv.setVisible(False)

    row1 = QHBoxLayout()
    row2 = QHBoxLayout()
    row3 = QHBoxLayout()
    row23 = QHBoxLayout()
    row223 = QHBoxLayout()
    row4 = QHBoxLayout()
    row5 = QHBoxLayout()
    row6 = QHBoxLayout()
    row7 = QHBoxLayout()
    row8 = QHBoxLayout()
    row9 = QHBoxLayout()
    row10 = QHBoxLayout()
    row11 = QHBoxLayout()
    row12 = QHBoxLayout()

    row1.addWidget(choose_task)
    row2.addWidget(btn_browse)
    row2.addWidget(btn_browse2)
    row223.addWidget(file_name_csv)
    row23.addWidget(lbl_data)
    row3.addWidget(get_y_name)
    row3.addWidget(percent_data)
    row3.addWidget(if_shuffle)
    row4.addWidget(lbl_model_param)
    row5.addWidget(choose_model_regression)
    row5.addWidget(choose_model_classification_csv)
    row6.addWidget(get_reg_forest_estimators)
    row6.addWidget(get_reg_forest_max_depth)
    row6.addWidget(get_reg_forest_min_samples_leaf)
    row6.addWidget(get_reg_linear_fit_intercept)
    row6.addWidget(get_reg_linear_copy_X)
    row6.addWidget(get_reg_linear_n_jobs)
    row7.addWidget(get_reg_forest_min_samples_split)
    row7.addWidget(get_reg_forest_max_features)
    row7.addWidget(get_reg_forest_max_samples)
    row7.addWidget(get_reg_boosting_learning_rate)
    row7.addWidget(get_reg_boosting_subsample)
    row8.addWidget(lbl_metrics)
    row9.addWidget(if_mse)
    row9.addWidget(if_mae)
    row9.addWidget(if_r2)
    row9.addWidget(if_accuracy)
    row9.addWidget(if_f1)
    row9.addWidget(if_roc_auc_score)
    row9.addWidget(if_confusion_matrix)
    row10.addWidget(lbl_files)
    row11.addWidget(if_model_save_regression)
    row11.addWidget(if_metrics_regression)
    row11.addWidget(if_model_save_class)
    row11.addWidget(if_metrics_class)
    row12.addWidget(btn_create_regression_csv)
    row12.addWidget(btn_create_classification_csv)
    row12.addWidget(btn_create_classification)

    def show_message(text):
        # Show Error Message Box
        msg = QMessageBox()
        msg.setText(text)
        msg.exec()

  
    def test_train_data_find(): # now its usless
        # Test will be same
        global directory_train
        directory_train = QFileDialog.getExistingDirectory(main, "Выбрать папку", QDir.homePath())
        if directory_train:
            print(directory_train)

    def csv_file_finder():

        global directory_csv
        directory_csv, _ = QFileDialog.getOpenFileName(main, "Выбрать файл", QDir.homePath(), "Все файлы (*.*)")
        if directory_csv:
            file_name_csv.setText(f'CSV файл - {directory_csv}')

    # Change task of Ai
    def change_task(index):

        if index == 0:
            btn_create_regression_csv.setVisible(True)
            btn_create_classification_csv.setVisible(False)
            btn_create_classification.setVisible(False)
            btn_browse2.setVisible(True)
            btn_browse.setVisible(False)

            choose_model_regression.setVisible(True)
            choose_model_classification_csv.setVisible(False)

            # Params
            get_reg_boosting_learning_rate.setVisible(False)
            get_reg_boosting_subsample.setVisible(False)

            get_reg_linear_fit_intercept.setVisible(False)
            get_reg_linear_copy_X.setVisible(False)
            get_reg_linear_n_jobs.setVisible(False)

            get_reg_forest_max_samples.setVisible(True)
            get_reg_forest_estimators.setVisible(True)
            get_reg_forest_max_depth.setVisible(True)
            get_reg_forest_min_samples_leaf.setVisible(True)
            get_reg_forest_min_samples_split.setVisible(True)
            get_reg_forest_max_features.setVisible(True)

            # Metrics
            if_mse.setVisible(True)
            if_mae.setVisible(True)
            if_r2.setVisible(True)

            if_accuracy.setVisible(False)
            if_f1.setVisible(False)
            if_roc_auc_score.setVisible(False)
            if_confusion_matrix.setVisible(False)
            if_accuracy.setChecked(False)
            if_f1.setChecked(False)
            if_roc_auc_score.setChecked(False)
            if_confusion_matrix.setChecked(False)

            if_model_save_regression.setVisible(True)
            if_metrics_regression.setVisible(True)
            if_model_save_class.setVisible(False)
            if_metrics_class.setVisible(False)

            choose_model_regression.setCurrentIndex(0)

        elif index == 1:
            btn_create_regression_csv.setVisible(False)
            btn_create_classification_csv.setVisible(True)
            btn_create_classification.setVisible(False)
            btn_browse2.setVisible(True)
            btn_browse.setVisible(False)

            choose_model_regression.setVisible(False)
            choose_model_classification_csv.setVisible(True)

            # Params
            get_reg_boosting_learning_rate.setVisible(False)
            get_reg_boosting_subsample.setVisible(False)

            get_reg_linear_fit_intercept.setVisible(False)
            get_reg_linear_copy_X.setVisible(False)
            get_reg_linear_n_jobs.setVisible(False)

            get_reg_forest_max_samples.setVisible(True)
            get_reg_forest_estimators.setVisible(True)
            get_reg_forest_max_depth.setVisible(True)
            get_reg_forest_min_samples_leaf.setVisible(True)
            get_reg_forest_min_samples_split.setVisible(True)
            get_reg_forest_max_features.setVisible(False)

            # Metrics
            if_mse.setVisible(False)
            if_mae.setVisible(False)
            if_r2.setVisible(False)
            if_mse.setChecked(False)
            if_mae.setChecked(False)
            if_r2.setChecked(False)

            if_accuracy.setVisible(True)
            if_f1.setVisible(True)
            if_roc_auc_score.setVisible(True)
            if_confusion_matrix.setVisible(True)

            if_model_save_regression.setVisible(False)
            if_metrics_regression.setVisible(False)
            if_model_save_class.setVisible(True)
            if_metrics_class.setVisible(True)

            choose_model_classification_csv.setCurrentIndex(0)
        
        elif index == 2: # now its usless
            btn_create_regression_csv.setVisible(False)
            btn_create_classification_csv.setVisible(False)
            btn_create_classification.setVisible(True)
            btn_browse2.setVisible(False)
            btn_browse.setVisible(True)

    # Change regression model
    def change_model_regression(index):

        if index == 0:
            get_reg_boosting_learning_rate.setVisible(False)
            get_reg_boosting_subsample.setVisible(False)

            get_reg_linear_fit_intercept.setVisible(False)
            get_reg_linear_copy_X.setVisible(False)
            get_reg_linear_n_jobs.setVisible(False)

            get_reg_forest_max_samples.setVisible(True)
            get_reg_forest_estimators.setVisible(True)
            get_reg_forest_max_depth.setVisible(True)
            get_reg_forest_min_samples_leaf.setVisible(True)
            get_reg_forest_min_samples_split.setVisible(True)
            get_reg_forest_max_features.setVisible(True)

        elif index == 1:
            get_reg_forest_max_samples.setVisible(False)
            get_reg_boosting_learning_rate.setVisible(True)
            get_reg_boosting_subsample.setVisible(True)

            get_reg_linear_fit_intercept.setVisible(False)
            get_reg_linear_copy_X.setVisible(False)
            get_reg_linear_n_jobs.setVisible(False)

            get_reg_forest_estimators.setVisible(True)
            get_reg_forest_max_depth.setVisible(True)
            get_reg_forest_min_samples_leaf.setVisible(True)
            get_reg_forest_min_samples_split.setVisible(True)
            get_reg_forest_max_features.setVisible(True)

        elif index == 2:
            get_reg_linear_fit_intercept.setVisible(True)
            get_reg_linear_copy_X.setVisible(True)
            get_reg_linear_n_jobs.setVisible(True)

            get_reg_boosting_learning_rate.setVisible(False)
            get_reg_boosting_subsample.setVisible(False)

            get_reg_forest_max_samples.setVisible(False)
            get_reg_forest_estimators.setVisible(False)
            get_reg_forest_max_depth.setVisible(False)
            get_reg_forest_min_samples_leaf.setVisible(False)
            get_reg_forest_min_samples_split.setVisible(False)
            get_reg_forest_max_features.setVisible(False)

    def change_model_classififcation_csv(index):
        
        if index == 0:
            get_reg_forest_max_samples.setVisible(True)

            get_reg_forest_max_features.setVisible(False)
            get_reg_boosting_learning_rate.setVisible(False)
            get_reg_boosting_subsample.setVisible(False)

        if index == 1:
            get_reg_forest_max_samples.setVisible(False)

            get_reg_forest_max_features.setVisible(True)
            get_reg_boosting_learning_rate.setVisible(True)
            get_reg_boosting_subsample.setVisible(True)

    def get_int(string, default=None):

        if string == '':
            print(default)
            return default

        try:
            value = float(string)
            if value.is_integer():
                print(int(value))
                return int(value)
            else:
                print(value)
                return value

        except ValueError:
            print(f"'{string}' не является допустимым числом.")
            return np.nan

    # Create regression models
    def create_model_regression():

        try:
            X_train,X_test,y_train,y_test = regression.data_test_train_split(directory_csv,
                                                                             get_y_name.text(),
                                                                             float(percent_data.text()),
                                                                             if_shuffle.isChecked())
            
        except:
            show_message('Error to create data for model, check your values in (Data)')
            return

        if choose_model_regression.currentText() == 'RandomForestRegressor':

            n_estimators_ = get_reg_forest_estimators.text()
            n_estimators_ = get_int(n_estimators_,50)
            if n_estimators_ is np.nan:
                show_message('Error in estimators')
                return 
            
            n_max_depth = get_reg_forest_max_depth.text()
            n_max_depth = get_int(n_max_depth,None)
            if n_max_depth is np.nan:
                show_message('Error in max_depth')
                return 

            n_min_samples_leaf = get_reg_forest_min_samples_leaf.text()
            n_min_samples_leaf = get_int(n_min_samples_leaf,1)
            if n_min_samples_leaf is np.nan:
                show_message('Error in min_samples_leaf')
                return 

            n_min_samples_split = get_reg_forest_min_samples_split.text()
            n_min_samples_split = get_int(n_min_samples_split,2)
            if n_min_samples_split is np.nan:
                show_message('Error in min_samples_split')
                return 

            n_max_features = get_reg_forest_max_features.text()
            n_max_features = get_int(n_max_features,2)
            if n_max_features is np.nan:
                show_message('Error in max_features')
                return 

            n_max_samples = get_reg_forest_max_samples.text()
            n_max_samples = get_int(n_max_samples,None)
            if n_max_samples is np.nan:
                show_message('Error in max_samples')
                return 
            
            global_params = regression.params_random_forest(n_estimators_,
                                            n_max_depth,
                                            n_min_samples_leaf,
                                            n_min_samples_split,
                                            n_max_features,
                                            n_max_samples)
            
            try:
                model = regression.random_forest(**global_params)
                model.fit(X_train,y_train)
                print('Лес')

            except:
                show_message('Model compile error')
                return

        elif choose_model_regression.currentText() == 'GradientBoostingRegressor':
            
            n_estimators_ = get_reg_forest_estimators.text()
            n_estimators_ = get_int(n_estimators_,50)
            if n_estimators_ is np.nan:
                show_message('Error in estimators')
                return 
            
            n_max_depth = get_reg_forest_max_depth.text()
            n_max_depth = get_int(n_max_depth,None)
            if n_max_depth is np.nan:
                show_message('Error in max_depth')
                return 

            n_min_samples_leaf = get_reg_forest_min_samples_leaf.text()
            n_min_samples_leaf = get_int(n_min_samples_leaf,1)
            if n_min_samples_leaf is np.nan:
                show_message('Error in min_samples_leaf')
                return 

            n_min_samples_split = get_reg_forest_min_samples_split.text()
            n_min_samples_split = get_int(n_min_samples_split,2)
            if n_min_samples_split is np.nan:
                show_message('Error in min_samples_split')
                return 

            n_max_features = get_reg_forest_max_features.text()
            n_max_features = get_int(n_max_features,2)
            if n_max_features is np.nan:
                show_message('Error in max_features')
                return 

            n_learning_rate = get_reg_boosting_learning_rate.text()
            n_learning_rate = get_int(n_learning_rate,0.1)
            if n_learning_rate is np.nan:
                show_message('Error in learning_rate')
                return
            
            n_subsample = get_reg_boosting_subsample.text()
            n_subsample = get_int(n_subsample,1)
            if n_subsample is np.nan:
                show_message('Error in subsample')
                return

            global_params = regression.params_gradient_boosting(n_estimators_,
                                                                n_max_depth,
                                                                n_min_samples_leaf,
                                                                n_min_samples_split,
                                                                n_max_features,
                                                                n_learning_rate,
                                                                n_subsample)
            
            try:
                model = regression.gradient_boosting_regression(**global_params)
                model.fit(X_train,y_train)
                print('Градиент')

            except:
                show_message('Model compile error')
                return

        else:
            n_jobs_ = get_reg_linear_n_jobs.text()
            n_jobs_ = get_int(n_jobs_,None)
            if n_jobs_ is np.nan:
                return 
            
            global_params = regression.params_line_regression(not(get_reg_linear_fit_intercept.isChecked()),
                                              not(get_reg_linear_copy_X.isChecked()),
                                              n_jobs_)
            try:

                model = regression.line_regression(**global_params)
                model.fit(X_train,y_train)
                print('Линейная')

            except:
                show_message('Model compile error')
                return

        y_pred = model.predict(X_test)
        
        regression.show_metrics_regression(y_pred,y_test,if_mse.isChecked(),if_mae.isChecked(),if_r2.isChecked())

        regression.saving_model_data_regression(model,if_model_save_regression.isChecked(),if_metrics_regression.isChecked(),y_test,y_pred)
    
    # Create CSV classification models
    def create_model_classification_csv():
        try:
            X_train,X_test,y_train,y_test = regression.data_test_train_split(directory_csv,
                                                                             get_y_name.text(),
                                                                             float(percent_data.text()),
                                                                             if_shuffle.isChecked())
            
        except:
            show_message('Error to create data for model, check your values in (Data)')
            return
        
        if choose_model_classification_csv.currentText() == "RandomForestClassifier":

            n_estimators_ = get_reg_forest_estimators.text()
            n_estimators_ = get_int(n_estimators_,50)
            if n_estimators_ is np.nan:
                show_message('Error in estimators')
                return 
            
            n_max_depth = get_reg_forest_max_depth.text()
            n_max_depth = get_int(n_max_depth,None)
            if n_max_depth is np.nan:
                show_message('Error in max_depth')
                return 

            n_min_samples_leaf = get_reg_forest_min_samples_leaf.text()
            n_min_samples_leaf = get_int(n_min_samples_leaf,1)
            if n_min_samples_leaf is np.nan:
                show_message('Error in min_samples_leaf')
                return 

            n_min_samples_split = get_reg_forest_min_samples_split.text()
            n_min_samples_split = get_int(n_min_samples_split,2)
            if n_min_samples_split is np.nan:
                show_message('Error in min_samples_split')
                return 

            n_max_samples = get_reg_forest_max_samples.text()
            n_max_samples = get_int(n_max_samples,None)
            if n_max_samples is np.nan:
                show_message('Error in max_samples')
                return 
            
            global_params = regression.params_random_forest_class(n_estimators_,
                                                                  n_max_depth,
                                                                  n_min_samples_leaf,
                                                                  n_min_samples_split,
                                                                  n_max_samples)
            try:
                model = regression.random_forest_class(**global_params)
                model.fit(X_train,y_train)
                print('Лесс классификатор')

            except:
                show_message('Model compile error')
                return

            y_pred = model.predict(X_test)

            regression.show_metrics_csv_class(y_pred,y_test,if_accuracy.isChecked(),if_f1.isChecked(),if_roc_auc_score.isChecked(),if_confusion_matrix.isChecked())

            regression.saving_model_data_csv_class(model,if_model_save_class.isChecked(),if_metrics_class.isChecked(),y_test,y_pred)
        
        if choose_model_classification_csv.currentText() == "GradientBoostingClassifier":

            n_estimators_ = get_reg_forest_estimators.text()
            n_estimators_ = get_int(n_estimators_,50)
            if n_estimators_ is np.nan:
                show_message('Error in estimators')
                return 
            
            n_max_depth = get_reg_forest_max_depth.text()
            n_max_depth = get_int(n_max_depth,None)
            if n_max_depth is np.nan:
                show_message('Error in max_depth')
                return 

            n_min_samples_leaf = get_reg_forest_min_samples_leaf.text()
            n_min_samples_leaf = get_int(n_min_samples_leaf,1)
            if n_min_samples_leaf is np.nan:
                show_message('Error in min_samples_leaf')
                return 

            n_min_samples_split = get_reg_forest_min_samples_split.text()
            n_min_samples_split = get_int(n_min_samples_split,2)
            if n_min_samples_split is np.nan:
                show_message('Error in min_samples_split')
                return 

            n_max_features = get_reg_forest_max_features.text()
            n_max_features = get_int(n_max_features,2)
            if n_max_features is np.nan:
                show_message('Error in max_features')
                return 

            n_learning_rate = get_reg_boosting_learning_rate.text()
            n_learning_rate = get_int(n_learning_rate,0.1)
            if n_learning_rate is np.nan:
                show_message('Error in learning_rate')
                return
            
            n_subsample = get_reg_boosting_subsample.text()
            n_subsample = get_int(n_subsample,1)
            if n_subsample is np.nan:
                show_message('Error in subsample')
                return

            global_params = regression.params_gradient_boosting(n_estimators_,
                                                                n_max_depth,
                                                                n_min_samples_leaf,
                                                                n_min_samples_split,
                                                                n_max_features,
                                                                n_learning_rate,
                                                                n_subsample)
            
            try:
                model = regression.gradient_boosting_class(**global_params)
                model.fit(X_train,y_train)
                print('Градиент классификатор')

            except:
                show_message('Model compile error')
                return
            y_pred = model.predict(X_test)

            regression.show_metrics_csv_class(y_pred,y_test,if_accuracy.isChecked(),if_f1.isChecked(),if_roc_auc_score.isChecked(),if_confusion_matrix.isChecked())

            regression.saving_model_data_csv_class(model,if_model_save_class.isChecked(),if_metrics_class.isChecked(),y_test,y_pred)
        
    btn_browse.clicked.connect(test_train_data_find)
    btn_browse2.clicked.connect(csv_file_finder)
    btn_create_regression_csv.clicked.connect(create_model_regression)
    btn_create_classification_csv.clicked.connect(create_model_classification_csv)

    choose_task.currentIndexChanged.connect(change_task)
    choose_model_regression.currentIndexChanged.connect(change_model_regression)
    choose_model_classification_csv.currentIndexChanged.connect(change_model_classififcation_csv)

    line.addLayout(row1)
    line.addLayout(row2)
    line.addLayout(row223)
    line.addLayout(row23)
    line.addLayout(row3)
    line.addLayout(row4)
    line.addLayout(row5)
    line.addLayout(row6)
    line.addLayout(row7)
    line.addLayout(row8)
    line.addLayout(row9)
    line.addLayout(row10)
    line.addLayout(row11)
    line.addLayout(row12)

    main.setLayout(line)
    main.show()
    app.exec_()
