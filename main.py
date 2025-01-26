from PyQt5.QtWidgets import *
from PyQt5.QtCore import QDir

import numpy as np

import regression

if __name__ == '__main__':

    app = QApplication([])
    main = QWidget()
    line = QVBoxLayout()
    main.setGeometry(600,500,600,500)

    btn_browse = QPushButton('Выбрать папку')
    btn_browse2 = QPushButton('Выбрать файл')
    btn_create_reg_forest = QPushButton('Создать модель')

    file_name_csv = QLabel('')
    lbl_data = QLabel('Data:')
    lbl_model_param = QLabel('Model:')
    lbl_metrics = QLabel('Графики на вывод:')
    lbl_files = QLabel('Сохранение даынных (В папке: Загрузки):')

    get_y_name = QLineEdit()
    get_y_name.setPlaceholderText('Название целевой переменной')
    get_y_name.setToolTip('Переменная, которую вы будете предсказывать')

    if_shuffle  = QCheckBox(' Перемешивать список при разделении данных ')
    if_mse = QCheckBox(' MSE (Mean squared error)')
    if_mae = QCheckBox(' MAE (Mean absolute error)')
    if_r2 = QCheckBox(' R2 score (R2 score)')

    if_model_save = QCheckBox(' Сохранить модель (.pkl) ')
    if_metrics = QCheckBox(' Значения метрик (.txt) ')

    percent_data = QLineEdit()
    percent_data.setPlaceholderText('% тестовых данных /100')
    percent_data.setToolTip('Процент тестовых данных в %/100')

    choose_task = QComboBox()
    choose_task.addItems(['Задача регрессии (CSV Файл)', 'Задача классификации (CSV Файл)', 'Задача классификации (Изображения)'])
    choose_model_regression = QComboBox()
    choose_model_regression.addItems(['RandomForestRegressor','GradientBoostingRegressor','LinearRegression'])

    get_reg_forest_estimators = QLineEdit()
    get_reg_forest_estimators.setPlaceholderText('n_estimators')
    get_reg_forest_estimators.setToolTip('Количество случайных деревьев')
    get_reg_forest_max_depth = QLineEdit()
    get_reg_forest_max_depth.setPlaceholderText('max_depth')
    get_reg_forest_max_depth.setToolTip('Максимальная глубина дерева')
    get_reg_forest_min_samples_leaf = QLineEdit()
    get_reg_forest_min_samples_leaf.setPlaceholderText('min_samples_leaf')
    get_reg_forest_min_samples_leaf.setToolTip('Минимальное кол-во объектов в листе дерева')
    get_reg_forest_min_samples_split = QLineEdit()
    get_reg_forest_min_samples_split.setPlaceholderText('min_samples_split')
    get_reg_forest_min_samples_split.setToolTip('Минимальное кол-во признаков для разбиения')
    get_reg_forest_max_features = QLineEdit()
    get_reg_forest_max_features.setPlaceholderText('max_features')
    get_reg_forest_max_features.setToolTip('Максимальное количество признаков')
    get_reg_forest_max_samples = QLineEdit()
    get_reg_forest_max_samples.setPlaceholderText('max_samples')
    get_reg_forest_max_samples.setToolTip('Максимальное количество образцов')

    get_reg_boosting_learning_rate = QLineEdit()
    get_reg_boosting_learning_rate.setPlaceholderText('learning_rate')
    get_reg_boosting_learning_rate.setToolTip('Диаопозон: [0,001;0,1]')
    get_reg_boosting_subsample = QLineEdit()
    get_reg_boosting_subsample.setPlaceholderText('subsample')
    get_reg_boosting_subsample.setToolTip('Диапазон: [0;1]')

    get_reg_linear_fit_intercept = QCheckBox('fit_intercept')
    get_reg_linear_copy_X = QCheckBox('copy_X')
    get_reg_linear_n_jobs = QLineEdit()
    get_reg_linear_n_jobs.setPlaceholderText('n_jobs')
    get_reg_linear_n_jobs.setToolTip('Кол-во ядер процессора')

    btn_browse2.setVisible(True)
    btn_browse.setVisible(False)
    get_reg_boosting_learning_rate.setVisible(False)
    get_reg_boosting_subsample.setVisible(False)
    get_reg_linear_fit_intercept.setVisible(False)
    get_reg_linear_copy_X.setVisible(False)
    get_reg_linear_n_jobs.setVisible(False)

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
    row10.addWidget(lbl_files)
    row11.addWidget(if_model_save)
    row11.addWidget(if_metrics)
    row12.addWidget(btn_create_reg_forest)

    def show_message(text):
        # Show Error Message Box
        msg = QMessageBox()
        msg.setText(text)
        msg.exec()


    def test_train_data_find():
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

    def change_task(index):

        if index == 0:
            btn_browse2.setVisible(True)
            btn_browse.setVisible(False)

        elif index == 0:
            btn_browse2.setVisible(True)
            btn_browse.setVisible(False)

        elif index == 2:
            btn_browse2.setVisible(False)
            btn_browse.setVisible(True)

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
                return 
            
            n_max_depth = get_reg_forest_max_depth.text()
            n_max_depth = get_int(n_max_depth,None)
            if n_max_depth is np.nan:
                return 

            n_min_samples_leaf = get_reg_forest_min_samples_leaf.text()
            n_min_samples_leaf = get_int(n_min_samples_leaf,1)
            if n_min_samples_leaf is np.nan:
                return 

            n_min_samples_split = get_reg_forest_min_samples_split.text()
            n_min_samples_split = get_int(n_min_samples_split,2)
            if n_min_samples_split is np.nan:
                return 

            n_max_features = get_reg_forest_max_features.text()
            n_max_features = get_int(n_max_features,2)
            if n_max_features is np.nan:
                return 

            n_max_samples = get_reg_forest_max_samples.text()
            n_max_samples = get_int(n_max_samples,None)
            if n_max_samples is np.nan:
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
                print('Ошибка компиляции модели')
                return

        elif choose_model_regression.currentText() == 'GradientBoostingRegressor':
            
            n_estimators_ = get_reg_forest_estimators.text()
            n_estimators_ = get_int(n_estimators_,50)
            if n_estimators_ is np.nan:
                return 
            
            n_max_depth = get_reg_forest_max_depth.text()
            n_max_depth = get_int(n_max_depth,None)
            if n_max_depth is np.nan:
                return 

            n_min_samples_leaf = get_reg_forest_min_samples_leaf.text()
            n_min_samples_leaf = get_int(n_min_samples_leaf,1)
            if n_min_samples_leaf is np.nan:
                return 

            n_min_samples_split = get_reg_forest_min_samples_split.text()
            n_min_samples_split = get_int(n_min_samples_split,2)
            if n_min_samples_split is np.nan:
                return 

            n_max_features = get_reg_forest_max_features.text()
            n_max_features = get_int(n_max_features,2)
            if n_max_features is np.nan:
                return 

            n_learning_rate = get_reg_boosting_learning_rate.text()
            n_learning_rate = get_int(n_learning_rate,0.1)
            if n_learning_rate is np.nan:
                return
            
            n_subsample = get_reg_boosting_subsample.text()
            n_subsample = get_int(n_subsample,1)
            if n_subsample is np.nan:
                return

            global_params = regression.params_gradient_boosting(n_estimators_,
                                                                n_max_depth,
                                                                n_min_samples_leaf,
                                                                n_min_samples_split,
                                                                n_max_features,
                                                                n_learning_rate,
                                                                n_subsample)
            
            try:
                model = regression.gradient_boosting(**global_params)
                model.fit(X_train,y_train)
                print('Градиент')

            except:
                print('Ошибка компиляции модели')
                return

        else:
            n_jobs_ = get_reg_linear_n_jobs.text()
            n_jobs_ = get_int(n_jobs_,None)
            if n_jobs_ is np.nan:
                return 
            
            global_params = regression.params_line_regression(not(get_reg_linear_fit_intercept.isChecked()),
                                              not(get_reg_linear_copy_X.isChecked()),
                                              n_jobs_)
            
            model = regression.line_regression(**global_params)
            model.fit(X_train,y_train)
            print('Линейная')

        y_pred = model.predict(X_test)
        
        regression.show_metrics(y_pred,y_test,if_mse.isChecked(),if_mae.isChecked(),if_r2.isChecked())

        regression.saving_model_data(model,if_model_save.isChecked(),if_metrics.isChecked(),y_test,y_pred)
        
    btn_browse.clicked.connect(test_train_data_find)
    btn_browse2.clicked.connect(csv_file_finder)
    btn_create_reg_forest.clicked.connect(create_model_regression)

    choose_task.currentIndexChanged.connect(change_task)
    choose_model_regression.currentIndexChanged.connect(change_model_regression)

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