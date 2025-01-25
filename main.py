from PyQt5.QtWidgets import *
from PyQt5.QtCore import QDir

import regression


if __name__ == '__main__':

    app = QApplication([])
    main = QWidget()
    line = QVBoxLayout()
    main.setGeometry(600,500,600,500)

    btn_browse = QPushButton('Выбрать папку')
    btn_browse2 = QPushButton('Выбрать файл')

    file_name_csv = QLabel('')
    lbl_data = QLabel('Данные:')
    lbl_model_param = QLabel('Модель:')
    lbl_metrics = QLabel('Графики на вывод:')

    get_y_name = QLineEdit()
    get_y_name.setPlaceholderText('Название целевой переменной')
    get_y_name.setToolTip('Переменная, которую вы будете предсказывать')

    if_shuffle  = QCheckBox(' Перемешивать список при разделении данных ')
    if_mse = QCheckBox(' MSE (Mean squared error)')
    if_mae = QCheckBox(' MAE (Mean absolute error)')
    if_r2 = QCheckBox(' R2 score (R2 score)')
    percent_data = QLineEdit()
    percent_data.setPlaceholderText('% тестовых данных')
    percent_data.setToolTip('Процент тестовых данных в %')

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

    btn_browse2.setVisible(True)
    btn_browse.setVisible(False)

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
    row7.addWidget(get_reg_forest_min_samples_split)
    row7.addWidget(get_reg_forest_max_features)
    row7.addWidget(get_reg_forest_max_samples)
    row8.addWidget(lbl_metrics)
    row9.addWidget(if_mse)
    row9.addWidget(if_mae)
    row9.addWidget(if_r2)

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
            pass

        elif index == 1:
            pass

        elif index == 3:
            pass

    btn_browse.clicked.connect(test_train_data_find)
    btn_browse2.clicked.connect(csv_file_finder)
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

    main.setLayout(line)
    main.show()
    app.exec_()
