from PyQt5.QtWidgets import *
from PyQt5.QtCore import QDir

app = QApplication([])
main = QWidget()
line = QVBoxLayout()
main.setGeometry(600, 500, 600, 500)

btn_browse = QPushButton('Выбрать папку')
btn_browse2 = QPushButton('Выбрать файл')
choose_task = QComboBox()
choose_task.addItems(['Задача регрессии (CSV Файл)', 'Задача классификации (CSV Файл)', 'Задача классификации (Изображения)'])

row1 = QHBoxLayout()
row2 = QHBoxLayout()

row1.addWidget(choose_task)
row2.addWidget(btn_browse)
row2.addWidget(btn_browse2)

btn_browse2.setVisible(True)
btn_browse.setVisible(False)

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
        print(directory_csv)

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

btn_browse.clicked.connect(test_train_data_find)
btn_browse2.clicked.connect(csv_file_finder)
choose_task.currentIndexChanged.connect(change_task)

line.addLayout(row1)
line.addLayout(row2)

main.setLayout(line)
main.show()
app.exec_()
