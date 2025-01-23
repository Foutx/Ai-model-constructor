from PyQt5.QtWidgets import *

app = QApplication([])
main = QWidget()
line = QVBoxLayout()

main.setLayout(line)
main.show()
app.exec_()