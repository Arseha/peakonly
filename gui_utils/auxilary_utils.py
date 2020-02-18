import os
from PyQt5 import QtWidgets


class FileListWidget(QtWidgets.QListWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file2path = {}

    def addFile(self, path: str):
        filename = os.path.basename(path)
        self.file2path[filename] = path
        self.addItem(filename)

    def deleteFile(self, item: QtWidgets.QListWidgetItem):
        del self.file2path[item.text()]
        self.takeItem(self.row(item))


class GetFolderWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        button = QtWidgets.QToolButton()
        button.setText('...')
        button.clicked.connect(self.set_folder)

        self.lineEdit = QtWidgets.QToolButton()
        self.lineEdit.setText(os.getcwd())

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.lineEdit, 85)
        layout.addWidget(button, 15)

        self.setLayout(layout)

    def set_folder(self):
        directory = str(QtWidgets.QFileDialog.getExistingDirectory())
        if directory:
            self.lineEdit.setText(directory)

    def get_folder(self):
        return self.lineEdit.text()
