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


class FeatureListWidget(QtWidgets.QListWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features = []

    def add_feature(self, feature):
        name = f'#{len(self.features)}: mz = {feature.mz:.4f}, rt = {feature.rtmin:.2f} - {feature.rtmax:.2f}'
        self.features.append(feature)
        self.addItem(name)

    def get_feature(self, item):
        number = item.text()
        number = int(number[number.find('#') + 1:number.find(':')])
        return self.features[number]

    def get_all(self):
        features = []
        for i in range(self.count()):
            item = self.item(i)
            features.append(self.get_feature(item))
        return features


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


class GetFileWidget(QtWidgets.QWidget):
    def __init__(self, extension, default_file, parent):
        super().__init__(parent)

        self.extension = extension

        button = QtWidgets.QToolButton()
        button.setText('...')
        button.clicked.connect(self.set_file)

        self.lineEdit = QtWidgets.QToolButton()
        self.lineEdit.setText(default_file)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.lineEdit, 85)
        layout.addWidget(button, 15)

        self.setLayout(layout)

    def set_file(self):
        filter = f'{self.extension} (*.{self.extension})'
        file, _ = QtWidgets.QFileDialog.getOpenFileName(None, None, None, filter)
        if file:
            self.lineEdit.setText(file)

    def get_file(self):
        return self.lineEdit.text()
