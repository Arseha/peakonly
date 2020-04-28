import os
from PyQt5 import QtWidgets, QtCore


class ClickableListWidget(QtWidgets.QListWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.double_click = None
        self.right_click = None

    def mousePressEvent(self, QMouseEvent):
        super(QtWidgets.QListWidget, self).mousePressEvent(QMouseEvent)
        if QMouseEvent.button() == QtCore.Qt.RightButton and self.right_click is not None:
            self.right_click()

    def mouseDoubleClickEvent(self, QMouseEvent):
        if self.double_click is not None:
            if QMouseEvent.button() == QtCore.Qt.LeftButton:
                item = self.itemAt(QMouseEvent.pos())
                if item is not None:
                    self.double_click(item)

    def connectDoubleClick(self, method):
        """
        Set a callable object which should be called when a user double-clicks on item
        Parameters
        ----------
        method : callable
            any callable object
        Returns
        -------
        - : None
        """
        self.double_click = method

    def connectRightClick(self, method):
        """
        Set a callable object which should be called when a user double-clicks on item
        Parameters
        ----------
        method : callable
            any callable object
        Returns
        -------
        - : None
        """
        self.right_click = method


class FileListWidget(ClickableListWidget):
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

    def getPath(self, item: QtWidgets.QListWidgetItem):
        return self.file2path[item.text()]


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


class ProgressBarsListItem(QtWidgets.QWidget):
    def __init__(self, text, pb=None, parent=None):
        super().__init__(parent)
        self.pb = pb
        if self.pb is None:
            self.pb = QtWidgets.QProgressBar()

        label = QtWidgets.QLabel(self)
        label.setText(text)

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(label, 30)
        main_layout.addWidget(self.pb, 70)

        self.setLayout(main_layout)

    def setValue(self, value):
        self.pb.setValue(value)


class ProgressBarsList(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.main_layout)

    def removeItem(self, item):
        self.layout().removeWidget(item)

    def addItem(self, item):
        self.layout().addWidget(item)


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


class GetFoldersWidget(QtWidgets.QWidget):
    def __init__(self, label, parent=None):
        super().__init__(parent)

        button = QtWidgets.QToolButton()
        button.setText('...')
        button.clicked.connect(self.add_folder)

        self.lineEdit = QtWidgets.QToolButton()
        self.lineEdit.setText(label)

        folder_getter_layout = QtWidgets.QHBoxLayout()
        folder_getter_layout.addWidget(self.lineEdit, 85)
        folder_getter_layout.addWidget(button, 15)

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(folder_getter_layout)
        main_layout.addWidget(self.list_widget)

        self.setLayout(main_layout)

    def add_folder(self):
        directory = str(QtWidgets.QFileDialog.getExistingDirectory())
        if directory:
            self.list_widget.addItem(directory)

    def get_folders(self):
        folders = [f.text() for f in self.list_widget.selectedItems()]
        return folders


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
