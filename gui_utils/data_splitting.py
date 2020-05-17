import os
import json
import numpy as np
from shutil import rmtree, copyfile
from PyQt5 import QtWidgets
from gui_utils.auxilary_utils import GetFoldersWidget


class SplitterParameterWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        self.parent = parent
        super().__init__(parent)
        self.setWindowTitle('peakonly: data splitting')

        self.json_files = set()

        # folder selection
        main_layout = QtWidgets.QVBoxLayout()
        self.folder_widget = GetFoldersWidget('Choose folder with annotated data: ')
        main_layout.addWidget(self.folder_widget)

        # calculation of total number of ROIs
        number_layout = QtWidgets.QHBoxLayout()
        calculate_rois_button = QtWidgets.QPushButton('Calculate number of ROIs in chosen folders: ')
        calculate_rois_button.clicked.connect(self.get_rois_number)
        self.rois_number_label = QtWidgets.QLabel()
        self.rois_number_label.setText('...')
        number_layout.addWidget(calculate_rois_button)
        number_layout.addWidget(self.rois_number_label)

        # set sizes
        val_size_layout = QtWidgets.QHBoxLayout()
        val_size_label = QtWidgets.QLabel()
        val_size_label.setText('Size of validation subset: ')
        self.val_size_getter = QtWidgets.QLineEdit(self)
        self.val_size_getter.setText('...')
        val_size_layout.addWidget(val_size_label)
        val_size_layout.addWidget(self.val_size_getter)

        test_size_layout = QtWidgets.QHBoxLayout()
        test_size_label = QtWidgets.QLabel()
        test_size_label.setText('Size of test subset:          ')
        self.test_size_getter = QtWidgets.QLineEdit(self)
        self.test_size_getter.setText('...')
        test_size_layout.addWidget(test_size_label)
        test_size_layout.addWidget(self.test_size_getter)

        # button
        self.split_button = QtWidgets.QPushButton('Split data')
        self.split_button.clicked.connect(self.split_data)

        # set main layout
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(self.folder_widget)
        main_layout.addLayout(number_layout)
        main_layout.addLayout(val_size_layout)
        main_layout.addLayout(test_size_layout)
        main_layout.addWidget(self.split_button)
        self.setLayout(main_layout)

    def get_rois_number(self):
        self.json_files = set()
        folders = self.folder_widget.get_folders()
        for folder in folders:
            self.search_json_files(folder)
        self.rois_number_label.setText(f'{len(self.json_files)}')
        val_size = int(0.15 * len(self.json_files))
        test_size = int(0.15 * len(self.json_files))
        self.val_size_getter.setText(f'{val_size}')
        self.test_size_getter.setText(f'{test_size}')

    def search_json_files(self, path):
        for sub_path in os.listdir(path):
            if not sub_path.startswith('.') and sub_path != '__MACOSX':
                sub_path = os.path.join(path, sub_path)
                if os.path.isdir(sub_path):
                    self.search_json_files(sub_path)
                elif sub_path.endswith('.json'):
                    self.json_files.add(sub_path)

    def split_data(self):
        try:
            self.json_files = set()
            folders = self.folder_widget.get_folders()
            for folder in folders:
                self.search_json_files(folder)
            if not self.json_files:
                raise ValueError
            val_size = int(self.val_size_getter.text())
            test_size = int(self.test_size_getter.text())
        except ValueError:
            # popup window with exception
            msg = QtWidgets.QMessageBox(self)
            msg.setText("Directory should include any *.json files and \n"
                        "sizes of test and validation datasets should be integers")
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.exec_()
            return None

        # delete old data and create new folders
        def remove_dir(path):
            try:
                rmtree(path)
            except OSError:
                pass

        def create_dir(path):
            try:
                os.mkdir(path)
            except OSError:
                pass

        train_dir = 'data/train'
        val_dir = 'data/val'
        test_dir = 'data/test'

        remove_dir(train_dir)
        remove_dir(val_dir)
        remove_dir(test_dir)

        create_dir(train_dir)
        create_dir(val_dir)
        create_dir(test_dir)

        # get labels of ROIs
        label2file = {0: [], 1: []}
        for file in self.json_files:
            with open(file) as json_file:
                roi = json.load(json_file)
                label2file[roi['label']].append(file)

        # copy files to val folder
        val0size = val_size // 2
        val1size = val_size - val0size
        for i in range(val0size):
            a = np.random.choice(np.arange(len(label2file[0])))
            file_name = label2file[0][a]
            copyfile(file_name, os.path.join(val_dir, os.path.basename(file_name)))
            label2file[0].pop(a)

        for i in range(val1size):
            a = np.random.choice(np.arange(len(label2file[1])))
            file_name = label2file[1][a]
            copyfile(file_name, os.path.join(val_dir, os.path.basename(file_name)))
            label2file[1].pop(a)

        # copy files to test folder
        test0size = test_size // 2
        test1size = test_size - test0size
        for i in range(test0size):
            a = np.random.choice(np.arange(len(label2file[0])))
            file_name = label2file[0][a]
            copyfile(file_name, os.path.join(test_dir, os.path.basename(file_name)))
            label2file[0].pop(a)

        for i in range(test1size):
            a = np.random.choice(np.arange(len(label2file[1])))
            file_name = label2file[1][a]
            copyfile(file_name, os.path.join(test_dir, os.path.basename(file_name)))
            label2file[1].pop(a)

        # the rest files copy to train folder
        for k, v in label2file.items():
            for file_name in v:
                copyfile(file_name, os.path.join(train_dir, os.path.basename(file_name)))
