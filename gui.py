import os
import sys
import urllib.request
import pymzml
import zipfile
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from PyQt5 import QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from processing_utils.roi import get_closest
from gui_utils.auxilary_utils import FileListWidget, FeatureListWidget
from gui_utils.mining import AnnotationParameterWindow, ReAnnotationParameterWindow
from gui_utils.processing import ProcessingParameterWindow
from gui_utils.training import TrainingParameterWindow
from gui_utils.evaluation import EvaluationParameterWindow
from gui_utils.data_splitting import SplitterParameterWindow


class MainWindow(QtWidgets.QMainWindow):
    # Initialization
    def __init__(self):
        self.magic = 0  # (used to change shifted parameter in feature.plot()) to do: create ContextMenu?
        super().__init__()
        # create menu
        self.create_menu()

        # create list of opened *.mzML file
        self.list_of_files = self.create_list_of_files()

        # create list of founded features
        self.list_of_features = self.create_list_of_features()

        # Main canvas and toolbar
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)  # plot here
        self.label2line = dict()  # a label (aka line name) to plotted line
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Layouts
        canvas_layout = QtWidgets.QVBoxLayout()
        canvas_layout.addWidget(self.toolbar)
        canvas_layout.addWidget(self.canvas)

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(self.list_of_files, 15)
        main_layout.addLayout(canvas_layout, 70)
        main_layout.addWidget(self.list_of_features, 15)
        widget = QtWidgets.QWidget()
        widget.setLayout(main_layout)

        self.setCentralWidget(widget)

        # Set geometry and title
        self.setGeometry(300, 300, 900, 600)
        self.setWindowTitle('peakonly')
        self.show()

    def create_menu(self):
        menu = self.menuBar()

        # file submenu
        file = menu.addMenu('File')

        file_import = QtWidgets.QMenu('Import', self)
        file_import_single = QtWidgets.QAction('Import *.mzML', self)
        file_import_single.triggered.connect(self.open_files)
        file_import.addAction(file_import_single)

        file.addMenu(file_import)

        # data submenu
        data = menu.addMenu('Data')

        data_processing = QtWidgets.QMenu('Processing', self)
        data_processing_all = QtWidgets.QAction("'All-in-one'", self)
        data_processing_all.triggered.connect(partial(self.data_processing, 'all in one'))
        data_processing.addAction(data_processing_all)
        data_processing_sequential = QtWidgets.QAction('Sequential', self)
        data_processing_sequential.triggered.connect(partial(self.data_processing, 'sequential'))
        data_processing.addAction(data_processing_sequential)

        data_mining = QtWidgets.QMenu('Mining', self)
        data_mining_novel_annotation = QtWidgets.QAction('Manual annotation', self)
        data_mining_novel_annotation.triggered.connect(partial(self.create_dataset, mode='manual'))
        data_mining.addAction(data_mining_novel_annotation)
        data_mining_novel_reannotation = QtWidgets.QAction('Reannotation', self)
        data_mining_novel_reannotation.triggered.connect(partial(self.create_dataset, mode='reannotation'))
        data_mining.addAction(data_mining_novel_reannotation)

        data_download = QtWidgets.QMenu('Download', self)
        data_download_models = QtWidgets.QAction('Download trained models', self)
        data_download_models.triggered.connect(partial(self.download, mode='models'))
        data_download.addAction(data_download_models)
        data_download_annotated_data = QtWidgets.QAction('Download annotated data', self)
        data_download_annotated_data.triggered.connect(partial(self.download, mode='data'))
        data_download.addAction(data_download_annotated_data)
        data_download_example = QtWidgets.QAction('Download *.mzML example', self)
        data_download_example.triggered.connect(partial(self.download, mode='example'))
        data_download.addAction(data_download_example)

        data_split = QtWidgets.QAction('Split data', self)
        data_split.triggered.connect(self.split_data)

        data.addMenu(data_processing)
        data.addMenu(data_mining)
        data.addMenu(data_download)
        data.addAction(data_split)

        # Processing model submenu
        model = menu.addMenu('Model')

        model_training = QtWidgets.QMenu('Training', self)
        model_training_all = QtWidgets.QAction("'All-in-one'", self)
        model_training_all.triggered.connect(partial(self.model_training, 'all in one'))
        model_training.addAction(model_training_all)
        model_training_sequential = QtWidgets.QAction('Sequential', self)
        model_training_sequential.triggered.connect(partial(self.model_training, 'sequential'))
        model_training.addAction(model_training_sequential)

        model_fine_tuning = QtWidgets.QMenu('Fine-tuning', self)
        model_fine_tuning_all = QtWidgets.QAction("'All-in-one'", self)
        model_fine_tuning_all.triggered.connect(partial(self.model_fine_tuning, 'all in one'))
        model_fine_tuning.addAction(model_fine_tuning_all)
        model_fine_tuning_sequential = QtWidgets.QAction('Sequential', self)
        model_fine_tuning_sequential.triggered.connect(partial(self.model_fine_tuning, 'sequential'))
        model_fine_tuning.addAction(model_fine_tuning_sequential)

        model_evaluation = QtWidgets.QMenu('Evaluation', self)
        model_evaluation_all = QtWidgets.QAction("'All-in-one'", self)
        model_evaluation_all.triggered.connect(partial(self.model_evaluation, 'all in one'))
        model_evaluation.addAction(model_evaluation_all)
        model_evaluation_sequential = QtWidgets.QAction('Sequential', self)
        model_evaluation_sequential.triggered.connect(partial(self.model_evaluation, 'sequential'))
        model_evaluation.addAction(model_evaluation_sequential)

        model.addMenu(model_training)
        model.addMenu(model_fine_tuning)
        model.addMenu(model_evaluation)


        # visualization submenu
        visual = menu.addMenu('Visualization')

        visual_tic = QtWidgets.QAction('Plot TIC', self)
        visual_tic.triggered.connect(self.plot_TIC)

        visual_eic = QtWidgets.QAction('Plot EIC', self)
        visual_eic.triggered.connect(self.get_EIC_parameters)

        visual_clear = QtWidgets.QAction('Clear', self)
        visual_clear.triggered.connect(self.open_clear_window)

        visual.addAction(visual_tic)
        visual.addAction(visual_eic)
        visual.addAction(visual_clear)

    def open_clear_window(self):
        subwindow = ClearMainCanvasWindow(self)
        subwindow.show()

    def create_list_of_files(self):
        # List of opened files
        list_of_files = FileListWidget()
        list_of_files.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        list_of_files.itemDoubleClicked.connect(self.file_click)
        return list_of_files

    def create_list_of_features(self):
        list_of_features = FeatureListWidget()
        list_of_features.itemDoubleClicked.connect(self.feature_click)
        return list_of_features

    # Auxiliary methods
    def file_click(self, item):
        FileContextMenu(self, item)

    def feature_click(self, item):
        feature = self.list_of_features.get_feature(item)
        self.plot_feature(feature)

    def open_files(self):
        filter = 'mzML (*.mzML)'
        filenames = QtWidgets.QFileDialog.getOpenFileNames(None, None, None, filter)[0]
        for name in filenames:
            self.list_of_files.addFile(name)

    def set_features(self, features):
        for feature in features:
            self.list_of_features.add_feature(feature)

    def get_EIC_parameters(self):
        subwindow = EICParameterWindow(self)
        subwindow.show()

    # Main functionality
    @staticmethod
    def download(mode):
        """
        Download necessary data
        Parameters
        ----------
        mode : str
            one of three ('models', 'data', 'example')
        """
        if mode == 'models':
            # to do: add functionality to download models
            pass
        elif mode == 'data':
            folder = 'data/annotation'
            if not os.path.exists(folder):
                os.mkdir(folder)
            url = 'https://getfile.dokpub.com/yandex/get/https://yadi.sk/d/f6BiwqWYF4UVnA'
            file = 'data/annotation/annotation.zip'
            urllib.request.urlretrieve(url, file)
            with zipfile.ZipFile(file) as zip_file:
                zip_file.extractall(folder)
            os.remove(file)
            # to do: create window, which writes that download is successful
        elif mode == 'example':
            url = 'https://getfile.dokpub.com/yandex/get/https://yadi.sk/d/BhQNge3db7M2Lw'
            file = 'data/mix.mzML'
            urllib.request.urlretrieve(url, file)
            # to do: create window, which writes that download is successful
        else:
            assert False, mode

    def split_data(self):
        subwindow = SplitterParameterWindow(self)
        subwindow.show()

    def create_dataset(self, mode='manual'):
        if mode != 'reannotation':
            files = [self.list_of_files.file2path[self.list_of_files.item(i).text()]
                     for i in range(self.list_of_files.count())]
            subwindow = AnnotationParameterWindow(files, mode, self)
            subwindow.show()
        else:
            subwindow = ReAnnotationParameterWindow(mode, self)
            subwindow.show()

    def data_processing(self, mode):
        files = [self.list_of_files.file2path[self.list_of_files.item(i).text()]
                 for i in range(self.list_of_files.count())]
        subwindow = ProcessingParameterWindow(files, mode, self)
        subwindow.show()

    # Model functionality
    def model_training(self, mode):
        subwindow = TrainingParameterWindow(mode, self)
        subwindow.show()

    def model_fine_tuning(self, mode):
        pass

    def model_evaluation(self, mode):
        subwindow = EvaluationParameterWindow(mode, self)
        subwindow.show()

    # Visualization
    def plot(self, x, y):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(x, y)
        self.canvas.draw()  # refresh canvas

    def plot_TIC(self):
        files = [f.text() for f in self.list_of_files.selectedItems()]
        for file in files:
            path = self.list_of_files.file2path[file]
            run = pymzml.run.Reader(path)
            t_measure = None
            time = []
            TIC = []
            label = f'TIC: {file[:file.rfind(".")]}'
            if label not in self.label2line:
                for scan in run:
                    if scan.ms_level == 1:
                        TIC.append(scan.TIC)  # get total ion of scan
                        t, measure = scan.scan_time  # get scan time
                        time.append(t)
                        if not t_measure:
                            t_measure = measure

                line = self.ax.plot(time, TIC, label=label)
                self.label2line[label] = line[0]  # save line
                self.ax.legend(loc='best')
        self.canvas.draw()  # refresh canvas

    def plot_EIC(self, mz, delta):
        files = [f.text() for f in self.list_of_files.selectedItems()]
        for file in files:
            path = self.list_of_files.file2path[file]
            run = pymzml.run.Reader(path)
            t_measure = None
            time = []
            EIC = []
            label = f'EIC {mz:.4f} ± {delta:.4f}: {file[:file.rfind(".")]}'
            if label not in self.label2line:
                for scan in run:
                    if scan.ms_level == 1:
                        t, measure = scan.scan_time  # get scan time
                        time.append(t)
                        pos = np.searchsorted(scan.mz, mz)
                        closest = get_closest(scan.mz, mz, pos)
                        if abs(scan.mz[closest] - mz) < delta:
                            EIC.append(scan.i[closest])
                        else:
                            EIC.append(0)
                        if not t_measure:
                            t_measure = measure

                line = self.ax.plot(time, EIC, label=label)
                self.label2line[label] = line[0]  # save line to remove then
                self.ax.legend(loc='best')
        self.canvas.draw()  # refresh canvas

    def plot_feature(self, feature):
        self.magic += 1
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        if self.magic % 2:
            feature.plot(self.ax, shifted=True)
        else:
            feature.plot(self.ax, shifted=False)
        self.canvas.draw()  # refresh canvas


class FileContextMenu(QtWidgets.QMenu):
    def __init__(self, window: MainWindow, item: QtWidgets.QListWidgetItem):
        super().__init__(window)
        for i in window.list_of_files.selectedItems():
            i.setSelected(False)
        item.setSelected(True)

        self.window = window
        self.item = item
        self.menu = QtWidgets.QMenu(window)

        self.tic = QtWidgets.QAction('Plot TIC', window)
        self.eic = QtWidgets.QAction('Plot EIC', window)
        self.close = QtWidgets.QAction('Close', window)

        self.menu.addAction(self.tic)
        self.menu.addAction(self.eic)
        self.menu.addAction(self.close)

        action = self.menu.exec_(QtGui.QCursor.pos())

        if action == self.tic:
            self.window.plot_TIC()
        elif action == self.eic:
            self.window.get_EIC_parameters()
        elif action == self.close:
            self.close_file()

    def close_file(self):
        self.window.list_of_files.deleteFile(self.item)


class EICParameterWindow(QtWidgets.QDialog):
    def __init__(self, parent: MainWindow):
        self.parent = parent
        super().__init__(self.parent)

        mz_layout = QtWidgets.QHBoxLayout()
        mz_label = QtWidgets.QLabel(self)
        mz_label.setText('mz=')
        self.mz_getter = QtWidgets.QLineEdit(self)
        self.mz_getter.setText('100.000')
        mz_layout.addWidget(mz_label)
        mz_layout.addWidget(self.mz_getter)

        delta_layout = QtWidgets.QHBoxLayout()
        delta_label = QtWidgets.QLabel(self)
        delta_label.setText('delta=±')
        self.delta_getter = QtWidgets.QLineEdit(self)
        self.delta_getter.setText('0.005')
        delta_layout.addWidget(delta_label)
        delta_layout.addWidget(self.delta_getter)

        self.plot_button = QtWidgets.QPushButton('Plot')
        self.plot_button.clicked.connect(self.plot)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(mz_layout)
        layout.addLayout(delta_layout)
        layout.addWidget(self.plot_button)
        self.setLayout(layout)

    def plot(self):
        # to do: raise exception
        mz = float(self.mz_getter.text())
        delta = float(self.delta_getter.text())
        self.parent.plot_EIC(mz, delta)
        self.close()


class ClearMainCanvasWindow(QtWidgets.QDialog):
    def __init__(self, parent: MainWindow):
        self.parent = parent
        super().__init__(self.parent)

        self.plotted_list = QtWidgets.QListWidget()
        self.plotted_list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
        for label in self.parent.label2line:
            self.plotted_list.addItem(label)

        clear_selected_button = QtWidgets.QPushButton('Clear selected')
        clear_selected_button.clicked.connect(self.clear_selected)

        clear_all_button = QtWidgets.QPushButton('Clear all')
        clear_all_button.clicked.connect(self.clear_all)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(clear_selected_button)
        button_layout.addWidget(clear_all_button)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.plotted_list)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def clear_selected(self):
        for select_item in self.plotted_list.selectedItems():
            self.parent.ax.lines.remove(self.parent.label2line[select_item.text()])
            del self.parent.label2line[select_item.text()]
        if self.parent.label2line:  # still not empty
            self.parent.ax.legend(loc='best')
            # recompute the ax.dataLim
            self.parent.ax.relim()
            # update ax.viewLim using the new dataLim
            self.parent.ax.autoscale_view()
        else:
            self.parent.figure.clear()
            self.parent.ax = self.parent.figure.add_subplot(111)
        self.parent.canvas.draw()  # refresh canvas
        self.close()

    def clear_all(self):
        self.parent.label2line = dict()  # reinitialize
        self.parent.figure.clear()
        self.parent.ax = self.parent.figure.add_subplot(111)
        self.parent.canvas.draw()  # refresh canvas
        self.close()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
