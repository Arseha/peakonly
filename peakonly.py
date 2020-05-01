import os
import sys
import urllib.request
import pymzml
import zipfile
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from processing_utils.roi import get_closest
from processing_utils.postprocess import ResultTable
from gui_utils.auxilary_utils import FileListWidget, FeatureListWidget, ProgressBarsList, ProgressBarsListItem
from gui_utils.mining import AnnotationParameterWindow, ReAnnotationParameterWindow
from gui_utils.processing import ProcessingParameterWindow
from gui_utils.training import TrainingParameterWindow
from gui_utils.evaluation import EvaluationParameterWindow
from gui_utils.data_splitting import SplitterParameterWindow
from gui_utils.threading import Worker


class MainWindow(QtWidgets.QMainWindow):
    # Initialization
    def __init__(self):
        super().__init__()
        self.threadpool = QtCore.QThreadPool()
        # create menu
        self.create_menu()

        # create list of opened *.mzML file
        self.list_of_files = self.create_list_of_files()

        # create list of founded features
        self.list_of_features = self.create_list_of_features()
        self.feature_parameters = None  # to do: add possibility process several batches

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

        canvas_files_features_layout = QtWidgets.QHBoxLayout()
        canvas_files_features_layout.addWidget(self.list_of_files, 15)
        canvas_files_features_layout.addLayout(canvas_layout, 70)
        canvas_files_features_layout.addWidget(self.list_of_features, 15)

        self.pb_list = ProgressBarsList(self)
        scrollable_pb_list = QtWidgets.QScrollArea()
        scrollable_pb_list.setWidget(self.pb_list)
        scrollable_pb_list.setWidgetResizable(True)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(canvas_files_features_layout, 90)
        main_layout.addWidget(scrollable_pb_list, 10)

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
        file_import_mzML = QtWidgets.QAction('Import *.mzML', self)
        file_import_mzML.triggered.connect(self.open_files)
        file_import.addAction(file_import_mzML)

        file_export = QtWidgets.QMenu('Export', self)
        file_export_features = QtWidgets.QAction('Export features as *.csv file', self)
        file_export_features.triggered.connect(self.export_features)
        file_export.addAction(file_export_features)

        file.addMenu(file_import)
        file.addMenu(file_export)

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
        data_download_models.triggered.connect(partial(self.download_button, mode='models'))
        data_download.addAction(data_download_models)
        data_download_annotated_data = QtWidgets.QAction('Download annotated data', self)
        data_download_annotated_data.triggered.connect(partial(self.download_button, mode='data'))
        data_download.addAction(data_download_annotated_data)
        data_download_example = QtWidgets.QAction('Download *.mzML example', self)
        data_download_example.triggered.connect(partial(self.download_button, mode='example'))
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
        visual_tic.triggered.connect(self.plot_tic_button)

        visual_eic = QtWidgets.QAction('Plot EIC', self)
        visual_eic.triggered.connect(self.get_eic_parameters)

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
        list_of_files.connectRightClick(partial(FileContextMenu, self))
        return list_of_files

    def create_list_of_features(self):
        list_of_features = FeatureListWidget()
        list_of_features.connectDoubleClick(self.feature_click)
        list_of_features.connectRightClick(partial(FeatureContextMenu, self))
        return list_of_features

    # Auxiliary methods
    def feature_click(self, item):
        feature = self.list_of_features.get_feature(item)
        self.plot_feature(feature)

    def open_files(self):
        filter = 'mzML (*.mzML)'
        filenames = QtWidgets.QFileDialog.getOpenFileNames(None, None, None, filter)[0]
        for name in filenames:
            self.list_of_files.addFile(name)

    def set_features(self, obj):
        features, parameters = obj
        self.list_of_features.clear()
        for feature in features:
            self.list_of_features.add_feature(feature)
        self.feature_parameters = parameters

    def export_features(self):
        if self.list_of_features.count() > 0:
            # to do: features should be QTreeWidget (root should keep basic information: files and parameters)
            files = self.feature_parameters['files']
            table = ResultTable(files, self.list_of_features.features)
            table.fill_zeros(self.feature_parameters['delta mz'])
            file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Export features', '',
                                                                 'csv (*.csv)')
            table.to_csv(file_name)

    def get_eic_parameters(self):
        subwindow = EICParameterWindow(self)
        subwindow.show()

    @staticmethod
    def show_downloading_progress(number_of_block, size_of_block, total_size, pb):
        pb.setValue(int(number_of_block * size_of_block * 100 / total_size))

    def threads_finisher(self, text=None, icon=None, pb=None):
        if pb is not None:
            self.pb_list.removeItem(pb)
            pb.setParent(None)
        if text is not None:
            msg = QtWidgets.QMessageBox(self)
            msg.setText(text)
            msg.setIcon(icon)
            msg.exec_()

    def plotter(self, obj):
        if not self.label2line:  # in case if 'feature' was plotted
            self.figure.clear()
            self.ax = self.figure.add_subplot(111)

        line = self.ax.plot(obj['x'], obj['y'], label=obj['label'])
        self.label2line[obj['label']] = line[0]  # save line
        self.ax.legend(loc='best')
        self.canvas.draw()

    # Buttons, which creates threads
    def download_button(self, mode):
        if mode == 'models':
            text = 'Downloading trained models:'
        elif mode == 'data':
            text = 'Downloading annotated data:'
        elif mode == 'example':
            text = 'Downloading *.mzML example:'
        else:
            assert False, mode

        pb = ProgressBarsListItem(text, parent=self.pb_list)
        self.pb_list.addItem(pb)
        worker = Worker(self.download, download=True, mode=mode)
        worker.signals.download_progress.connect(partial(self.show_downloading_progress, pb=pb))
        worker.signals.finished.connect(partial(self.threads_finisher,
                                                text='Download is successful',
                                                icon=QtWidgets.QMessageBox.Information,
                                                pb=pb))
        self.threadpool.start(worker)

    def plot_tic_button(self):
        for file in self.list_of_files.selectedItems():
            file = file.text()
            label = f'TIC: {file[:file.rfind(".")]}'
            if label not in self.label2line:
                path = self.list_of_files.file2path[file]

                pb = ProgressBarsListItem(f'Plotting TIC: {file}', parent=self.pb_list)
                self.pb_list.addItem(pb)
                worker = Worker(self.construct_tic, path, label)
                worker.signals.progress.connect(pb.setValue)
                worker.signals.result.connect(self.plotter)
                worker.signals.finished.connect(partial(self.threads_finisher, pb=pb))

                self.threadpool.start(worker)

    def plot_eic_button(self, mz, delta):
        for file in self.list_of_files.selectedItems():
            file = file.text()
            label = f'EIC {mz:.4f} ± {delta:.4f}: {file[:file.rfind(".")]}'
            if label not in self.label2line:
                path = self.list_of_files.file2path[file]

                pb = ProgressBarsListItem(f'Plotting EIC (mz={mz:.4f}): {file}', parent=self.pb_list)
                self.pb_list.addItem(pb)
                worker = Worker(self.construct_eic, path, label, mz, delta)
                worker.signals.progress.connect(pb.setValue)
                worker.signals.result.connect(self.plotter)
                worker.signals.finished.connect(partial(self.threads_finisher, pb=pb))

                self.threadpool.start(worker)

    # Main functionality
    @staticmethod
    def download(mode, progress_callback):
        """
        Download necessary data
        Parameters
        ----------
        mode : str
            one of three ('models', 'data', 'example')
        progress_callback : QtCore.pyqtSignal
            indicating progress in %
        """
        if mode == 'models':
            folder = 'data/weights'
            if not os.path.exists(folder):
                os.mkdir(folder)
            # Classifier
            url = 'https://getfile.dokpub.com/yandex/get/https://yadi.sk/d/rAhl2u7WeIUGYA'
            file = os.path.join(folder, 'Classifier.pt')
            urllib.request.urlretrieve(url, file, progress_callback.emit)
            # Segmentator
            url = 'https://getfile.dokpub.com/yandex/get/https://yadi.sk/d/9m5e3C0q0HKbuw'
            file = os.path.join(folder, 'Segmentator.pt')
            urllib.request.urlretrieve(url, file, progress_callback.emit)
            # RecurrentCNN
            url = 'https://getfile.dokpub.com/yandex/get/https://yadi.sk/d/1IrXRWDWhANqKw'
            file = os.path.join(folder, 'RecurrentCNN.pt')
            urllib.request.urlretrieve(url, file, progress_callback.emit)
        elif mode == 'data':
            folder = 'data/annotation'
            if not os.path.exists(folder):
                os.mkdir(folder)
            url = 'https://getfile.dokpub.com/yandex/get/https://yadi.sk/d/f6BiwqWYF4UVnA'
            file = 'data/annotation/annotation.zip'
            urllib.request.urlretrieve(url, file, progress_callback.emit)
            with zipfile.ZipFile(file) as zip_file:
                zip_file.extractall(folder)
            os.remove(file)
        elif mode == 'example':
            url = 'https://getfile.dokpub.com/yandex/get/https://yadi.sk/d/BhQNge3db7M2Lw'
            file = 'data/mix.mzML'
            urllib.request.urlretrieve(url, file, progress_callback.emit)
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
            subwindow = ReAnnotationParameterWindow(self)
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
    @staticmethod
    def construct_tic(path, label, progress_callback=None):
        run = pymzml.run.Reader(path)
        t_measure = None
        time = []
        TIC = []
        spectrum_count = run.get_spectrum_count()
        for i, scan in enumerate(run):
            if scan.ms_level == 1:
                TIC.append(scan.TIC)  # get total ion of scan
                t, measure = scan.scan_time  # get scan time
                time.append(t)
                if not t_measure:
                    t_measure = measure
                if progress_callback is not None and not i % 10:
                    progress_callback.emit(int(i * 100 / spectrum_count))
        if t_measure == 'second':
            time = np.array(time) / 60
        return {'x': time, 'y': TIC, 'label': label}

    @staticmethod
    def construct_eic(path, label, mz, delta, progress_callback=None):
        run = pymzml.run.Reader(path)
        t_measure = None
        time = []
        EIC = []
        spectrum_count = run.get_spectrum_count()
        for i, scan in enumerate(run):
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
                if progress_callback is not None and not i % 10:
                    progress_callback.emit(int(i * 100 / spectrum_count))
        return {'x': time, 'y': EIC, 'label': label}

    def plot_feature(self, feature, shifted=True):
        self.label2line = dict()  # empty plotted TIC and EIC
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        feature.plot(self.ax, shifted=shifted)
        self.canvas.draw()  # refresh canvas


class FileContextMenu(QtWidgets.QMenu):
    def __init__(self, parent: MainWindow):
        super().__init__(parent)

        self.parent = parent
        self.menu = QtWidgets.QMenu(parent)

        self.tic = QtWidgets.QAction('Plot TIC', parent)
        self.eic = QtWidgets.QAction('Plot EIC', parent)
        self.close = QtWidgets.QAction('Close', parent)

        self.menu.addAction(self.tic)
        self.menu.addAction(self.eic)
        self.menu.addAction(self.close)

        action = self.menu.exec_(QtGui.QCursor.pos())

        if action == self.tic:
            self.parent.plot_tic_button()
        elif action == self.eic:
            self.parent.get_eic_parameters()
        elif action == self.close:
            self.close_files()

    def close_files(self):
        for item in self.parent.list_of_files.selectedItems():
            self.parent.list_of_files.deleteFile(item)


class FeatureContextMenu(QtWidgets.QMenu):
    def __init__(self, parent: MainWindow):
        self.parent = parent
        super().__init__(parent)
        self.feature = None
        for item in self.parent.list_of_features.selectedItems():
            self.feature = self.parent.list_of_features.get_feature(item)
        self.menu = QtWidgets.QMenu(parent)

        self.with_rt_correction = QtWidgets.QAction('Plot with rt correction', parent)
        self.without_rt_correction = QtWidgets.QAction('Plot without rt correction', parent)

        self.menu.addAction(self.with_rt_correction)
        self.menu.addAction(self.without_rt_correction)

        action = self.menu.exec_(QtGui.QCursor.pos())

        if action == self.with_rt_correction:
            self.parent.plot_feature(self.feature, shifted=True)
        elif action == self.without_rt_correction:
            self.parent.plot_feature(self.feature, shifted=False)


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
        self.parent.plot_eic_button(mz, delta)
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