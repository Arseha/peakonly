import os
import sys
import urllib.request
import zipfile
from functools import partial
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtGui, QtCore
from processing_utils.postprocess import ResultTable
from processing_utils.run_utils import find_mzML
from gui_utils.abstract_main_window import AbtractMainWindow
from gui_utils.auxilary_utils import ProgressBarsListItem
from gui_utils.mining import AnnotationParameterWindow, ReAnnotationParameterWindow
from gui_utils.visualization import EICParameterWindow, VisualizationWindow
from gui_utils.processing import ProcessingParameterWindow
from gui_utils.training import TrainingParameterWindow
from gui_utils.evaluation import EvaluationParameterWindow
from gui_utils.data_splitting import SplitterParameterWindow
from gui_utils.threading import Worker


class MainWindow(AbtractMainWindow):
    # Initialization
    def __init__(self):
        super().__init__()
        # create menu
        self._create_menu()

        # tune list of files
        self._list_of_files.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._list_of_files.connectRightClick(partial(FileContextMenu, self))

        # tune list of features
        self._list_of_features.connectDoubleClick(self.plot_feature)
        self._list_of_features.connectRightClick(partial(FeatureContextMenu, self))

        self._init_ui()

        # Set geometry and title
        self.setGeometry(300, 300, 900, 600)
        self.setWindowTitle('peakonly')
        self.show()

    def _create_menu(self):
        menu = self.menuBar()

        # file submenu
        file = menu.addMenu('File')

        file_import = QtWidgets.QMenu('Open', self)
        file_import_mzML = QtWidgets.QAction('Open *.mzML', self)
        file_import_mzML.triggered.connect(self._open_file)
        file_import.addAction(file_import_mzML)
        file_import_folder_mzML = QtWidgets.QAction('Open folder with *.mzML files', self)
        file_import_folder_mzML.triggered.connect(self._open_folder)
        file_import.addAction(file_import_folder_mzML)


        file_export = QtWidgets.QMenu('Save', self)
        file_export_features_csv = QtWidgets.QAction('Save a *.csv file with detected features', self)
        file_export_features_csv.triggered.connect(partial(self._export_features, 'csv'))
        file_export.addAction(file_export_features_csv)
        file_export_features_png = QtWidgets.QAction('Save features as *.png files', self)
        file_export_features_png.triggered.connect(partial(self._export_features, 'png'))
        file_export.addAction(file_export_features_png)
        file_export_feature_traces_csv = QtWidgets.QAction('Save features traces as *.csv files', self)
        file_export_feature_traces_csv.triggered.connect(partial(self._export_features, 'traces_csv'))
        file_export.addAction(file_export_feature_traces_csv)

        file_clear = QtWidgets.QMenu('Clear', self)
        file_clear_features = QtWidgets.QAction('Clear panel with detected features', self)
        file_clear_features.triggered.connect(self._list_of_features.clear)
        file_clear.addAction(file_clear_features)

        file_exit = QtWidgets.QAction("Exit", self)
        file_exit.triggered.connect(QtWidgets.QApplication.quit)  # to do: create visualization

        file.addMenu(file_import)
        file.addMenu(file_export)
        file.addMenu(file_clear)
        file.addAction(file_exit)

        # data submenu
        data = menu.addMenu('Data')

        data_processing = QtWidgets.QAction('Feature detection', self)
        data_processing.triggered.connect(partial(self._data_processing, 'simple'))

        data_download = QtWidgets.QMenu('Download', self)
        data_download_models = QtWidgets.QAction('Download trained models', self)
        data_download_models.triggered.connect(partial(self._download_button, mode='models'))
        data_download.addAction(data_download_models)
        data_download_annotated_data = QtWidgets.QAction('Download annotated data', self)
        data_download_annotated_data.triggered.connect(partial(self._download_button, mode='data'))
        data_download.addAction(data_download_annotated_data)
        data_download_example = QtWidgets.QAction('Download *.mzML example', self)
        data_download_example.triggered.connect(partial(self._download_button, mode='example'))
        data_download.addAction(data_download_example)

        data_visualization = QtWidgets.QAction('Visualization', self)
        data_visualization.triggered.connect(self._open_visualization_window)  # to do: create visualization

        data.addAction(data_processing)
        data.addMenu(data_download)
        data.addAction(data_visualization)

        # advanced submenu
        advanced = menu.addMenu('Advanced')

        advanced_data_processing = QtWidgets.QMenu('Advanced feature detection', self)
        advanced_data_processing_all = QtWidgets.QAction('RecurrentCNN (testing)', self)
        advanced_data_processing_all.triggered.connect(partial(self._data_processing, 'all in one'))
        advanced_data_processing.addAction(advanced_data_processing_all)
        advanced_data_processing_sequential = QtWidgets.QAction('Two subsequent CNNs', self)
        advanced_data_processing_sequential.triggered.connect(partial(self._data_processing, 'sequential'))
        advanced_data_processing.addAction(advanced_data_processing_sequential)

        advanced_data_mining = QtWidgets.QMenu('Data mining', self)
        advanced_data_mining_manual = QtWidgets.QAction('Manual annotation', self)
        advanced_data_mining_manual.triggered.connect(partial(self._data_mining, mode='manual'))
        advanced_data_mining.addAction(advanced_data_mining_manual)
        advanced_data_mining_reannotation = QtWidgets.QAction('Reannotation', self)
        advanced_data_mining_reannotation.triggered.connect(partial(self._data_mining, mode='reannotation'))
        advanced_data_mining.addAction(advanced_data_mining_reannotation)
        advanced_data_mining_split = QtWidgets.QAction('Split data', self)
        advanced_data_mining_split.triggered.connect(self._split_data)

        advanced_model = QtWidgets.QMenu('Model', self)
        advanced_model_training = QtWidgets.QMenu('Training', self)  # training
        advanced_model_training_all = QtWidgets.QAction('RecurrentCNN (testing)', self)
        advanced_model_training_all.triggered.connect(partial(self._model_training, 'all in one'))
        advanced_model_training.addAction(advanced_model_training_all)
        advanced_model_training_sequential = QtWidgets.QAction('Two subsequent CNNs', self)
        advanced_model_training_sequential.triggered.connect(partial(self._model_training, 'sequential'))
        advanced_model_training.addAction(advanced_model_training_sequential)
        advanced_model_fine_tuning = QtWidgets.QMenu('Fine-tuning (in developing)', self)  # fine-tuning
        advanced_model_fine_tuning_all = QtWidgets.QAction('RecurrentCNN (testing)', self)
        advanced_model_fine_tuning_all.triggered.connect(partial(self._model_fine_tuning, 'all in one'))
        advanced_model_fine_tuning.addAction(advanced_model_fine_tuning_all)
        advanced_model_fine_tuning_sequential = QtWidgets.QAction('Two subsequent CNNs', self)
        advanced_model_fine_tuning_sequential.triggered.connect(partial(self._model_fine_tuning, 'sequential'))
        advanced_model_fine_tuning.addAction(advanced_model_fine_tuning_sequential)
        advanced_model_evaluation = QtWidgets.QMenu('Evaluation', self)  # evaluation
        advanced_model_evaluation_all = QtWidgets.QAction('RecurrentCNN (testing)', self)
        advanced_model_evaluation_all.triggered.connect(partial(self._model_evaluation, 'all in one'))
        advanced_model_evaluation.addAction(advanced_model_evaluation_all)
        advanced_model_evaluation_sequential = QtWidgets.QAction('Two subsequent CNNs', self)
        advanced_model_evaluation_sequential.triggered.connect(partial(self._model_evaluation, 'sequential'))
        advanced_model_evaluation.addAction(advanced_model_evaluation_sequential)
        advanced_model.addMenu(advanced_model_training)  # add to menu
        advanced_model.addMenu(advanced_model_fine_tuning)
        advanced_model.addMenu(advanced_model_evaluation)

        advanced.addMenu(advanced_data_processing)
        advanced.addMenu(advanced_data_mining)
        advanced.addMenu(advanced_model)

    def _init_ui(self):
        # Layouts
        files_layout = QtWidgets.QVBoxLayout()
        files_label = QtWidgets.QLabel(self)
        files_label.setText('Opened files:')
        files_layout.addWidget(files_label)
        files_layout.addWidget(self._list_of_files)

        features_layout = QtWidgets.QVBoxLayout()
        features_label = QtWidgets.QLabel(self)
        features_label.setText('Detected features:')
        features_layout.addWidget(features_label)
        features_layout.addWidget(self._list_of_features)

        canvas_layout = QtWidgets.QVBoxLayout()
        canvas_layout.addWidget(self._toolbar)
        canvas_layout.addWidget(self._canvas)

        canvas_files_features_layout = QtWidgets.QHBoxLayout()
        canvas_files_features_layout.addLayout(files_layout, 15)
        canvas_files_features_layout.addLayout(canvas_layout, 70)
        canvas_files_features_layout.addLayout(features_layout, 15)

        scrollable_pb_list = QtWidgets.QScrollArea()
        scrollable_pb_list.setWidget(self._pb_list)
        scrollable_pb_list.setWidgetResizable(True)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(canvas_files_features_layout, 90)
        main_layout.addWidget(scrollable_pb_list, 10)

        widget = QtWidgets.QWidget()
        widget.setLayout(main_layout)

        self.setCentralWidget(widget)

    # Auxiliary methods
    def _open_file(self):
        files_names = QtWidgets.QFileDialog.getOpenFileNames(None, '', '', 'mzML (*.mzML)')[0]
        for name in files_names:
            self._list_of_files.addFile(name)

    def _open_folder(self):
        path = str(QtWidgets.QFileDialog.getExistingDirectory())
        for name in sorted(find_mzML(path)):
            self._list_of_files.addFile(name)

    def _export_features(self, mode):
        if self._list_of_features.count() > 0:
            if mode == 'csv':
                # to do: features should be QTreeWidget (root should keep basic information: files and parameters)
                files = self._feature_parameters['files']
                table = ResultTable(files, self._list_of_features.features)
                table.fill_zeros(self._feature_parameters['delta mz'])
                file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Export features', '',
                                                                     'csv (*.csv)')
                if file_name:
                    table.to_csv(file_name)
            elif mode == 'png':
                directory = str(QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose a directory where to save'))

                worker = Worker(self._save_features_png, features=self._list_of_features.features, directory=directory)
                self.run_thread('Saving features as *.png files:', worker)
            elif mode == 'traces_csv':
                directory = str(QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose a directory where to save'))

                worker = Worker(self._save_feature_traces_csv, features=self._list_of_features.features, directory=directory)
                self.run_thread('Saving feature traces as *.csv files', worker)
            else:
                assert False, mode
        else:
            msg = QtWidgets.QMessageBox(self)
            msg.setText('You should firstly detect features in *mzML files:\n'
                        'Data -> Feature detection')
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.exec_()

    def _get_eic_parameters(self):
        subwindow = EICParameterWindow(self)
        subwindow.show()

    @staticmethod
    def _show_downloading_progress(number_of_block, size_of_block, total_size, pb):
        pb.setValue(int(number_of_block * size_of_block * 100 / total_size))

    # Buttons, which creates threads
    def _download_button(self, mode):
        if mode == 'models':
            text = 'Downloading trained models:'
        elif mode == 'data':
            text = 'Downloading annotated data:'
        elif mode == 'example':
            text = 'Downloading *.mzML example:'
        else:
            assert False, mode

        pb = ProgressBarsListItem(text, parent=self._pb_list)
        self._pb_list.addItem(pb)
        worker = Worker(self._download, download=True, mode=mode)
        worker.signals.download_progress.connect(partial(self._show_downloading_progress, pb=pb))
        worker.signals.finished.connect(partial(self._threads_finisher,
                                                text='Download is successful',
                                                icon=QtWidgets.QMessageBox.Information,
                                                pb=pb))
        self._thread_pool.start(worker)

    # Main functionality
    @staticmethod
    def _download(mode, progress_callback):
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

    @staticmethod
    def _save_features_png(features, directory, progress_callback):
        fig = plt.figure()
        for i, feature in enumerate(features):
            ax = fig.add_subplot(111)
            feature.plot(ax, shifted=True)
            fig.savefig(os.path.join(directory, f'{i}.png'))
            fig.clear()
            progress_callback.emit(int(i * 100 / len(features)))
        plt.close(fig)

    @staticmethod
    def _save_feature_traces_csv(features, directory, progress_callback):
        for i, feature in enumerate(features):
            feature.save_as_csv(os.path.join(directory, f'{i}.csv'))
            progress_callback.emit(int(i * 100 / len(features)))

    def _split_data(self):
        subwindow = SplitterParameterWindow(self)
        subwindow.show()

    def _data_mining(self, mode='manual'):
        if mode != 'reannotation':
            files = [self._list_of_files.file2path[self._list_of_files.item(i).text()]
                     for i in range(self._list_of_files.count())]
            subwindow = AnnotationParameterWindow(files, mode, self)
            subwindow.show()
        else:
            subwindow = ReAnnotationParameterWindow(self)
            subwindow.show()

    def _data_processing(self, mode):
        if mode == 'simple' and (not os.path.isfile(os.path.join('data', 'weights', 'Classifier.pt'))
                                 or not os.path.isfile(os.path.join('data', 'weights', 'Segmentator.pt'))):
            msg = QtWidgets.QMessageBox(self)
            msg.setText('You should download models in order to process your data:\n'
                        'Data -> Download -> Download trained models')
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.exec_()
        else:
            files = [self._list_of_files.file2path[self._list_of_files.item(i).text()]
                     for i in range(self._list_of_files.count())]
            if not files:
                msg = QtWidgets.QMessageBox(self)
                msg.setText('You should firstly open *.mzML files:\n'
                            'File -> Open -> Open *.mzML')
                msg.setIcon(QtWidgets.QMessageBox.Warning)
                msg.exec_()
            else:
                subwindow = ProcessingParameterWindow(files, mode, self)
                subwindow.show()

    def _open_visualization_window(self):
        files = [self._list_of_files.file2path[self._list_of_files.item(i).text()]
                 for i in range(self._list_of_files.count())]
        subwindow = VisualizationWindow(files, self)
        subwindow.show()

    # Model functionality
    def _model_training(self, mode):
        subwindow = TrainingParameterWindow(mode, self)
        subwindow.show()

    def _model_fine_tuning(self, mode):
        pass

    def _model_evaluation(self, mode):
        subwindow = EvaluationParameterWindow(mode, self)
        subwindow.show()


class FileContextMenu(QtWidgets.QMenu):
    def __init__(self, parent: MainWindow):
        self.parent = parent
        super().__init__(parent)

        menu = QtWidgets.QMenu(parent)

        tic = QtWidgets.QAction('Plot TIC', parent)
        eic = QtWidgets.QAction('Plot EIC', parent)
        close = QtWidgets.QAction('Close', parent)

        menu.addAction(tic)
        menu.addAction(eic)
        menu.addAction(close)

        action = menu.exec_(QtGui.QCursor.pos())

        if action == tic:
            for file in self.parent.get_selected_files():
                file = file.text()
                self.parent.plot_tic(file)
        elif action == eic:
            subwindow = EICParameterWindow(self.parent)
            subwindow.show()
        elif action == close:
            self.close_files()

    def close_files(self):
        for item in self.parent.get_selected_files():
            self.parent.close_file(item)


class FeatureContextMenu(QtWidgets.QMenu):
    def __init__(self, parent: MainWindow):
        self.parent = parent
        super().__init__(parent)
        feature = None
        for item in self.parent.get_selected_features():
            feature = item

        menu = QtWidgets.QMenu(parent)

        with_rt_correction = QtWidgets.QAction('Plot with rt correction', parent)
        without_rt_correction = QtWidgets.QAction('Plot without rt correction', parent)

        menu.addAction(with_rt_correction)
        menu.addAction(without_rt_correction)

        action = menu.exec_(QtGui.QCursor.pos())

        if action == with_rt_correction:
            self.parent.plot_feature(feature, shifted=True)
        elif action == without_rt_correction:
            self.parent.plot_feature(feature, shifted=False)


if __name__ == '__main__':
    plt.switch_backend('Agg')  # to do: check if it is alright???
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
