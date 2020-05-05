import os
import torch
from PyQt5 import QtWidgets
from gui_utils.abstract_main_window import AbtractMainWindow
from gui_utils.auxilary_utils import FileListWidget, GetFileWidget
from gui_utils.threading import Worker
from processing_utils.runner import FilesRunner
from models.rcnn import RecurrentCNN
from models.cnn_classifier import Classifier
from models.cnn_segmentator import Segmentator


class ProcessingParameterWindow(QtWidgets.QDialog):
    """
    Main Processing Window, where one can choose files and parameters for data processing

    Parameters
    ----------
    files : list of str
        A list of *.mzML files
    mode : str
        A one of two 'all in one' of 'sequential'
    parent : MainWindow(QtWidgets.QMainWindow)
        -
    Attributes
    ----------
    mode : str
        A one of two 'all in one' of 'sequential'
    parent : MainWindow(QtWidgets.QMainWindow)
        -
    list_of_files : FileListWidget
        QtWidget which stores and shows *.mzML files
    weights_widget : GetFileWidget
        Stores a path for 'All in one' ANN (optional attribute)
    weights_classifier_widget : GetFileWidget
        Stores a path for a classifier (optional attribute)
    weights_segmentator_widget : GetFileWidget
        Stores a path for a segmentator (optional attribute)
    mz_getter : QtWidgets.QLineEdit
        A getter for delta_mz parameter
    roi_points_getter : QtWidgets.QLineEdit
        A getter for required_points parameter
    dropped_points_getter : QtWidgets.QLineEdit
        A getter for dropped_points parameter
    peak_points_getter : QtWidgets.QLineEdit
        A getter for peak_minimum_points parameter
    """
    def __init__(self, files, mode, parent: AbtractMainWindow):
        self.parent = parent
        self.mode = mode
        super().__init__(parent)
        self._init_ui(files)  # initialize user interface

    def _init_ui(self, files):
        # files selection
        choose_file_label = QtWidgets.QLabel()
        choose_file_label.setText('Choose files to process:')
        self.list_of_files = FileListWidget()
        for file in files:
            self.list_of_files.addFile(file)
        for i in range(self.list_of_files.count()):
            self.list_of_files.item(i).setSelected(True)
        self.list_of_files.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        # left 'half' layout
        left_half_layout = QtWidgets.QVBoxLayout()
        left_half_layout.addWidget(choose_file_label)
        left_half_layout.addWidget(self.list_of_files)

        # ANN's weights selection
        weights_layout = QtWidgets.QVBoxLayout()
        if self.mode == 'all in one':
            choose_weights_label = QtWidgets.QLabel()
            choose_weights_label.setText("Choose weights for a 'all in one' model:")
            self.weights_widget = GetFileWidget('pt', os.path.join(os.getcwd(), 'data', 'weights', 'RecurrentCNN.pt'),
                                                self.parent)
            weights_layout.addWidget(choose_weights_label)
            weights_layout.addWidget(self.weights_widget)
        elif self.mode == 'sequential':
            choose_classifier_weights_label = QtWidgets.QLabel()
            choose_classifier_weights_label.setText('Choose weights for a Classifier:')
            self.weights_classifier_widget = GetFileWidget('pt', os.path.join(os.getcwd(),
                                                                              'data', 'weights', 'Classifier.pt'),
                                                           self.parent)
            choose_segmentator_weights_label = QtWidgets.QLabel()
            choose_segmentator_weights_label.setText('Choose weights for a Segmentator:')
            self.weights_segmentator_widget = GetFileWidget('pt', os.path.join(os.getcwd(),
                                                                               'data', 'weights', 'Segmentator.pt'),
                                                            self.parent)
            weights_layout.addWidget(choose_classifier_weights_label)
            weights_layout.addWidget(self.weights_classifier_widget)
            weights_layout.addWidget(choose_segmentator_weights_label)
            weights_layout.addWidget(self.weights_segmentator_widget)

        # Selection of parameters
        parameters_layout = QtWidgets.QVBoxLayout()

        mz_label = QtWidgets.QLabel()
        mz_label.setText('m/z deviation:')
        self.mz_getter = QtWidgets.QLineEdit(self)
        self.mz_getter.setText('0.005')

        roi_points_label = QtWidgets.QLabel()
        roi_points_label.setText('Minimal length of ROI:')
        self.roi_points_getter = QtWidgets.QLineEdit(self)
        self.roi_points_getter.setText('15')

        dropped_points_label = QtWidgets.QLabel()
        dropped_points_label.setText('Maximal number of zero points in a row:')
        self.dropped_points_getter = QtWidgets.QLineEdit(self)
        self.dropped_points_getter.setText('3')

        peak_points_label = QtWidgets.QLabel()
        peak_points_label.setText('Minimal length of peak:')
        self.peak_points_getter = QtWidgets.QLineEdit(self)
        self.peak_points_getter.setText('8')

        parameters_layout.addWidget(mz_label)
        parameters_layout.addWidget(self.mz_getter)
        parameters_layout.addWidget(roi_points_label)
        parameters_layout.addWidget(self.roi_points_getter)
        parameters_layout.addWidget(dropped_points_label)
        parameters_layout.addWidget(self.dropped_points_getter)
        parameters_layout.addWidget(peak_points_label)
        parameters_layout.addWidget(self.peak_points_getter)

        # run button
        run_button = QtWidgets.QPushButton('Run processing')
        run_button.clicked.connect(self.start_processing)

        # right 'half' layout
        right_half_layout = QtWidgets.QVBoxLayout()
        right_half_layout.addLayout(weights_layout)
        right_half_layout.addLayout(parameters_layout)
        right_half_layout.addWidget(run_button)

        # main layout
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(left_half_layout, 30)
        main_layout.addLayout(right_half_layout, 70)
        self.setLayout(main_layout)

    def start_processing(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # to do: device should be customizable parameter
        try:
            delta_mz = float(self.mz_getter.text())
            required_points = int(self.roi_points_getter.text())
            dropped_points = int(self.dropped_points_getter.text())
            minimum_peak_points = int(self.peak_points_getter.text())
            if self.mode == 'all in one':
                # to do: save models as pytorch scripts
                model = RecurrentCNN().to(device)
                path2weights = self.weights_widget.get_file()
                model.load_state_dict(torch.load(path2weights, map_location=device))
                model.eval()
                models = [model]
            elif self.mode == 'sequential':
                classifier = Classifier().to(device)
                path2classifier_weights = self.weights_classifier_widget.get_file()
                classifier.load_state_dict(torch.load(path2classifier_weights, map_location=device))
                classifier.eval()
                segmentator = Segmentator().to(device)
                path2segmentator_weights = self.weights_segmentator_widget.get_file()
                segmentator.load_state_dict(torch.load(path2segmentator_weights, map_location=device))
                segmentator.eval()
                models = [classifier, segmentator]
            elif self.mode == 'simple':
                self.mode = 'sequential'
                classifier = Classifier().to(device)
                path2classifier_weights = os.path.join('data', 'weights', 'Classifier.pt')
                classifier.load_state_dict(torch.load(path2classifier_weights, map_location=device))
                classifier.eval()
                segmentator = Segmentator().to(device)
                path2segmentator_weights = os.path.join('data', 'weights', 'Segmentator.pt')
                segmentator.load_state_dict(torch.load(path2segmentator_weights, map_location=device))
                segmentator.eval()
                models = [classifier, segmentator]
            else:
                assert False, self.mode

            path2mzml = []
            for file in self.list_of_files.selectedItems():
                path2mzml.append(self.list_of_files.file2path[file.text()])
            if not path2mzml:
                raise ValueError

            runner = FilesRunner(self.mode, models, delta_mz,
                                 required_points, dropped_points,
                                 minimum_peak_points, device)

            worker = Worker(runner, path2mzml)
            worker.signals.result.connect(self.parent.set_features)
            self.parent.run_thread('Data processing:', worker)

            self.close()
        except ValueError:
            # popup window with exception
            msg = QtWidgets.QMessageBox(self)
            msg.setText("Check parameters. Something is wrong!")
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.exec_()
