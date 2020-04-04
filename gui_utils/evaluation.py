import os
import json
import torch
import threading
import numpy as np
from functools import partial
from PyQt5 import QtWidgets
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from gui_utils.auxilary_utils import GetFolderWidget, GetFileWidget, FeatureListWidget
from models.rcnn import RecurrentCNN
from models.cnn_classifier import Classifier
from models.cnn_segmentator import Segmentator
from processing_utils.runner import BasicRunner
from processing_utils.roi import construct_ROI
from processing_utils.run_utils import Feature


class EvaluationParameterWindow(QtWidgets.QDialog):
    """
    Evaluation Parameter Window, where one should choose parameters for evaluation

    Parameters
    ----------
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
    test_folder_getter : GetFolderWidget
        A getter for a path to test data
    model_weights_getter : GetFileWidget
        A getter for a path to weights for 'all-in-one' model (optional)
    classifier_weights_getter : GetFileWidget
        A getter for a path to weights for 'all-in-one' model (optional)
    peak_points_getter : QtWidgets.QLineEdit
        A getter for peak_minimum_points parameter
    segmentator_weights_getter : GetFileWidget
        A getter for a path to weights for 'all-in-one' model (optional)
    """
    def __init__(self, mode, parent=None):
        self.mode = mode
        self.parent = parent
        super().__init__(parent)

        test_folder_label = QtWidgets.QLabel()
        test_folder_label.setText('Choose a folder with test data:')
        self.test_folder_getter = GetFolderWidget()

        if mode == 'all in one':
            model_weights_label = QtWidgets.QLabel()
            model_weights_label.setText("Choose weights for 'all-in-one' model")
            # to do: save a pytorch script, not a model state
            self.model_weights_getter = GetFileWidget('pt', os.path.join(os.getcwd(),
                                                                         'data/weights/RecurrentCNN.pt'), self)
        elif mode == 'sequential':
            classifier_weights_label = QtWidgets.QLabel()
            classifier_weights_label.setText('Choose weights for a classifier')
            # to do: save a pytorch script, not a model state
            self.classifier_weights_getter = GetFileWidget('pt', os.path.join(os.getcwd(),
                                                                              'data/weights/Classifier.pt'), self)
            segmentator_weights_label = QtWidgets.QLabel()
            segmentator_weights_label.setText('Choose weights for a segmentator')
            # to do: save a pytorch script, not a model state
            self.segmentator_weights_getter = GetFileWidget('pt', os.path.join(os.getcwd(),
                                                                              'data/weights/Segmentator.pt'), self)
        else:
            assert False, mode

        peak_points_label = QtWidgets.QLabel()
        peak_points_label.setText('Minimal length of peak:')
        self.peak_points_getter = QtWidgets.QLineEdit(self)
        self.peak_points_getter.setText('8')

        run_button = QtWidgets.QPushButton('Run evaluation')
        run_button.clicked.connect(self._run_evaluation)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(test_folder_label)
        main_layout.addWidget(self.test_folder_getter)
        if mode == 'all in one':
            main_layout.addWidget(model_weights_label)
            main_layout.addWidget(self.model_weights_getter)
        elif mode == 'sequential':
            main_layout.addWidget(classifier_weights_label)
            main_layout.addWidget(self.classifier_weights_getter)
            main_layout.addWidget(segmentator_weights_label)
            main_layout.addWidget(self.segmentator_weights_getter)
        main_layout.addWidget(peak_points_label)
        main_layout.addWidget(self.peak_points_getter)
        main_layout.addWidget(run_button)

        self.setLayout(main_layout)

    def _run_evaluation(self):
        try:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            # to do: device should be customizable parameter
            test_folder = self.test_folder_getter.get_folder()
            if self.mode == 'all in one':
                # to do: save models as pytorch scripts
                model = RecurrentCNN().to(device)
                path2weights = self.model_weights_getter.get_file()
                model.load_state_dict(torch.load(path2weights, map_location=device))
                model.eval()
                models = [model]
            elif self.mode == 'sequential':
                classifier = Classifier().to(device)
                path2classifier_weights = self.classifier_weights_getter.get_file()
                classifier.load_state_dict(torch.load(path2classifier_weights, map_location=device))
                classifier.eval()
                segmentator = Segmentator().to(device)
                path2segmentator_weights = self.segmentator_weights_getter.get_file()
                segmentator.load_state_dict(torch.load(path2segmentator_weights, map_location=device))
                segmentator.eval()
                models = [classifier, segmentator]
            else:
                assert False, self.mode
            minimum_peak_points = int(self.peak_points_getter.text())

            runner = BasicRunner(self.mode, models,
                                 minimum_peak_points, device)

            main_window = EvaluationMainWindow(test_folder, runner, self.parent)
            main_window.show()
            self.close()
        except ValueError:
            pass  # to do: create error window


class EvaluationMainWindow(QtWidgets.QDialog):
    """
    Evaluation Main Window, where one can look into the model quality

    Parameters
    ----------
    test_folder : str
        A path to folder with test data
    runner : BasicRunner
        -
    parent : MainWindow(QtWidgets.QMainWindow)
        -
    Attributes
    ----------
    test_folder : str
        A path to folder with test data
    runner : BasicRunner
        -
    parent : MainWindow(QtWidgets.QMainWindow)
        -
    tp_features : FeatureListWidget
        true positives features
    tn_features : FeatureListWidget
        true negatives features
    fp_features : FeatureListWidget
        false positives features
    fn_features : FeatureListWidget
        false negatives features
    figure : Figure
        -
    ax : Axes
        -
    canvas : FigureCanvasQTAgg
        -
    """
    def __init__(self, test_folder, runner, parent):
        self.parent = parent
        super().__init__(parent)

        self.test_folder = test_folder
        self.runner = runner

        self._init_ui()

    def _init_ui(self):
        # create lists of features
        lists_layout = QtWidgets.QHBoxLayout()

        tp_layout = QtWidgets.QVBoxLayout()
        tp_label = QtWidgets.QLabel()
        tp_label.setText('True positives:')
        tp_layout.addWidget(tp_label)
        self.tp_features = self.create_list_of_features()
        tp_layout.addWidget(self.tp_features)
        tp_next_button = QtWidgets.QPushButton('next')
        tp_next_button.clicked.connect(partial(self.next_feature, self.tp_features))
        tp_layout.addWidget(tp_next_button)
        lists_layout.addLayout(tp_layout)

        tn_layout = QtWidgets.QVBoxLayout()
        tn_label = QtWidgets.QLabel()
        tn_label.setText('True negatives:')
        tn_layout.addWidget(tn_label)
        self.tn_features = self.create_list_of_features()
        tn_layout.addWidget(self.tn_features)
        tn_next_button = QtWidgets.QPushButton('next')
        tn_next_button.clicked.connect(partial(self.next_feature, self.tn_features))
        tn_layout.addWidget(tn_next_button)
        lists_layout.addLayout(tn_layout)

        fp_layout = QtWidgets.QVBoxLayout()
        fp_label = QtWidgets.QLabel()
        fp_label.setText('False positives:')
        fp_layout.addWidget(fp_label)
        self.fp_features = self.create_list_of_features()
        fp_layout.addWidget(self.fp_features)
        fp_next_button = QtWidgets.QPushButton('next')
        fp_next_button.clicked.connect(partial(self.next_feature, self.fp_features))
        fp_layout.addWidget(fp_next_button)
        lists_layout.addLayout(fp_layout)

        fn_layout = QtWidgets.QVBoxLayout()
        fn_label = QtWidgets.QLabel()
        fn_label.setText('False negatives:')
        fn_layout.addWidget(fn_label)
        self.fn_features = self.create_list_of_features()
        fn_layout.addWidget(self.fn_features)
        fn_next_button = QtWidgets.QPushButton('next')
        fn_next_button.clicked.connect(partial(self.next_feature, self.fn_features))
        fn_layout.addWidget(fn_next_button)
        lists_layout.addLayout(fn_layout)

        # statistic button
        right_half_layout = QtWidgets.QVBoxLayout()
        right_half_layout.addLayout(lists_layout)

        statistics_button = QtWidgets.QPushButton('get more statistics')
        statistics_button.clicked.connect(self.get_statistics)
        right_half_layout.addWidget(statistics_button)

        # Main canvas and toolbar
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)  # plot here
        self.canvas = FigureCanvas(self.figure)
        toolbar = NavigationToolbar(self.canvas, self)
        canvas_layout = QtWidgets.QVBoxLayout()
        canvas_layout.addWidget(toolbar)
        canvas_layout.addWidget(self.canvas)

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(canvas_layout, 60)
        main_layout.addLayout(right_half_layout, 40)
        self.setLayout(main_layout)

        thread = threading.Thread(target=self.update)
        thread.start()

    def create_list_of_features(self):
        list_of_features = FeatureListWidget()
        list_of_features.itemDoubleClicked.connect(self.feature_click)
        return list_of_features

    def feature_click(self, item):
        list_widget = item.listWidget()
        feature = list_widget.get_feature(item)
        self.plot_feature(feature)

    def next_feature(self, list_widget):
        raw = list_widget.currentRow()
        item = list_widget.item(min(raw + 1, list_widget.count() - 1))
        list_widget.setCurrentItem(item)
        feature = list_widget.get_feature(item)
        self.plot_feature(feature)

    def update(self):
        for file in os.listdir(self.test_folder):
            if file[0] != '.':
                with open(os.path.join(self.test_folder, file)) as json_file:
                    dict_roi = json.load(json_file)
                # get predicted features
                roi = construct_ROI(dict_roi)
                features = self.runner(roi, 'predicted/' + file)
                # append gt (ground truth) features
                for border in dict_roi['borders']:
                    gt = np.zeros(len(roi.i), dtype=np.bool)
                    gt[border[0]:border[1]+1] = 1
                    scan_frequency = (roi.scan[1] - roi.scan[0]) / (roi.rt[1] - roi.rt[0])
                    rtmin = roi.rt[0] + border[0] / scan_frequency
                    rtmax = roi.rt[0] + border[1] / scan_frequency
                    match = False
                    for feature in features:
                        if len(feature) == 1 and feature.samples[0][:2] == 'pr':
                            predicted_border = feature.borders[0]
                            pred = np.zeros(len(roi.i), dtype=np.bool)
                            pred[predicted_border[0]:predicted_border[1]+1] = 1
                            # calculate iou
                            intersection = (pred & gt).sum()  # will be zero if Truth=0 or Prediction=0
                            union = (pred | gt).sum()
                            if intersection / union > 0.5:
                                match = True
                                feature.append('gt/' + file, roi, border, 0, np.sum(roi.i[border[0]:border[1]]),
                                               roi.mzmean, rtmin, rtmax)
                                break
                    if not match:
                        features.append(Feature(['gt/' + file], [roi], [border], [0], [np.sum(roi.i[border[0]:border[1]])],
                                                roi.mzmean, rtmin, rtmax, 0, 0))

                # append tp, tn, fp, fn
                for feature in features:
                    if len(feature) == 2:
                        self.tp_features.add_feature(feature)
                    elif len(feature) == 1 and feature.samples[0][:2] == 'pr':
                        self.fp_features.add_feature(feature)
                    elif len(feature) == 1 and feature.samples[0][:2] == 'gt':
                        self.fn_features.add_feature(feature)
                    else:
                        print(len(feature)), print(feature.samples[0][:2])
                        assert False, feature.samples

                if len(features) == 0:
                    noise_feature = Feature(['noise/' + file], [roi], [[0, 0]], [0], [0],
                                            roi.mzmean, roi.rt[0], roi.rt[1], 0, 0)
                    self.tn_features.add_feature(noise_feature)

    def plot_feature(self, feature):
        self.ax.clear()
        feature.plot(self.ax, shifted=False, show_legend=True)
        self.canvas.draw()  # refresh canvas

    def get_statistics(self):
        # to do: create a window with stats
        tp_features = self.tp_features.get_all()
        tn_features = self.tn_features.get_all()
        fp_features = self.fp_features.get_all()
        fn_features = self.fn_features.get_all()
        subwindow = EvaluationStatisticsWindow(tp_features, tn_features, fp_features, fn_features, self)
        subwindow.show()
        pass


class EvaluationStatisticsWindow(QtWidgets.QDialog):
    def __init__(self, tp_features, tn_features, fp_features, fn_features, parent):
        self.parent = parent
        super().__init__(parent)
        # auxiliary calculations
        precision = len(tp_features) / (len(tp_features) + len(fp_features))
        recall = len(tp_features) / (len(tp_features) + len(fn_features))

        integration_accuracy = np.zeros(len(tp_features))
        for i, feature in enumerate(tp_features):
            integration_accuracy[i] = np.abs(feature.intensities[0] - feature.intensities[1]) / feature.intensities[1]
        integration_accuracy = 1 - np.mean(integration_accuracy)

        # print metrics
        precision_label = QtWidgets.QLabel()
        precision_label.setText(f'Precision = {precision:.2f}')
        recall_label = QtWidgets.QLabel()
        recall_label.setText(f'Recall = {recall:.2f}')
        integration_accuracy_label = QtWidgets.QLabel()
        integration_accuracy_label.setText(f'Integration accuracy = {integration_accuracy:.2f}')

        # canvas for confusion matrix
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)  # plot here
        self.canvas = FigureCanvas(self.figure)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(self.canvas)
        main_layout.addWidget(precision_label)
        main_layout.addWidget(recall_label)
        main_layout.addWidget(integration_accuracy_label)

        self.setLayout(main_layout)

        self.plot_confusion_matrix(len(tp_features), len(tn_features), len(fp_features), len(fn_features))

    def plot_confusion_matrix(self, tp, tn, fp, fn):
        confusion_matrix = np.zeros((2, 2), np.int)
        confusion_matrix[0, 0] = tp
        confusion_matrix[0, 1] = fp
        confusion_matrix[1, 0] = fn
        confusion_matrix[1, 1] = tn

        self.ax.set_title("Confusion matrix")
        res = self.ax.imshow(confusion_matrix, cmap='GnBu', interpolation='nearest')
        self.figure.colorbar(res)
        self.ax.set_xticks(np.arange(2))
        self.ax.set_xticklabels(['peak', 'noise'])
        self.ax.set_yticks(np.arange(2))
        self.ax.set_yticklabels(['peak', 'noise'])
        self.ax.set_ylabel("predicted")
        self.ax.set_xlabel("ground truth")
        for i, row in enumerate(confusion_matrix):
            for j, count in enumerate(row):
                plt.text(j, i, count, fontsize=14, horizontalalignment='center', verticalalignment='center')

        self.canvas.draw()
