import os
import torch
import torch.nn as nn
import torch.optim as optim
import shutil
import threading
from torch.utils.data import DataLoader
from PyQt5 import QtWidgets
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from gui_utils.auxilary_utils import GetFolderWidget
from training_utils.dataset import ROIDataset
from training_utils.training import train_model, CombinedLoss, accuracy, iou
from models.rcnn import RecurrentCNN
from models.cnn_classifier import Classifier
from models.cnn_segmentator import Segmentator


class TrainingParameterWindow(QtWidgets.QDialog):
    """
    Training Parameter Window, where one should choose parameters for training

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
    train_folder_getter : GetFolderWidget
        A getter for a path to train data
    val_folder_getter : GetFolderWidget
        A getter for a path to validation data
    """
    def __init__(self, mode, parent=None):
        self.mode = mode
        self.parent = parent
        super().__init__(parent)

        train_folder_label = QtWidgets.QLabel()
        train_folder_label.setText('Choose a folder with train data:')
        self.train_folder_getter = GetFolderWidget()

        val_folder_label = QtWidgets.QLabel()
        val_folder_label.setText('Choose a folder with validation data:')
        self.val_folder_getter = GetFolderWidget()

        continue_button = QtWidgets.QPushButton('Continue')
        continue_button.clicked.connect(self._continue)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(train_folder_label)
        main_layout.addWidget(self.train_folder_getter)
        main_layout.addWidget(val_folder_label)
        main_layout.addWidget(self.val_folder_getter)
        main_layout.addWidget(continue_button)

        self.setLayout(main_layout)

    def _continue(self):
        try:
            train_folder = self.train_folder_getter.get_folder()
            val_folder = self.val_folder_getter.get_folder()
            main_window = TrainingMainWindow(self.mode, train_folder, val_folder, self.parent)
            main_window.show()
            self.close()
        except ValueError:
            pass  # to do: create error window


class TrainingMainWindow(QtWidgets.QDialog):
    """
    Training Main Window, where training process occurs

    Parameters
    ----------
    mode : str
        A one of two 'all in one' of 'sequential'
    train_folder : str
        A path to the folder with training data
    val_folder : str
        A path to the folder with validation data
    parent : MainWindow(QtWidgets.QMainWindow)
        -
    Attributes
    ----------
    mode : str
        A one of two 'all in one' of 'sequential'
    parent : MainWindow(QtWidgets.QMainWindow)
        -

    """
    def __init__(self, mode, train_folder, val_folder, parent):
        self.mode = mode
        self.parent = parent
        super().__init__(parent)
        main_layout = QtWidgets.QVBoxLayout()
        # to do: device should be adjustable parameter
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.mode == 'all in one':
            # create data loaders
            train_dataset = ROIDataset(path=train_folder, device=device, balanced=True)
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
            val_dataset = ROIDataset(path=val_folder, device=device, balanced=False)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
            # create model
            model = RecurrentCNN().to(device)
            optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
            label_criterion = nn.CrossEntropyLoss()
            integration_criterion = nn.BCEWithLogitsLoss()
            intersection_criterion = nn.BCEWithLogitsLoss()
            # add training widget
            main_layout.addWidget(TrainingMainWidget(train_loader, val_loader, model, optimizer, accuracy, iou,
                                                     None, label_criterion, integration_criterion,
                                                     intersection_criterion, 64, self))
        elif self.mode == 'sequential':
            # create data loaders
            batch_size = 64
            train_dataset = ROIDataset(path=train_folder, device=device, interpolate=True, length=256, balanced=True)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataset = ROIDataset(path=val_folder, device=device, interpolate=True, length=256, balanced=False)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            # classifier
            classifier = Classifier().to(device)
            optimizer = optim.Adam(params=classifier.parameters(), lr=1e-3)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
            label_criterion = nn.CrossEntropyLoss()
            main_layout.addWidget(TrainingMainWidget(train_loader, val_loader, classifier, optimizer, accuracy, None,
                                                     scheduler, label_criterion, None, None, 1, self))
            # segmentator
            segmentator = Segmentator().to(device)
            optimizer = optim.Adam(params=segmentator.parameters(), lr=1e-2)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
            integration_criterion = CombinedLoss([0.4, 0.2])
            intersection_criterion = CombinedLoss([0.1, 2])
            main_layout.addWidget(TrainingMainWidget(train_loader, val_loader, segmentator, optimizer, None, iou,
                                                     scheduler, None, integration_criterion,
                                                     intersection_criterion, 1, self))
        self.setLayout(main_layout)


class TrainingMainWidget(QtWidgets.QWidget):
    """
    Training Main Widget, where training process of one model occurs

    Parameters
    ----------
    train_loader : DataLoader
        -
    val_loader : DataLoader
        -
    model : nn.Module
        model to train
    parent : QDialog
        -
    Attributes
    ----------
    parent : MainWindow(QtWidgets.QMainWindow)
        -
   """
    def __init__(self, train_loader, val_loader, model, optimizer, classification_metric, segmenatation_metric,
                 scheduler, label_criterion, integration_criterion, intersection_criterion, accumulation, parent):
        self.parent = parent
        super().__init__(parent)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.optimizer = optimizer
        self.classification_metric = classification_metric
        self.segmentation_metric = segmenatation_metric
        self.scheduler = scheduler
        self.label_criterion = label_criterion
        self.integration_criterion = integration_criterion
        self.intersection_criterion = intersection_criterion
        self.accumulation = accumulation

        self._init_ui()

    def _init_ui(self):
        # canvas layout (with 3 subplots)
        self.figure = plt.figure()
        self.loss_ax = self.figure.add_subplot(131)
        self.loss_ax.set_title('Loss function')
        self.classification_score_ax = self.figure.add_subplot(132)
        self.classification_score_ax.set_title('Classification score')
        self.segmentation_score_ax = self.figure.add_subplot(133)
        self.segmentation_score_ax.set_title('Segmentation score')
        self.figure.tight_layout()
        self.canvas = FigureCanvas(self.figure)
        toolbar = NavigationToolbar(self.canvas, self)
        canvas_layout = QtWidgets.QVBoxLayout()
        canvas_layout.addWidget(toolbar)
        canvas_layout.addWidget(self.canvas)


        # training parameters layout
        parameters_layout = QtWidgets.QVBoxLayout()
        empty_label = QtWidgets.QLabel()

        number_of_epochs_label = QtWidgets.QLabel()
        number_of_epochs_label.setText('Number of epochs:')
        self.number_of_epochs_getter = QtWidgets.QLineEdit(self)
        self.number_of_epochs_getter.setText('100')

        learning_rate_label = QtWidgets.QLabel()
        learning_rate_label.setText('Learning rate:')
        self.learning_rate_getter = QtWidgets.QLineEdit(self)
        self.learning_rate_getter.setText('1e-3')

        parameters_layout.addWidget(empty_label, 80)
        parameters_layout.addWidget(number_of_epochs_label, 5)
        parameters_layout.addWidget(self.number_of_epochs_getter, 5)
        parameters_layout.addWidget(learning_rate_label, 5)
        parameters_layout.addWidget(self.learning_rate_getter, 5)

        # buttons layout
        buttons_layout = QtWidgets.QHBoxLayout()
        restart_button = QtWidgets.QPushButton('Restart')
        restart_button.clicked.connect(self.restart)
        buttons_layout.addWidget(restart_button)
        save_weights_button = QtWidgets.QPushButton('Save weights')
        save_weights_button.clicked.connect(self.save_weights)
        buttons_layout.addWidget(save_weights_button)
        run_training_button = QtWidgets.QPushButton('Run training')
        run_training_button.clicked.connect(self.run_training)
        buttons_layout.addWidget(run_training_button)

        # main layouts
        upper_layout = QtWidgets.QHBoxLayout()
        upper_layout.addLayout(canvas_layout, 85)
        upper_layout.addLayout(parameters_layout, 15)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(upper_layout)
        main_layout.addLayout(buttons_layout)
        self.setLayout(main_layout)

    def restart(self):
        # to do: change restart (problem with optimizer, etc.)
        self.loss_ax.clear()
        self.loss_ax.set_title('Loss function')
        self.classification_score_ax.clear()
        self.classification_score_ax.set_title('Classification score')
        self.segmentation_score_ax.clear()
        self.classification_score_ax.set_title('Segmentation score')
        self.figure.tight_layout()
        self.canvas.draw()
        self.model = self.model.__class__()

    def save_weights(self):
        subwindow = SaveModelWindow(self.model, self)
        subwindow.show()

    def run_training(self):
        try:
            number_of_epoch = int(self.number_of_epochs_getter.text())
            learning_rate = float(self.learning_rate_getter.text())
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate

            t = threading.Thread(target=train_model, args=(self.model, self.train_loader, self.val_loader,
                                                           self.optimizer, number_of_epoch, 10,
                                                           self.classification_metric, self.segmentation_metric,
                                                           self.scheduler, self.label_criterion,
                                                           self.integration_criterion, self.intersection_criterion,
                                                           self.accumulation, self.loss_ax,
                                                           self.classification_score_ax, self.segmentation_score_ax,
                                                           self.figure, self.canvas))
            t.start()
        except ValueError:
            pass  # to do: create error window


class SaveModelWindow(QtWidgets.QDialog):
    def __init__(self, model, parent):
        self.parent = parent
        super().__init__(parent)
        self.model = model

        folder_label = QtWidgets.QLabel()
        folder_label.setText('Choose a folder where to save:')
        self.folder_getter = GetFolderWidget()

        name_label = QtWidgets.QLabel()
        name_label.setText('Set a name of file: ')
        self.name_getter = QtWidgets.QLineEdit(self)
        self.name_getter.setText('model.pt')

        save_button = QtWidgets.QPushButton('Restart')
        save_button.clicked.connect(self.save)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(folder_label)
        main_layout.addWidget(self.folder_getter)
        main_layout.addWidget(name_label)
        main_layout.addWidget(self.name_getter)
        main_layout.addWidget(save_button)

    def save(self):
        folder = self.folder_getter.get_folder()
        name = self.name_getter.text()
        shutil.copyfile(os.path.join('data/tmp_weights', self.model.__class__.__name__),
                        os.path.join(folder, name))
