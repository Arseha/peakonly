import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5 import QtWidgets
from utils.roi import get_ROIs
from gui_utils.auxilary_utils import FileListWidget, GetFolderWidget


class ParameterWindow(QtWidgets.QDialog):
    def __init__(self, files, parent=None):
        super().__init__(parent)

        choose_file_label = QtWidgets.QLabel()
        choose_file_label.setText('Choose file:')
        self.list_of_files = FileListWidget()
        for file in files:
            self.list_of_files.addFile(file)

        save_to_label = QtWidgets.QLabel()
        save_to_label.setText('Choose a folder where to save annotated ROIs:')
        self.folder_widget = GetFolderWidget()

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(choose_file_label)
        layout.addWidget(self.list_of_files)
        layout.addWidget(save_to_label)
        layout.addWidget(self.folder_widget)
        self.setLayout(layout)


class AnnotationWindow(QtWidgets.QDialog):
    def __init__(self, list_of_files, delta_mz, required_points, dropped_points, parent=None):

        self.ROIs = []
        for file in list_of_files:
            print(file)
            self.ROIs.extend(get_ROIs(file, delta_mz, required_points, dropped_points))
        self.current = 0

        super().__init__(parent)

        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that displays the `figure`
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.noise_button = QtWidgets.QPushButton('Noise')
        self.peak_button = QtWidgets.QPushButton('Peak')
        self.uncertain_button = QtWidgets.QPushButton('Uncertain peak')

        # Just some button connected to `plot` method
        self.next_button = QtWidgets.QPushButton('Next')
        self.next_button.clicked.connect(self.plot)

        # set the layout
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.noise_button)
        button_layout.addWidget(self.peak_button)
        button_layout.addWidget(self.uncertain_button)
        button_layout.addWidget(self.next_button)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)


    def plot(self):
        ''' plot some random stuff '''
        # random data
        data = self.ROIs[self.current].i
        self.current += 1
        # instead of ax.hold(False)
        self.figure.clear()
        # create an axis
        ax = self.figure.add_subplot(111)
        # discards the old graph
        # ax.hold(False) # deprecated, see above
        # plot data
        ax.plot(data, '*-')
        # refresh canvas
        self.canvas.draw()
