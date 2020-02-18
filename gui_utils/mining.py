import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5 import QtWidgets
from utils.roi import get_ROIs
from gui_utils.auxilary_utils import FileListWidget, GetFolderWidget


class ParameterWindow(QtWidgets.QDialog):
    def __init__(self, files, parent=None):
        self.parent = parent
        super().__init__(parent)

        # file and folder selection
        choose_file_label = QtWidgets.QLabel()
        choose_file_label.setText('Choose file:')
        self.list_of_files = FileListWidget()
        for file in files:
            self.list_of_files.addFile(file)

        save_to_label = QtWidgets.QLabel()
        save_to_label.setText('Choose a folder where to save annotated ROIs:')
        self.folder_widget = GetFolderWidget()

        file_layout = QtWidgets.QVBoxLayout()
        file_layout.addWidget(choose_file_label)
        file_layout.addWidget(self.list_of_files)
        file_layout.addWidget(save_to_label)
        file_layout.addWidget(self.folder_widget)

        # parameters selection

        instrumental_label = QtWidgets.QLabel()
        instrumental_label.setText('Instrumentals description')
        self.instrumental_getter = QtWidgets.QLineEdit(self)
        self.instrumental_getter.setText('Q-oa-TOF, total time=10 min, scan frequency=10Hz')

        prefix_label = QtWidgets.QLabel()
        prefix_label.setText('Prefix of filename: ')
        self.prefix_getter = QtWidgets.QLineEdit(self)
        self.prefix_getter.setText('Example')

        suffix_label = QtWidgets.QLabel()
        suffix_label.setText('Code of file (suffix, will be increased during annotation): ')
        self.suffix_getter = QtWidgets.QLineEdit(self)
        self.suffix_getter.setText('0')

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

        self.run_button = QtWidgets.QPushButton('Run annotation')
        self.run_button.clicked.connect(self.start_annotation)

        parameter_layout = QtWidgets.QVBoxLayout()
        parameter_layout.addWidget(instrumental_label)
        parameter_layout.addWidget(self.instrumental_getter)
        parameter_layout.addWidget(prefix_label)
        parameter_layout.addWidget(self.prefix_getter)
        parameter_layout.addWidget(suffix_label)
        parameter_layout.addWidget(self.suffix_getter)
        parameter_layout.addWidget(mz_label)
        parameter_layout.addWidget(self.mz_getter)
        parameter_layout.addWidget(roi_points_label)
        parameter_layout.addWidget(self.roi_points_getter)
        parameter_layout.addWidget(dropped_points_label)
        parameter_layout.addWidget(self.dropped_points_getter)
        parameter_layout.addWidget(self.run_button)

        # main layout
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(file_layout)
        main_layout.addLayout(parameter_layout)
        self.setLayout(main_layout)

    def start_annotation(self):
        try:
            description = self.instrumental_getter.text()
            file_prefix = self.prefix_getter.text()
            file_suffix = int(self.suffix_getter.text())
            delta_mz = float(self.mz_getter.text())
            required_points = int(self.roi_points_getter.text())
            dropped_points = int(self.dropped_points_getter.text())
            path2mzml = None
            for file in self.list_of_files.selectedItems():
                path2mzml = self.list_of_files.file2path[file.text()]
            if path2mzml is None:
                raise ValueError
            folder = self.folder_widget.get_folder()
            subwindow = AnnotationWindow(path2mzml, delta_mz,
                                         required_points, dropped_points,
                                         folder, file_prefix, file_suffix,
                                         description, parent=self.parent)
            subwindow.show()
            self.close()
        except ValueError:
            pass  # to do: create error window


class AnnotationWindow(QtWidgets.QDialog):
    def __init__(self, file, delta_mz, required_points, dropped_points,
                 folder, file_prefix, file_suffix, description, parent=None):

        self.ROIs = get_ROIs(file, delta_mz, required_points, dropped_points)
        self.current = 0
        self.parent = parent
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
