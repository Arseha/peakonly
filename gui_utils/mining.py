import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5 import QtWidgets
from utils.roi import get_ROIs, construct_ROI
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
            subwindow = AnnotationMainWindow(path2mzml, delta_mz,
                                         required_points, dropped_points,
                                         folder, file_prefix, file_suffix,
                                         description, parent=self.parent)
            subwindow.show()
            self.close()
        except ValueError:
            pass  # to do: create error window


class AnnotationMainWindow(QtWidgets.QDialog):
    def __init__(self, file, delta_mz, required_points, dropped_points,
                 folder, file_prefix, file_suffix, description, parent=None):
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        self.description = description
        self.folder = folder
        self.plotted_roi = None
        self.plotted_path = None
        self.current_flag = False

        self.ROIs = get_ROIs(file, delta_mz, required_points, dropped_points)
        np.random.seed(1313)
        np.random.shuffle(self.ROIs)  # shuffle ROIs
        self.parent = parent
        super().__init__(parent)

        self.figure = plt.figure()  # a figure instance to plot on
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        # canvas layout
        canvas_layout = QtWidgets.QVBoxLayout()
        canvas_layout.addWidget(self.toolbar)
        canvas_layout.addWidget(self.canvas)

        self.rois_list = FileListWidget()
        for created_file in os.listdir(self.folder):
            if created_file.endswith('.json'):
                self.rois_list.addFile(os.path.join(self.folder, created_file))

        # canvas and files list layout
        canvas_files_layout = QtWidgets.QHBoxLayout()
        canvas_files_layout.addLayout(canvas_layout)
        canvas_files_layout.addWidget(self.rois_list)

        self.plot_current_button = QtWidgets.QPushButton('Plot current ROI')
        self.plot_current_button.clicked.connect(self.plot_current)
        self.noise_button = QtWidgets.QPushButton('Noise')
        self.noise_button.clicked.connect(self.noise)
        self.peak_button = QtWidgets.QPushButton('Peak')
        self.peak_button.clicked.connect(self.peak)
        self.uncertain_button = QtWidgets.QPushButton('Uncertain peak')
        self.uncertain_button.clicked.connect(self.uncertain)
        # Just some button connected to `plot` method
        self.skip_button = QtWidgets.QPushButton('Skip')
        self.skip_button.clicked.connect(self.skip)
        self.plot_chosen_button = QtWidgets.QPushButton('Plot chosen ROI')
        self.plot_chosen_button.clicked.connect(self.plot_chosen)

        # button layout
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.plot_current_button)
        button_layout.addWidget(self.noise_button)
        button_layout.addWidget(self.peak_button)
        button_layout.addWidget(self.uncertain_button)
        button_layout.addWidget(self.skip_button)
        button_layout.addWidget(self.plot_chosen_button)

        # main layout
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(canvas_files_layout)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

        self.plot_current()  # initial plot

    def skip(self):
        self.file_suffix += 1
        self.current_flag = False
        self.plot_current()

    def plot_current(self):
        if not self.current_flag:
            self.current_flag = True
            # plot current ROI
            self.plotted_roi = self.ROIs[self.file_suffix]
            filename = f'{self.file_prefix}{self.file_suffix}.json'
            self.plotted_path = os.path.join(self.folder, filename)
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(self.plotted_roi.i)
            ax.set_title(filename)
            self.canvas.draw()  # refresh canvas

    def plot_chosen(self):
        try:
            path2roi = None
            for file in self.rois_list.selectedItems():
                filename = file.text()
                path2roi = self.rois_list.file2path[filename]
            if path2roi is None:
                raise ValueError
            with open(path2roi) as json_file:
                roi = json.load(json_file)

            self.plotted_roi = construct_ROI(roi)
            self.plotted_path = path2roi
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(self.plotted_roi.i)
            for begin, end in zip(roi['begins'], roi['ends']):
                ax.fill_between(range(begin, end+1), self.plotted_roi.i[begin:end+1], alpha=0.5)
            ax.set_title(filename)
            self.canvas.draw()  # refresh canvas
            self.current_flag = False
        except ValueError:
            pass  # to do: create error window

    def noise(self):
        code = os.path.basename(self.plotted_path)
        code = code[:code.rfind('.')]
        self.plotted_roi.save_annotated(self.plotted_path,
                                        0, code, description=self.description)
        if self.current_flag:
            self.parent.current_flag = False
            self.rois_list.addFile(self.plotted_path)
            self.file_suffix += 1
            self.plot_current()
        else:
            self.plot_current()

    def uncertain(self):
        title = 'Try to pick out peaks, if possible, otherwise just press "save".'
        subwindow = GetBordersWindow(title, 2, self)
        subwindow.show()

    def peak(self):
        title = 'Annotate peak borders and press "save".'
        subwindow = GetBordersWindow(title, 1, self)
        subwindow.show()


class GetBordersWindow(QtWidgets.QDialog):
    def __init__(self, title: str, label: int, parent: AnnotationMainWindow):
        self.parent = parent
        self.label = label  # ROIs class
        super().__init__(parent)

        label = QtWidgets.QLabel()
        label.setText(title)

        n_of_peaks_layout = QtWidgets.QHBoxLayout()
        n_of_peaks_label = QtWidgets.QLabel()
        n_of_peaks_label.setText('number of peaks = ')
        self.n_of_peaks_getter = QtWidgets.QLineEdit(self)
        self.n_of_peaks_getter.setText('0')
        n_of_peaks_layout.addWidget(n_of_peaks_label)
        n_of_peaks_layout.addWidget(self.n_of_peaks_getter)

        begins_layout = QtWidgets.QHBoxLayout()
        begins_label = QtWidgets.QLabel()
        begins_label.setText('begins = ')
        self.begins_getter = QtWidgets.QLineEdit(self)
        begins_layout.addWidget(begins_label)
        begins_layout.addWidget(self.begins_getter)

        ends_layout = QtWidgets.QHBoxLayout()
        ends_label = QtWidgets.QLabel()
        ends_label.setText('ends = ')
        self.ends_getter = QtWidgets.QLineEdit(self)
        ends_layout.addWidget(ends_label)
        ends_layout.addWidget(self.ends_getter)

        intersections_layout = QtWidgets.QHBoxLayout()
        intersections_label = QtWidgets.QLabel()
        intersections_label.setText('intersections = ')
        self.intersections_getter = QtWidgets.QLineEdit(self)
        intersections_layout.addWidget(intersections_label)
        intersections_layout.addWidget(self.intersections_getter)

        save_button = QtWidgets.QPushButton('Save')
        save_button.clicked.connect(self.save_roi)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(label)
        main_layout.addLayout(n_of_peaks_layout)
        main_layout.addLayout(begins_layout)
        main_layout.addLayout(ends_layout)
        main_layout.addLayout(intersections_layout)
        main_layout.addWidget(save_button)

        self.setLayout(main_layout)

    def save_roi(self):
        try:
            number_of_peaks = int(self.n_of_peaks_getter.text())
            begins = []
            for begin in re.split('[ ;.,]+', self.begins_getter.text()):
                if begin:
                    begins.append(int(begin))
            ends = []
            for end in re.split('[ ;.,]+', self.ends_getter.text()):
                if end:
                    ends.append(int(end))
            intersections = []
            for intersection in re.split('[ ;.,]+', self.intersections_getter.text()):
                if intersection:
                    intersections.append(int(intersection))
            if number_of_peaks != len(begins) or number_of_peaks != len(ends) or \
                    number_of_peaks - 1 != len(intersections):
                raise ValueError
        except ValueError:
            return  # to do: create error window

        code = os.path.basename(self.parent.plotted_path)
        code = code[:code.rfind('.')]
        self.parent.plotted_roi.save_annotated(self.parent.plotted_path, self.label, code, number_of_peaks,
                                               begins, ends, intersections, self.parent.description)
        if self.parent.current_flag:
            self.parent.current_flag = False
            self.parent.rois_list.addFile(self.parent.plotted_path)
            self.parent.file_suffix += 1
            self.parent.plot_current()
        else:
            self.parent.plot_current()
        self.close()
