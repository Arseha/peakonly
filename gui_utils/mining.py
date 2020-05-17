import os
import json
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5 import QtWidgets, QtGui
from processing_utils.roi import get_ROIs, construct_ROI
# from processing_utils.run_utils import classifier_prediction
from gui_utils.abstract_main_window import AbtractMainWindow
from gui_utils.auxilary_utils import FileListWidget, GetFolderWidget, ProgressBarsListItem
from gui_utils.threading import Worker


class ReAnnotationParameterWindow(QtWidgets.QDialog):
    def __init__(self, parent: AbtractMainWindow):
        self.mode = 'reannotation'
        self.parent = parent
        super().__init__(parent)
        self.setWindowTitle('peakonly: reannotation')

        save_to_label = QtWidgets.QLabel()
        save_to_label.setText('Choose a folder with annotated ROIs:')
        self.folder_widget = GetFolderWidget()

        self.run_button = QtWidgets.QPushButton('Run reannotation')
        self.run_button.clicked.connect(self.start_reannotation)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(save_to_label)
        main_layout.addWidget(self.folder_widget)
        main_layout.addWidget(self.run_button)

        self.setLayout(main_layout)

    def start_reannotation(self):
        folder = self.folder_widget.get_folder()
        subwindow = AnnotationMainWindow([], folder, None, None,
                                         None, self.mode,
                                         None, parent=self.parent)
        subwindow.show()
        self.close()


class AnnotationParameterWindow(QtWidgets.QDialog):
    def __init__(self, files, mode, parent: AbtractMainWindow):
        self.mode = mode
        self.parent = parent
        super().__init__(parent)
        self.setWindowTitle('peakonly: manual annotation')

        self.files = files
        self.description = None
        self.file_prefix = None
        self.file_suffix = None
        self.minimum_peak_points = None
        self.folder = None

        self._init_ui()

    def _init_ui(self):
        # file and folder selection
        choose_file_label = QtWidgets.QLabel()
        choose_file_label.setText('Choose a file to annotate:')
        self.list_of_files = FileListWidget()
        for file in self.files:
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

        if self.mode == 'semi-automatic':
            peak_points_label = QtWidgets.QLabel()
            peak_points_label.setText('Minimal length of peak:')
            self.peak_points_getter = QtWidgets.QLineEdit(self)
            self.peak_points_getter.setText('8')

        dropped_points_label = QtWidgets.QLabel()
        dropped_points_label.setText('Maximal number of zero points in a row:')
        self.dropped_points_getter = QtWidgets.QLineEdit(self)
        self.dropped_points_getter.setText('3')

        run_button = QtWidgets.QPushButton('Run annotation')
        run_button.clicked.connect(self._run_button)

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
        # if self.mode == 'semi-automatic':
        #     parameter_layout.addWidget(peak_points_label)
        #     parameter_layout.addWidget(self.peak_points_getter)
        parameter_layout.addWidget(dropped_points_label)
        parameter_layout.addWidget(self.dropped_points_getter)
        parameter_layout.addWidget(run_button)

        # main layout
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(file_layout)
        main_layout.addLayout(parameter_layout)

        self.setLayout(main_layout)

    def _run_button(self):
        try:
            self.description = self.instrumental_getter.text()
            self.file_prefix = self.prefix_getter.text()
            self.file_suffix = int(self.suffix_getter.text())
            delta_mz = float(self.mz_getter.text())
            required_points = int(self.roi_points_getter.text())
            dropped_points = int(self.dropped_points_getter.text())
            if self.mode == 'semi-automatic':
                self.minimum_peak_points = int(self.peak_points_getter.text())

            self.folder = self.folder_widget.get_folder()
            path2mzml = None
            for file in self.list_of_files.selectedItems():
                path2mzml = self.list_of_files.file2path[file.text()]
            if path2mzml is None:
                raise ValueError

            worker = Worker(get_ROIs, path2mzml, delta_mz, required_points, dropped_points)
            worker.signals.result.connect(self._start_annotation)
            self.parent.run_thread('ROI detection:', worker)

            self.close()
        except ValueError:
            # popup window with exception
            msg = QtWidgets.QMessageBox(self)
            msg.setText("Check parameters. Something is wrong!")
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.exec_()

    def _start_annotation(self, rois):
        self.rois = rois
        subwindow = AnnotationMainWindow(self.rois, self.folder, self.file_prefix, self.file_suffix,
                                         self.description, self.mode,
                                         self.minimum_peak_points, parent=self.parent)
        subwindow.show()


class AnnotationMainWindow(QtWidgets.QDialog):
    def __init__(self, ROIs, folder, file_prefix, file_suffix, description, mode,
                 minimum_peak_points, parent=None):
        super().__init__(parent)
        self.setWindowTitle('peakonly: annotation window')
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        self.description = description
        self.current_description = description
        self.folder = folder
        self.mode = mode
        self.plotted_roi = None
        self.plotted_path = None
        self.plotted_item = None  # data reannotation
        self.current_flag = False

        # if self.mode == 'semi-automatic':  # load models
        #     self.minimum_peak_points = minimum_peak_points
        #     self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #     self.classify = Classifier()
        #     self.classify.load_state_dict(torch.load('data/Classifier', map_location=self.device))
        #     self.classify.to(self.device)
        #     self.classify.eval()
        #    self.integrate = Integrator()
        #     self.integrate.load_state_dict(torch.load('data/Integrator', map_location=self.device))
        #     self.integrate.to(self.device)
        #     self.integrate.eval()
        #     # variables where save CNNs predictions
        #     self.label = 0
        #     self.borders = []
        # if self.mode == 'skip noise':
        #     self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #     self.classify = Classifier()
        #     self.classify.load_state_dict(torch.load('data/Classifier', map_location=self.device))
        #     self.classify.to(self.device)
        #     self.classify.eval()
        #     # variables where save CNN predictions
        #     self.label = 0
        # shuffle ROIs
        self.ROIs = ROIs
        np.random.seed(1313)
        np.random.shuffle(self.ROIs)

        self.figure = plt.figure()  # a figure instance to plot on
        self.canvas = FigureCanvas(self.figure)

        self.rois_list = FileListWidget()
        self.rois_list.connectRightClick(self.file_right_click)
        self.rois_list.connectDoubleClick(self.file_double_click)
        files = []
        for created_file in os.listdir(self.folder):
            if created_file.endswith('.json'):
                begin = created_file.find('_') + 1
                end = created_file.find('.json')
                code = int(created_file[begin:end])
                files.append((code, created_file))
        for _, file in sorted(files):
            self.rois_list.addFile(os.path.join(self.folder, file))

        self._init_ui()  # initialize user interface

        self.plot_current()  # initial plot

    def _init_ui(self):
        """
        Initialize all buttons and layouts.
        """
        # canvas layout
        toolbar = NavigationToolbar(self.canvas, self)
        canvas_layout = QtWidgets.QVBoxLayout()
        canvas_layout.addWidget(toolbar)
        canvas_layout.addWidget(self.canvas)

        # canvas and files list layout
        canvas_files_layout = QtWidgets.QHBoxLayout()
        canvas_files_layout.addLayout(canvas_layout, 80)
        canvas_files_layout.addWidget(self.rois_list, 20)

        if self.mode != 'reannotation':
            # plot current button
            plot_current_button = QtWidgets.QPushButton('Plot current ROI')
            plot_current_button.clicked.connect(self.plot_current)
        # noise button
        noise_button = QtWidgets.QPushButton('Noise')
        noise_button.clicked.connect(self.noise)
        # peak button
        peak_button = QtWidgets.QPushButton('Peak')
        peak_button.clicked.connect(self.peak)
        # skip button
        skip_button = QtWidgets.QPushButton('Skip')
        skip_button.clicked.connect(self.skip)
        # plot chosen button
        plot_chosen_button = QtWidgets.QPushButton('Plot chosen ROI')
        plot_chosen_button.clicked.connect(self.press_plot_chosen)


        # button layout
        button_layout = QtWidgets.QHBoxLayout()
        if self.mode != 'reannotation':
            button_layout.addWidget(plot_current_button)
        button_layout.addWidget(noise_button)
        button_layout.addWidget(peak_button)
        button_layout.addWidget(skip_button)
        # if self.mode == 'semi-automatic':
        #     #  agree button
        #     agree_button = QtWidgets.QPushButton('Save CNNs annotation')
        #     agree_button.clicked.connect(self.save_auto_annotation)
        #     button_layout.addWidget(agree_button)
        button_layout.addWidget(plot_chosen_button)

        # main layout
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(canvas_files_layout)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    # Auxiliary methods
    def file_right_click(self):
        FileContextMenu(self)

    def file_double_click(self, item):
        self.plotted_item = item
        self.plot_chosen()

    def get_chosen(self):
        chosen_item = None
        for item in self.rois_list.selectedItems():
            chosen_item = item
        return chosen_item

    def close_file(self, item):
        if item == self.plotted_item:
            index = min(self.rois_list.row(self.plotted_item) + 1, self.rois_list.count() - 2)
            self.plotted_item = self.rois_list.item(index)
            self.plotted_item.setSelected(True)
            self.plot_chosen()
        self.rois_list.deleteFile(item)

    def delete_file(self, item):
        os.remove(self.rois_list.getPath(item))
        self.close_file(item)

    # Buttons
    def noise(self):
        code = os.path.basename(self.plotted_path)
        code = code[:code.rfind('.')]
        label = 0
        self.plotted_roi.save_annotated(self.plotted_path, code, label, description=self.current_description)

        if self.current_flag:
            self.current_flag = False
            self.rois_list.addFile(self.plotted_path)
            self.file_suffix += 1
            self.plot_current()
        else:
            self.plotted_item.setSelected(False)
            index = min(self.rois_list.row(self.plotted_item) + 1, self.rois_list.count() - 1)
            self.plotted_item = self.rois_list.item(index)
            self.plotted_item.setSelected(True)
            self.plot_chosen()

    def peak(self):
        title = 'Annotate peak borders and press "save".'
        subwindow = AnnotationGetNumberOfPeaksNovel(self)
        subwindow.show()

    def skip(self):
        if self.current_flag:
            self.file_suffix += 1
            self.current_flag = False
            self.plot_current()
        else:
            self.plotted_item.setSelected(False)
            index = min(self.rois_list.row(self.plotted_item) + 1, self.rois_list.count() - 1)
            self.plotted_item = self.rois_list.item(index)
            self.plotted_item.setSelected(True)
            self.plot_chosen()

    def save_auto_annotation(self):
        if self.current_flag:
            number_of_peaks = len(self.borders)
            begins = []
            ends = []
            for begin, end in self.borders:
                begins.append(int(begin))
                ends.append(int(end))
            intersections = []
            for i in range(number_of_peaks - 1):
                intersections.append(int(np.argmin(self.plotted_roi.i[ends[i]:begins[i+1]]) + ends[i]))

            code = os.path.basename(self.plotted_path)
            code = code[:code.rfind('.')]
            self.plotted_roi.save_annotated(self.plotted_path, int(self.label), code, number_of_peaks,
                                            begins, ends, intersections, self.description)

            self.current_flag = False
            self.rois_list.addFile(self.plotted_path)
            self.file_suffix += 1
            self.plot_current()

    def press_plot_chosen(self):
        try:
            self.plotted_item = self.get_chosen()
            if self.plotted_item is None:
                raise ValueError
            self.plot_chosen()
        except ValueError:
            # popup window with exception
            msg = QtWidgets.QMessageBox(self)
            msg.setText('Choose a ROI to plot from the list!')
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.exec_()

    # Visualization
    def plot_current(self):
        if self.mode != 'reannotation':
            if not self.current_flag:
                self.current_flag = True
                self.current_description = self.description
                self.plotted_roi = self.ROIs[self.file_suffix]
                # if self.mode == 'skip noise':
                #     self.label = classifier_prediction(self.plotted_roi, self.classify, self.device)
                #     while self.label == 0:
                #         self.file_suffix += 1
                #         self.plotted_roi = self.ROIs[self.file_suffix]
                #         self.label = classifier_prediction(self.plotted_roi, self.classify, self.device)

                filename = f'{self.file_prefix}_{self.file_suffix}.json'
                self.plotted_path = os.path.join(self.folder, filename)

                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.plot(self.plotted_roi.i, label=filename)
                title = f'mz = {self.plotted_roi.mzmean:.3f}, ' \
                        f'rt = {self.plotted_roi.rt[0]:.1f} - {self.plotted_roi.rt[1]:.1f}'

                # if self.mode == 'semi-automatic':  # label and border predictions
                #     self.label = classifier_prediction(self.plotted_roi, self.classify, self.device)
                #     self.borders = []
                #     if self.label != 0:
                #         self.borders = border_prediction(self.plotted_roi, self.integrate,
                #                                          self.device, self.minimum_peak_points)
                #     if self.label == 0:
                #         title = 'label = noise, ' + title
                #     elif self.label == 1:
                #         title = 'label = peak, ' + title
                #     elif self.label == 2:
                #         title = 'label = uncertain peak, ' + title
                #
                #     for begin, end in self.borders:
                #         ax.fill_between(range(begin, end + 1), self.plotted_roi.i[begin:end + 1], alpha=0.5)

                ax.legend(loc='best')
                ax.set_title(title)
                self.canvas.draw()  # refresh canvas

    def plot_chosen(self):
        filename = self.plotted_item.text()
        path2roi = self.rois_list.file2path[filename]
        with open(path2roi) as json_file:
            roi = json.load(json_file)
        self.current_description = roi['description']
        self.plotted_roi = construct_ROI(roi)
        self.plotted_path = path2roi
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(self.plotted_roi.i, label=filename)
        title = f'mz = {self.plotted_roi.mzmean:.3f}, ' \
                f'rt = {self.plotted_roi.rt[0]:.1f} - {self.plotted_roi.rt[1]:.1f}'

        if roi['label'] == 0:
            title = 'label = noise, ' + title
        elif roi['label'] == 1:
            title = 'label = peak, ' + title

        for border, peak_label in zip(roi['borders'], roi["peaks' labels"]):
            begin, end = border
            ax.fill_between(range(begin, end + 1), self.plotted_roi.i[begin:end + 1], alpha=0.5,
                            label=f"pl: {peak_label}, borders={begin}-{end}")

        ax.set_title(title)
        ax.legend(loc='best')
        self.canvas.draw()
        self.current_flag = False

    def plot_preview(self, borders):
        filename = os.path.basename(self.plotted_path)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(self.plotted_roi.i, label=filename)
        title = f'mz = {self.plotted_roi.mzmean:.3f}, ' \
                f'rt = {self.plotted_roi.rt[0]:.1f} - {self.plotted_roi.rt[1]:.1f}'

        for border in borders:
            begin, end = border
            ax.fill_between(range(begin, end + 1), self.plotted_roi.i[begin:end + 1], alpha=0.5)
        ax.set_title(title)
        ax.legend(loc='best')
        self.canvas.draw()  # refresh canvas


class AnnotationGetNumberOfPeaksNovel(QtWidgets.QDialog):
    def __init__(self, parent: AnnotationMainWindow):
        self.parent = parent
        super().__init__(parent)
        self.setWindowTitle('annotation: number of peaks')

        label = QtWidgets.QLabel()
        label.setText('Print number of peaks in current ROI:')

        n_of_peaks_layout = QtWidgets.QHBoxLayout()
        n_of_peaks_label = QtWidgets.QLabel()
        n_of_peaks_label.setText('number of peaks = ')
        self.n_of_peaks_getter = QtWidgets.QLineEdit(self)
        self.n_of_peaks_getter.setText('0')
        n_of_peaks_layout.addWidget(n_of_peaks_label)
        n_of_peaks_layout.addWidget(self.n_of_peaks_getter)

        continue_button = QtWidgets.QPushButton('Continue')
        continue_button.clicked.connect(self.proceed)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(label)
        main_layout.addLayout(n_of_peaks_layout)
        main_layout.addWidget(continue_button)

        self.setLayout(main_layout)

    def proceed(self):
        try:
            number_of_peaks = int(self.n_of_peaks_getter.text())
        except ValueError:
            # popup window with exception
            msg = QtWidgets.QMessageBox(self)
            msg.setText("'Number of peaks' should be an integer value!")
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.exec_()
            return None

        subwindow = AnnotationGetBordersWindowNovel(number_of_peaks, self.parent)
        subwindow.show()
        self.close()


class AnnotationPeakLayoutNovel(QtWidgets.QWidget):
    def __init__(self, peak_number, parent):
        super().__init__(parent)

        borders_layout = QtWidgets.QHBoxLayout()

        label = QtWidgets.QLabel()
        label.setText(f'Peak #{peak_number}')

        begin_label = QtWidgets.QLabel()
        begin_label.setText('begin = ')
        self.begin_getter = QtWidgets.QLineEdit(self)
        end_label = QtWidgets.QLabel()
        end_label.setText('end = ')
        self.end_getter = QtWidgets.QLineEdit(self)
        borders_layout.addWidget(begin_label)
        borders_layout.addWidget(self.begin_getter)
        borders_layout.addWidget(end_label)
        borders_layout.addWidget(self.end_getter)

        peak_label_layout = QtWidgets.QHBoxLayout()
        peak_label_label = QtWidgets.QLabel()
        peak_label_label.setText('peak label = ')
        self.peak_label_getter = QtWidgets.QComboBox(self)
        self.peak_label_getter.addItems(['<None>', 'Good (smooth, high intensive)', 'Low intensive (close to LOD)',
                                         'Lousy (not good)', 'Noisy, strange (probably chemical noise)'])
        peak_label_layout.addWidget(peak_label_label)
        peak_label_layout.addWidget(self.peak_label_getter)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(label)
        main_layout.addLayout(borders_layout)
        main_layout.addLayout(peak_label_layout)

        self.setLayout(main_layout)


class AnnotationGetBordersWindowNovel(QtWidgets.QDialog):
    def __init__(self, number_of_peaks: int, parent: AnnotationMainWindow):
        self.str2label = {'': 0, '<None>': 0, 'Good (smooth, high intensive)': 1,
                          'Low intensive (close to LOD)': 2, 'Lousy (not good)': 3,
                          'Noisy, strange (probably chemical noise)': 4}
        self.number_of_peaks = number_of_peaks
        self.parent = parent
        super().__init__(parent)
        self.setWindowTitle("annotation: peaks' borders")

        main_layout = QtWidgets.QVBoxLayout()
        self.peak_layouts = []
        for i in range(number_of_peaks):
            self.peak_layouts.append(AnnotationPeakLayoutNovel(i + 1, self))
            main_layout.addWidget(self.peak_layouts[-1])

        preview_button = QtWidgets.QPushButton('Preview')
        preview_button.clicked.connect(self.preview)
        main_layout.addWidget(preview_button)

        save_button = QtWidgets.QPushButton('Save')
        save_button.clicked.connect(self.save)
        main_layout.addWidget(save_button)

        self.setLayout(main_layout)

    def preview(self):
        try:
            borders = []
            for pl in self.peak_layouts:
                if pl.begin_getter.text() and pl.end_getter.text():
                    begin = pl.begin_getter.text()
                    end = pl.end_getter.text()
                    if begin and end:
                        begin = int(begin)
                        end = int(end)
                        borders.append((begin, end))
            self.parent.plot_preview(borders)
        except ValueError:
            # popup window with exception
            msg = QtWidgets.QMessageBox(self)
            msg.setText("'begin' and 'end' of each peak should be integer numbers!")
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.exec_()

    def save(self):
        try:
            code = os.path.basename(self.parent.plotted_path)
            code = code[:code.rfind('.')]
            label = 1
            number_of_peaks = self.number_of_peaks
            peaks_labels = []
            borders = []
            for pl in self.peak_layouts:
                peak_label = self.str2label[pl.peak_label_getter.currentText()]
                peaks_labels.append(peak_label)

                begin = int(pl.begin_getter.text())
                end = int(pl.end_getter.text())
                borders.append((begin, end))
        except ValueError:
            # popup window with exception
            msg = QtWidgets.QMessageBox(self)
            msg.setText("Check parameters. Something is wrong!")
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.exec_()

        self.parent.plotted_roi.save_annotated(self.parent.plotted_path, code, label, number_of_peaks,
                                               peaks_labels, borders, description=self.parent.current_description)

        if self.parent.current_flag:
            self.parent.current_flag = False
            self.parent.rois_list.addFile(self.parent.plotted_path)
            self.parent.file_suffix += 1
            self.parent.plot_current()
        else:
            self.parent.plotted_item.setSelected(False)
            index = min(self.parent.rois_list.row(self.parent.plotted_item) + 1, self.parent.rois_list.count() - 1)
            self.parent.plotted_item = self.parent.rois_list.item(index)
            self.parent.plotted_item.setSelected(True)
            self.parent.plot_chosen()
        self.close()


class FileContextMenu(QtWidgets.QMenu):
    def __init__(self, parent: AnnotationMainWindow):
        super().__init__(parent)

        self.parent = parent
        self.menu = QtWidgets.QMenu(parent)

        self.close = QtWidgets.QAction('Close', parent)
        self.delete = QtWidgets.QAction('Delete', parent)

        self.menu.addAction(self.close)
        self.menu.addAction(self.delete)

        action = self.menu.exec_(QtGui.QCursor.pos())

        if action == self.close:
            self.close_file()
        elif action == self.delete:
            self.delete_file()

    def close_file(self):
        item = self.parent.get_chosen()
        self.parent.close_file(item)

    def delete_file(self):
        item = self.parent.get_chosen()
        self.parent.delete_file(item)
