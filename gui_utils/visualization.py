from functools import partial
from PyQt5 import QtWidgets, QtGui
from gui_utils.abstract_main_window import AbtractMainWindow
from gui_utils.auxilary_utils import ClickableListWidget, FileListWidget


class EICParameterWindow(QtWidgets.QDialog):
    def __init__(self, parent: AbtractMainWindow):
        self.parent = parent
        super().__init__(self.parent)
        self.setWindowTitle('peakonly: plot EIC')

        mz_layout = QtWidgets.QHBoxLayout()
        mz_label = QtWidgets.QLabel(self)
        mz_label.setText('m/z=')
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

        plot_button = QtWidgets.QPushButton('Plot')
        plot_button.clicked.connect(self.plot)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(mz_layout)
        layout.addLayout(delta_layout)
        layout.addWidget(plot_button)
        self.setLayout(layout)

    def plot(self):
        try:
            mz = float(self.mz_getter.text())
            delta = float(self.delta_getter.text())
            for file in self.parent.get_selected_files():
                file = file.text()
                self.parent.plot_eic(file, mz, delta)
            self.close()
        except ValueError:
            # popup window with exception
            msg = QtWidgets.QMessageBox(self)
            msg.setText("'m/z' and 'delta' should be float numbers!")
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.exec_()


class VisualizationWindow(QtWidgets.QDialog):
    def __init__(self, files, parent: AbtractMainWindow):
        self.parent = parent
        super().__init__(self.parent)
        self.setWindowTitle('peakonly: visualization')

        # files selection
        files_layout = QtWidgets.QVBoxLayout()
        choose_file_label = QtWidgets.QLabel()
        choose_file_label.setText('Choose files to visualize:')
        self._list_of_files = FileListWidget()
        for file in files:
            self._list_of_files.addFile(file)
        self._list_of_files.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        files_layout.addWidget(choose_file_label)
        files_layout.addWidget(self._list_of_files)

        # plotted mode
        self.plotted_mode_getter = QtWidgets.QComboBox(self)
        self.plotted_mode_getter.addItems(['Total Ion Chromatogram (TIC)', 'Extracted Ion Chromatogram (EIC)'])

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

        plot_button = QtWidgets.QPushButton('Plot')
        plot_button.clicked.connect(self._plot)

        files_layout.addWidget(self.plotted_mode_getter)
        files_layout.addLayout(mz_layout)
        files_layout.addLayout(delta_layout)
        files_layout.addWidget(plot_button)

        # list of lines
        plotted_lines_layout = QtWidgets.QVBoxLayout()
        plotted_label = QtWidgets.QLabel()
        plotted_label.setText('Currently plotted: ')
        self._plotted_list = ClickableListWidget()
        self._plotted_list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
        for label in self.parent.get_plotted_lines():
            self._plotted_list.addItem(label)
        self._plotted_list.connectRightClick(partial(LineContextMenu, self))
        plotted_lines_layout.addWidget(plotted_label)
        plotted_lines_layout.addWidget(self._plotted_list)

        # main layout
        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(files_layout)
        layout.addLayout(plotted_lines_layout)
        self.setLayout(layout)

    def get_selected_lines(self):
        return self._plotted_list.selectedItems()

    def delete_selected(self):
        for item in self._plotted_list.selectedItems():
            self.parent.delete_line(item.text())
            self._plotted_list.takeItem(self._plotted_list.row(item))  # delete item from list
        self.parent.refresh_canvas()

    def _plot(self):
        mode = self.plotted_mode_getter.currentText()
        if mode == 'Total Ion Chromatogram (TIC)':
            for file in self._list_of_files.selectedItems():
                file = file.text()
                plotted, label = self.parent.plot_tic(file)
                if plotted:
                    self._plotted_list.addItem(label)
        elif mode == 'Extracted Ion Chromatogram (EIC)':
            try:
                mz = float(self.mz_getter.text())
                delta = float(self.delta_getter.text())
                for file in self._list_of_files.selectedItems():
                    file = file.text()
                    plotted, label = self.parent.plot_eic(file, mz, delta)
                    if plotted:
                        self._plotted_list.addItem(label)
            except ValueError:
                # popup window with exception
                msg = QtWidgets.QMessageBox(self)
                msg.setText("'mz' and 'delta' should be float numbers!")
                msg.setIcon(QtWidgets.QMessageBox.Warning)
                msg.exec_()


class LineContextMenu(QtWidgets.QMenu):
    def __init__(self, parent: VisualizationWindow):
        self.parent = parent
        super().__init__(parent)
        lines = list(self.parent.get_selected_lines())

        menu = QtWidgets.QMenu(parent)

        clear = QtWidgets.QAction('Clear', parent)

        menu.addAction(clear)

        action = menu.exec_(QtGui.QCursor.pos())

        if action == clear:
            self.parent.delete_selected()
