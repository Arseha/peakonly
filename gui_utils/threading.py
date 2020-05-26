from PyQt5 import QtCore


class WorkerSignals(QtCore.QObject):
    """
    Defines the signals available from a running worker thread.

    Attributes
    ----------
    finished : QtCore.pyqtSignal
        No data
    error : QtCore.pyqtSignal
        `tuple` (exctype, value, traceback.format_exc() )
    result : QtCore.pyqtSignal
        `object` data returned from processing, anything
    progress : QtCore.pyqtSignal
        `int` indicating % progress
    download_progress : QtCore.pyqtSignal
        `int`, `int`, `int` used to show a count of blocks transferred,
        a block size in bytes, the total size of the file
    """
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(tuple)
    result = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int)
    operation = QtCore.pyqtSignal(str)
    download_progress = QtCore.pyqtSignal(int, int, int)


class Worker(QtCore.QRunnable):
    """
    Worker thread

    Parameters
    ----------
    function : callable
        Any callable object

    Attributes
    ----------
    mode : str
        A one of two 'all in one' of 'sequential'
    model : nn.Module
        an ANN model if mode is 'all in one' (optional)
    classifier : nn.Module
        an ANN model for classification (optional)
    segmentator : nn.Module
        an ANN model for segmentation (optional)
    peak_minimum_points : int
        minimum peak length in points

    """
    def __init__(self, function, *args, download=False, multiple_process=False, **kwargs):
        super(Worker, self).__init__()

        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        if not download:
            self.kwargs['progress_callback'] = self.signals.progress
        else:
            self.kwargs['progress_callback'] = self.signals.download_progress

        if multiple_process:
            self.kwargs['operation_callback'] = self.signals.operation

    @QtCore.pyqtSlot()
    def run(self):
        result = self.function(*self.args, **self.kwargs)
        self.signals.result.emit(result)  # return results
        self.signals.finished.emit()  # done
