import os
import numpy as np
try:
    from cython_utils.roi import get_ROIs
except ImportError:
    from processing_utils.roi import get_ROIs
from processing_utils.matching import construct_mzregions, rt_grouping, align_component
from processing_utils.run_utils import preprocess, get_borders, Feature, \
    border_correction, build_features, feature_collapsing


class BasicRunner:
    """
    A runner to process single roi

    Parameters
    ----------
    mode : str
        A one of two 'all in one' of 'sequential'
    models : list
        a list of models
    peak_minimum_points : int
        -

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
    def __init__(self, mode, models, peak_minimum_points, device):
        self.mode = mode
        if self.mode == 'all in one':
            self.model = models[0]
        elif self.mode == 'sequential':
            self.classifier, self.segmentator = models
        else:
            assert False, mode
        self.peak_minimum_points = peak_minimum_points
        self.device = device

    def __call__(self, roi, sample_name, progress_callback=None, operation_callback=None):
        """
        Processing single roi

        Parameters
        ----------
        roi : ROI
            -
        sample_name : str
            Arbitrary sample name
        Returns
        -------
        feature : list
            a list of  'Feature' objects
        """
        if self.mode == 'all in one':
            signal = preprocess(roi.i, self.device)
            classifier_output, segmentator_output = self.model(signal)
        elif self.mode == 'sequential':
            signal = preprocess(roi.i, self.device, interpolate=True, length=256)
            classifier_output, _ = self.classifier(signal)
            # to do: second step should be only for peaks
            _, segmentator_output = self.segmentator(signal)
        else:
            assert False, self.mode
        classifier_output = classifier_output.data.cpu().numpy()
        segmentator_output = segmentator_output.data.sigmoid().cpu().numpy()

        # get label
        label = np.argmax(classifier_output)
        # get borders
        features = []
        if label == 1:
            borders = get_borders(segmentator_output[0, 0, :], segmentator_output[0, 1, :],
                                  peak_minimum_points=self.peak_minimum_points,
                                  interpolation_factor=len(signal[0, 0]) / len(roi.i))
            for border in borders:
                # to do: check correctness of rt calculations
                scan_frequency = (roi.scan[1] - roi.scan[0]) / (roi.rt[1] - roi.rt[0])
                rtmin = roi.rt[0] + border[0] / scan_frequency
                rtmax = roi.rt[0] + border[1] / scan_frequency
                feature = Feature([sample_name], [roi], [border], [0], [np.sum(roi.i[border[0]:border[1]])],
                                  roi.mzmean, rtmin, rtmax, 0, 0)
                features.append(feature)
        return features


class FilesRunner(BasicRunner):
    """
    A runner to process *.mzML files

    Parameters
    ----------
    mode : str
        A one of two 'all in one' of 'sequential'
    models : list
        a list of models
    delta_mz : float
        -
    required_points : int
        -
    peak_minimum_points : int
        -

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
    delta_mz : float
        a parameters for mz window in ROI detection
    required_points : int
        minimum ROI length in points
    dropped_points : int
        maximal number of zero points in a row (for ROI detection)
    peak_minimum_points : int
        minimum peak length in points

    """
    def __init__(self, mode, models, delta_mz,
                 required_points, dropped_points,
                 peak_minimum_points, device):
        super(FilesRunner, self).__init__(mode, models, peak_minimum_points, device)
        self.delta_mz = delta_mz
        self.required_points = required_points
        self.dropped_points = dropped_points

    def __call__(self, files, progress_callback=None, operation_callback=None):
        if len(files) == 1:
            file = files[0]
            features = self._single_run(file, progress_callback, operation_callback)
        elif len(files) > 1:
            features = self._batch_run(files, progress_callback, operation_callback)
        else:
            features = []
        return features

    def _single_run(self, file, progress_callback=None, operation_callback=None):
        """
        Processing single *.mzML file

        Parameters
        ----------
        file : str
            path to *.mzML file

        Returns
        -------
        features : list
            a list of 'Feature' objects (each consist of single ROI)
        """
        # get ROIs from raw spectrum
        if operation_callback is not None:
            operation_callback.emit(f'Detecting ROIs in {os.path.basename(file)}:')
        rois = get_ROIs(file, self.delta_mz, self.required_points, self.dropped_points, progress_callback)
        features = []
        percentage = -1
        if operation_callback is not None:
            operation_callback.emit(f'Finding peaks in detected ROIs:')
        for i, roi in enumerate(rois):
            features_from_roi = super(FilesRunner, self).__call__(roi, file)
            features.extend(features_from_roi)
            new_percentage = int(i * 100 / len(rois))
            if progress_callback is not None and new_percentage > percentage:
                percentage = new_percentage
                progress_callback.emit(percentage)

        parameters = {'files': [file], 'delta mz': self.delta_mz, 'required points': self.required_points,
                      'dropped_points': self.dropped_points, 'peak minimum points': self.peak_minimum_points}
        return features, parameters

    def _batch_run(self, files, progress_callback=None, operation_callback=None):
        """
        Processing a batch of  *.mzML files

        Parameters
        ----------
        files : list
            list of paths to *.mzML files

        Returns
        -------
        features : list
            a list of 'Feature' objects
        """
        # ROI detection
        rois = {}
        for file in files:  # get ROIs for every file
            if operation_callback is not None:
                operation_callback.emit(f'Detecting ROIs in {os.path.basename(file)}:')
            rois[file] = get_ROIs(file, self.delta_mz, self.required_points, self.dropped_points, progress_callback)


        if operation_callback is not None:
            operation_callback.emit(f'Alignment of ROIs:')
        # ROI alignment
        mzregions = construct_mzregions(rois, self.delta_mz)  # construct mz regions
        components = rt_grouping(mzregions)  # group ROIs in mz regions based on RT
        aligned_components = []  # component alignment
        percentage = -1
        for i, component in enumerate(components):
            aligned_components.append(align_component(component))
            new_percentage = int(i * 100 / len(components))
            if progress_callback is not None and new_percentage > percentage:
                percentage = new_percentage
                progress_callback.emit(percentage)

        if operation_callback is not None:
            operation_callback.emit(f'Finding peaks in detected ROIs:')
        # Classification, integration and correction
        component_number = 0
        features = []
        percentage = -1
        for j, component in enumerate(aligned_components):  # run through components
            borders = {}  # borders for rois with peaks
            to_delete = []  # noisy rois in components
            for i, (sample, roi) in enumerate(zip(component.samples, component.rois)):
                if self.mode == 'all in one':
                    signal = preprocess(roi.i, self.device)
                    classifier_output, segmentator_output = self.model(signal)
                    classifier_output = classifier_output.data.cpu().numpy()
                    label = np.argmax(classifier_output)
                    if label == 1:
                        segmentator_output = segmentator_output.data.sigmoid().cpu().numpy()
                        borders[sample] = get_borders(segmentator_output[0, 0, :], segmentator_output[0, 1, :],
                                                      peak_minimum_points=self.peak_minimum_points)
                    else:
                        to_delete.append(i)
                elif self.mode == 'sequential':
                    signal = preprocess(roi.i, self.device, interpolate=True, length=256)
                    classifier_output, _ = self.classifier(signal)
                    classifier_output = classifier_output.data.cpu().numpy()
                    label = np.argmax(classifier_output)
                    if label == 1:
                        _, segmentator_output = self.segmentator(signal)
                        segmentator_output = segmentator_output.data.sigmoid().cpu().numpy()
                        borders[sample] = get_borders(segmentator_output[0, 0, :], segmentator_output[0, 1, :],
                                                      peak_minimum_points=self.peak_minimum_points,
                                                      interpolation_factor=len(signal[0, 0]) / len(roi.i))
                    else:
                        to_delete.append(i)
                else:
                    assert False, self.mode

            if len(borders) > len(files) // 3:  # enough rois contain a peak
                component.pop(to_delete)  # delete ROIs which don't contain peaks
                border_correction(component, borders)
                features.extend(build_features(component, borders, component_number))
                component_number += 1

            new_percentage = int(j * 100 / len(aligned_components))
            if progress_callback is not None and new_percentage > percentage:
                percentage = new_percentage
                progress_callback.emit(percentage)

        features = feature_collapsing(features)
        # to do: is it necessary?
        # explicitly delete features which were found in not enough quantity of ROIs
        to_delete = []
        for i, feature in enumerate(features):
            if len(feature) <= len(files) // 3:  # to do: adjustable parameter
                to_delete.append(i)
        for j in to_delete[::-1]:
            features.pop(j)
        print('total number of features: {}'.format(len(features)))
        parameters = {'files': files, 'delta mz': self.delta_mz, 'required points': self.required_points,
                      'dropped_points': self.dropped_points, 'peak minimum points': self.peak_minimum_points}
        return features, parameters
