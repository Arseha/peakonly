import os
import torch
import numpy as np
from scipy.interpolate import interp1d


def find_mzML(path, array=None):
    if array is None:
        array = []
    for obj in os.listdir(path):
        obj_path = os.path.join(path, obj)
        if os.path.isdir(obj_path):  # object is a directory
            find_mzML(obj_path, array)
        elif obj_path[-4:] == 'mzML':  # object is mzML file
            array.append(obj_path)
    return array


def preprocess(signal, device, points):
    """
    :param signal: intensities in roi
    :param device: cpu or gpu
    :param points: number of point needed for CNN
    :return: preprocessed intensities which can be used in CNN
    """
    interpolate = interp1d(np.arange(len(signal)), signal, kind='linear')
    signal = interpolate(np.arange(points) / (points - 1) * (len(signal) - 1))
    signal = torch.tensor(signal / np.max(signal), dtype=torch.float32, device=device)
    return signal.view(1, 1, -1)


def classifier_prediction(roi, classifier, device, points=256):
    """
    :param roi: an ROI object
    :param classifier: CNN for classification
    :param device: cpu or gpu
    :param points: number of point needed for CNN
    :return: class/label
    """
    signal = preprocess(roi.i, device, points)
    proba = classifier(signal)[0].softmax(0)
    return np.argmax(proba.cpu().detach().numpy())


def border_prediction(roi, integrator, device, peak_minimum_points, points=256, split_threshold=0.95, threshold=0.5):
    """
    :param roi: an ROI object
    :param integrator: CNN for border prediction
    :param device: cpu or gpu
    :param peak_minimum_points: minimum points in peak
    :param points: number of point needed for CNN
    :param split_threshold: threshold for probability of splitter
    :return: borders as an list of size n_peaks x 2
    """
    signal = preprocess(roi.i, device, points)
    logits = integrator(signal).sigmoid()
    splitter = logits[0, 0, :].cpu().detach().numpy()
    domain = (1 - splitter) * logits[0, 1, :].cpu().detach().numpy() > threshold

    borders_signal = []
    borders_roi = []
    begin = 0 if domain[0] else -1
    peak_wide = 1 if domain[0] else 0
    number_of_peaks = 0
    for n in range(len(domain) - 1):
        if domain[n + 1] and not domain[n]:  # peak begins
            begin = n + 1
            peak_wide = 1
        elif domain[n + 1] and begin != -1:  # peak continues
            peak_wide += 1
        elif not domain[n + 1] and begin != -1:  # peak ends
            if peak_wide / points * len(roi.i) > peak_minimum_points:
                number_of_peaks += 1
                borders_signal.append([begin, n + 2])  # to do: why n+2?
                borders_roi.append([np.max((int((begin + 1) * len(roi.i) // points - 1), 0)),
                                    int((n + 2) * len(roi.i) // points - 1) + 1])
            begin = -1
            peak_wide = 0
    # to do: if the peak doesn't end?
    # delete the smallest peak if there is no splitter between them
    n = 0
    while n < number_of_peaks - 1:
        if not np.any(splitter[borders_signal[n][1]:borders_signal[n + 1][0]] > split_threshold):
            intensity1 = np.sum(roi.i[borders_roi[n][0]:borders_signal[n][1]])
            intensity2 = np.sum(roi.i[borders_roi[n + 1][0]:borders_signal[n + 1][1]])
            smallest = n if intensity1 < intensity2 else n + 1
            borders_signal.pop(smallest)
            borders_roi.pop(smallest)
            number_of_peaks -= 1
        else:
            n += 1
    return borders_roi


def correct_classification(labels):
    """
    :param labels: a dictionary, where key is a file name and value is a prediction
    :return: None (in-place correction)
    """
    pred = np.array([v for v in labels.values()])
    counter = [np.sum(pred[pred == label]) for label in [0, 1, 2]]
    # if the majority is "ones" change "two" to "one"
    if counter[1] >= len(labels) // 3:
        for k, v in labels.items():
            if v == 2:
                labels[k] = 1


def border_correction(component, borders):
    """
    https://cs.stackexchange.com/questions/10713/algorithm-to-return-largest-subset-of-non-intersecting-intervals
    :param component: a groupedROI object
    :param borders: dict - key is a sample name, value is a n_borders x 2 matrix;
        predicted, corrected and transformed to normal values borders
    :return: corrected borders
    """
    pass
    '''
     n_samples = len(component.samples)
    scan_borders = []  # define borders in shifted scans (size n_samples x n_borders*2 (may vary))
    for k, sample in enumerate(component.samples):
        scan_begin, _ = component.roi[k].scan
        shift = component.shifts[k]
        flatten_borders = []
        for border in borders[sample]:
            flatten_borders.append(border[0] + scan_begin + shift)
            flatten_borders.append(border[1] + scan_begin + shift)
        scan_borders.append(flatten_borders)

    # find intersected gaps
    for border, shift in zip(borders, component.shifts):
        scan_borders.append(border - shift)
    '''


class Feature:
    def __init__(self, samples, intensities, mz, rtmin, rtmax):
        self.samples = samples
        self.intensities = intensities
        self.mz = mz
        self.rtmin = rtmin
        self.rtmax = rtmax


def build_features(component, borders):
    intensities = []
    rtmin, rtmax, mz = None, None, None
    for k, sample in enumerate(component.samples):
        begin, end = borders[sample][0]  # only single peaks
        intensity = np.sum(component.rois[k].i[begin:end])
        intensities.append(intensity)
        if mz is None:
            mz = component.rois[k].mzmean
            rtmin, rtmax = component.rois[k].rt
        else:
            mz = (mz * k + component.rois[k].mzmean) / (k + 1)
            rtmin = min((rtmin, component.rois[k].rt[0]))
            rtmax = max((rtmax, component.rois[k].rt[1]))
    return Feature(component.samples, intensities, mz, rtmin, rtmax)
