import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.sparse.csgraph import connected_components
from utils.roi import ROI


class mzRegion:
    """
    A class that stores the beginning and the of the mass region
    and all the ROIs that lays there
    """
    def __init__(self, mzbegin, mzend, rois=None):
        """
        :param mzbegin: begin of the mass region
        :param mzend: end of the mass region
        :param rois: ROIs in the region should be defaultdict(list)
        """
        self.mzbegin = mzbegin
        self.mzend = mzend
        self.rois = defaultdict(list) if rois is None else rois

    def __contains__(self, mz):
        return self.mzbegin <= mz <= self.mzend

    def __len__(self):
        return len(self.rois)

    def extend(self, sample_rois):
        for k, v in sample_rois.items():
            self.rois[k].extend(v)

    def append(self, sample, roi):
        self.rois[sample].append(roi)


def construct_mzregions(ROIs, delta_mz):
    """
    :param ROIs: a dictionary, where keys are file names and values are lists of rois
    :param delta_mz: int
    :return: a list of mzRegion objects
    """
    mz_mins = np.array([min(roi.mz) for s in ROIs.values() for roi in s])
    mz_maxs = np.array([max(roi.mz) for s in ROIs.values() for roi in s])
    rois = np.array([(name, roi) for name, s in ROIs.items() for roi in s])

    # reorder values based on mins
    order = np.argsort(mz_mins)
    mz_mins = mz_mins[order]
    mz_maxs = mz_maxs[order]
    rois = rois[order]

    mzregions = []
    roi_dict = defaultdict(list)  # save all rois within current region
    region_begin, region_end = mz_mins[0], mz_maxs[0]
    for begin, end, name_roi in zip(mz_mins, mz_maxs, rois):
        if begin > region_end + delta_mz:
            # add new mzRegion object
            mzregions.append(mzRegion(region_begin, region_end, roi_dict))
            roi_dict = defaultdict(list)
            region_begin, region_end = begin, end
        else:
            region_end = end if end > region_end else region_end
        name, roi = name_roi
        roi_dict[name].append(roi)
    mzregions.append(mzRegion(region_begin, region_end, roi_dict))
    return mzregions


def intersected(begin1, end1, begin2, end2, percentage=None):
    """
    A simple function which determines if two segments intersect
    :return: bool
    """
    lower = (end1 < end2) and (end1 > begin2)
    bigger = (end1 > end2) and (end2 > begin1)
    if percentage is None:
        ans = (lower or bigger)
    else:
        if lower:
            intersection = end1 - np.max([begin1, begin2])
            smallest = min((end1 - begin1, end2 - begin2))
            ans = (intersection / smallest) > percentage
        elif bigger:
            intersection = end2 - np.max([begin1, begin2])
            smallest = min((end1 - begin1, end2 - begin2))
            ans = (intersection / smallest) > percentage
        else:
            ans = False
    return ans


def roi_intersected(one_roi, two_roi, percentage=None):
    """
    A function that determines if two roi intersect based on rt and mz.
    :return: bool
    """
    ans = False
    if intersected(min(one_roi.mz),
                   max(one_roi.mz),
                   min(two_roi.mz),
                   max(two_roi.mz)) and \
       intersected(one_roi.rt[0],
                   one_roi.rt[1],
                   two_roi.rt[0],
                   two_roi.rt[1],
                   percentage):
        ans = True
    return ans


def rt_grouping(mzregions):
    """
    A function that groups roi inside mzregions.
    :param mzregions: a list of mzRegion objects
    :return: a list of defaultdicts, where the key is the name of file and value is a list of ROIs
    """
    components = []
    for region in mzregions:
        region = np.array([(name, roi) for name, s in region.rois.items() for roi in s])
        n = len(region)
        graph = np.zeros((n, n), dtype=np.uint8)
        for i in range(n - 1):
            for j in range(i + 1, n):
                graph[i, j] = roi_intersected(region[i][1], region[j][1])
        n_components, labels = connected_components(graph, directed=False)

        for k in range(n_components):
            rois = region[labels == k]
            component = defaultdict(list)
            for roi in rois:
                component[roi[0]].append(roi[1])
            components.append(component)
    return components


class groupedROI:
    """
    A class that represents a group of ROIs
    """

    def __init__(self, rois, shifts, samples, grouping):
        self.rois = rois  # rois
        self.shifts = shifts  # shifts for each roi
        self.samples = samples  # samples names
        self.grouping = grouping  # similarity groups

    def __len__(self):
        length = len(self.rois)
        assert length == len(self.shifts)
        assert length == len(self.samples)
        assert length == len(self.grouping)
        return length

    def append(self, roi, shift, sample, group_number):
        self.rois.append(roi)
        self.shifts.append(shift)
        self.samples.append(sample)
        self.grouping.append(group_number)

    def pop(self, idx):
        if isinstance(idx, list):
            for j in sorted(idx, reverse=True):
                self.rois.pop(j)
                self.shifts.pop(j)
                self.samples.pop(j)
                self.grouping.pop(j)
        else:
            assert isinstance(idx, int)
            self.rois.pop(idx)
            self.shifts.pop(idx)
            self.samples.pop(idx)
            self.grouping.pop(idx)

    def plot(self, based_on_grouping=False):
        """
            Visualize a groupedROI object
        """
        name2label = {}
        label2class = {}
        labels = set()
        if based_on_grouping:
            labels = set(self.grouping)
            for i, sample in enumerate(self.samples):
                name2label[sample] = self.grouping[i]
            for label in labels:
                label2class[label] = label  # identical transition
        else:
            for sample in self.samples:
                end = sample.rfind('/')
                begin = sample[:end].rfind('/') + 1
                label = sample[begin:end]
                labels.add(label)
                name2label[sample] = label

            for i, label in enumerate(labels):
                label2class[label] = i

        m = len(labels)
        mz = []
        scan_begin = []
        scan_end = []
        fig, axes = plt.subplots(1, 2)
        for sample, roi, shift in zip(self.samples, self.rois, self.shifts):
            mz.append(roi.mzmean)
            scan_begin.append(roi.scan[0] + shift)
            scan_end.append(roi.scan[1] + shift)
            y = roi.i
            x = np.linspace(roi.scan[0], roi.scan[1], len(y))
            x_shifted = np.linspace(roi.scan[0] + shift, roi.scan[1] + shift, len(y))
            label = label2class[name2label[sample]]
            c = [label / m, 0.0, (m - label) / m]
            axes[0].plot(x, y, color=c)
            axes[1].plot(x_shifted, y, color=c)
        fig.suptitle('mz = {:.4f}, scan = {:.2f} -{:.2f}'.format(np.mean(mz), min(scan_begin), max(scan_end)))

    def adjust(self, history, adjustment_threshold):
        labels, counts = np.unique(self.grouping, return_counts=True)
        counter = {label: count for label, count in zip(labels, counts)}

        for i, sample in enumerate(self.samples):
            if counter[self.grouping[i]] == 1:
                best_gn, best_corr, best_shift = None, None, None
                for gn, corr, shift in history[sample]:
                    if (best_corr is None or corr > best_corr) and counter[gn] != 1:
                        best_gn = gn
                        best_corr = corr
                        best_shift = shift
                if best_corr is not None and best_corr > adjustment_threshold:
                    self.grouping[i] = best_gn
                    self.shifts[i] = best_shift


def stitch_component(component):
    """
    Stitching roi, which resulted from one file in one group
    :param component: defaultdict where the key is the name of file and value is a list of ROIs
    :return: new_component with stitched ROIs
    """
    new_component = defaultdict(list)
    #
    for file in component:
        begin_scan, end_scan = component[file][0].scan
        begin_rt, end_rt = component[file][0].rt
        for roi in component[file]:
            begin, end = roi.scan
            if begin < begin_scan:
                begin_scan = begin
                begin_rt = roi.rt[0]
            if end > end_scan:
                end_scan = end
                end_rt = roi.rt[1]

        # to do: use parameter with missing zeros (not 7)
        i = np.zeros(end_scan - begin_scan + 1)
        mz = np.zeros(end_scan - begin_scan + 1)
        for roi in component[file]:
            begin, end = roi.scan
            i[begin - begin_scan:end - begin_scan + 1] = roi.i
            mz[begin - begin_scan:end - begin_scan + 1] = roi.mz
        mzmean = np.mean(mz[mz != 0])  # mean based on nonzero elements
        new_component[file] = [ROI([begin_scan, end_scan],
                                   [begin_rt, end_rt],
                                   i, mz, mzmean)]
    return new_component


def conv2correlation(roi_i, base_roi_i, conv_vector):
    n = np.zeros_like(conv_vector)
    x = np.sum(roi_i)
    y = np.sum(base_roi_i)
    x_square = np.sum(np.power(roi_i, 2))  # to do: make roi.i np.array by default
    y_square = np.sum(np.power(base_roi_i, 2))

    min_l = min((len(roi_i), len(base_roi_i)))
    max_l = max((len(roi_i), len(base_roi_i)))

    n[:min_l - 1] = np.arange(len(conv_vector), max_l, -1)
    n[min_l - 1:min_l + max_l - min_l] = max_l
    n[min_l + max_l - min_l:] = np.arange(max_l + 1, len(conv_vector) + 1, 1)

    return (n * conv_vector - x * y) / (np.sqrt(n * x_square - x ** 2) * np.sqrt(n * y_square - y ** 2))


def align_component(component, max_shift=20):
    """
    Align ROIs in component based on point-wise correlation
    :param component: defaultdict where the key is the name of file and value is a list of ROIs
    :param max_shift: maximum shift in scans
    :return: an groupedROI object
    """
    # stitching first
    component = stitch_component(component)
    # find base_sample which correspond to the sample with highest intensity within roi
    correlation_threshold = 0.8
    adjustment_threshold = 0.4
    group_number = 0
    aligned_component = groupedROI([], [], [], [])
    # save (group-number, correlation, shift)
    history = defaultdict(list)

    while len(component) != 0:
        # choose base ROI from the remaining
        max_i = 0
        base_sample, base_roi = None, None
        for sample in component:
            assert len(component[sample]) == 1
            for roi in component[sample]:  # in fact there are only one roi
                i = np.max(roi.i)
                if i > max_i:
                    max_i = i
                    base_sample, base_roi = sample, roi

        component.pop(base_sample)  # delete chosen ROI from component
        aligned_component.append(base_roi, 0, base_sample, group_number)

        to_delete = []
        for sample in component:
            roi = component[sample][0]
            # position, when two ROIs begins simultaneously
            pos = len(roi.i) - 1  # to do: check it
            conv_vector = np.convolve(roi.i[::-1], base_roi.i, mode='full')  # reflection is necessary
            corr_vector = conv2correlation(roi.i, base_roi.i, conv_vector)
            # to do: find local maxima greater than threshold
            shift = np.argmax(corr_vector) - pos - roi.scan[0] + base_roi.scan[0]
            max_corr = np.max(corr_vector)
            if max_corr > correlation_threshold:
                to_delete.append(sample)  # delete ROI from component
                aligned_component.append(roi, shift, sample, group_number)
            history[sample].append((group_number, max_corr, shift))  # history for after adjustment

        for sample in to_delete:
            component.pop(sample)

        group_number += 1  # increase group number

    aligned_component.adjust(history, adjustment_threshold)  # to do: decide is it necessary
    return aligned_component
