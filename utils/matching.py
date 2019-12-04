import numpy as np
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


def roi_intersected(one_roi, two_roi, percentage=0.7):
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
    def __init__(self, rois, shifts, samples):
        self.rois = rois
        self.shifts = shifts
        self.samples = samples

    def append(self, roi, shift, sample):
        self.rois.append(roi)
        self.shifts.append(shift)
        self.samples.append(sample)

    def pop(self, idx):
        if isinstance(idx, list):
            for j in sorted(idx, reverse=True):
                self.rois.pop(j)
                self.shifts.pop(j)
                self.samples.pop(j)
        else:
            assert isinstance(idx, int)
            self.rois.pop(idx)
            self.shifts.pop(idx)
            self.samples.pop(idx)


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
        mzmean = np.mean(mz)
        new_component[file] = [ROI([begin_scan, end_scan],
                                   [begin_rt, end_rt],
                                   i, mz, mzmean)]
    return new_component


def scale(array):
    """
    :return: a scaled version of the array
    """
    return (array - np.mean(array))/np.std(array)


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
    max_i = 0
    max_len = 0
    base_sample, base_roi = None, None
    for sample in component:
        for roi in component[sample]:
            # should be only one roi!
            i = np.max(roi.i)
            if i > max_i:
                max_i = i
                base_sample, base_roi = sample, roi
            if len(roi.i) > max_len:
                max_len = len(roi.i)
    n = max_len  # to do: come up with a better idea
    base_roi_scaled = np.pad(scale(base_roi.i), (n, n))
    aligned_rois = groupedROI([], [], [])
    for sample in component:
        assert len(component[sample]) == 1  # should be only one roi after stitching
        roi = component[sample][0]
        if sample == base_sample:  # doesn't need to be corrected
            aligned_rois.append(roi, 0, sample)
        else:
            # position in correlation vector, which would correspond to zero shift
            pos = n - base_roi.scan[0] + roi.scan[0]
            len_corr_vector = len(base_roi_scaled) - len(roi.i) + 1  # length of correlation vector
            if pos > len_corr_vector or pos < 0:
                aligned_rois.append(roi, 0, sample)  # too far to shift
            else:
                roi_scaled = scale(roi.i)
                corr_vector = np.convolve(roi_scaled[::-1],  base_roi_scaled, mode='valid')  # reflection is necessary
                assert len(corr_vector) == len_corr_vector
                begin = max(pos - max_shift, 0)
                shift = np.argmax(corr_vector[begin:pos + max_shift]) - pos + begin
                aligned_rois.append(roi, shift, sample)
    return aligned_rois