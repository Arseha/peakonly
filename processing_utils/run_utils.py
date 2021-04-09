import os
import torch
import numpy as np
from collections import defaultdict
from scipy.interpolate import interp1d
from processing_utils.matching import intersected, conv2correlation
from itertools import permutations


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


def preprocess(signal, device, interpolate=False, length=None):
    """
    :param signal: intensities in roi
    :param device: cpu or gpu
    :param points: number of point needed for CNN
    :return: preprocessed intensities which can be used in CNN
    """
    if interpolate:
        interpolate = interp1d(np.arange(len(signal)), signal, kind='linear')
        signal = interpolate(np.arange(length) / (length - 1) * (len(signal) - 1))
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


def border_intersection(border, avg_border):
    """
    Check if borders intersect
    :param border: the real border in number of scans
    :param avg_border: averaged within similarity group border in number of scans
    :return: True/False
    """
    # to do: adjustable parameter?
    return intersected(border[0], border[1], avg_border[0], avg_border[1], 0.6)


def get_borders(integration_mask, intersection_mask, peak_minimum_points=5,
                threshold=0.5, interpolation_factor=1):
    """
    Parameters
    ----------
    integration_mask
    intersection_mask
    peak_minimum_points
    split_threshold
    threshold
    interpolation_factor

    Returns
    -------

    """
    domain = integration_mask * (1 - intersection_mask) > threshold
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
            if peak_wide / interpolation_factor > peak_minimum_points:
                number_of_peaks += 1
                b = int(begin // interpolation_factor)
                e = int((n + 2) // interpolation_factor)  # to do: why n+2?
                borders_roi.append([b, e])
            begin = -1
            peak_wide = 0
    if begin != -1 and peak_wide * interpolation_factor > peak_minimum_points:
        number_of_peaks += 1
        b = int(begin // interpolation_factor)
        e = int(len(domain) // interpolation_factor)
        borders_roi.append([b, e])
    return borders_roi


def intersection(begin1, end1, begin2, end2):
    """Returns the fraction of the shortest interval that is covered by the
    intersection of the two"""
    lower = (end1 <= end2) and (end1 > begin2)
    bigger = (end1 > end2) and (end2 > begin1)
    if lower:
        intersection = end1 - np.max([begin1, begin2])
        smallest = min((end1 - begin1, end2 - begin2))
        ans = (intersection / smallest)
    elif bigger:
        intersection = end2 - np.max([begin1, begin2])
        smallest = min((end1 - begin1, end2 - begin2))
        ans = (intersection / smallest)
    else:
        ans = 0.
    return ans


def border2average_correction(borders, averaged_borders):
    """
    Correct borders based on averaged borders
    :param borders: borders for current ROI in number of scans
    :param averaged_borders: averaged within similarity group borders in number of scans
    :return: corrected borders for current ROI in number of scan
    """

    if len(borders) == 0:
        return averaged_borders
    #if len(averaged_borders) == 0:
    #    return borders

    # Use the fractional overlap between borders and averaged_borders to construct
    # a mapping matrix between the two (a possible fix for issue #9)
    overlap = np.zeros((len(borders), len(averaged_borders)))
    for i, border in enumerate(borders):
        for j, avg_border in enumerate(averaged_borders):
            overlap[i,j] = intersection(border[0], border[1], avg_border[0], avg_border[1])

    mapping_matrix = np.zeros(overlap.shape, dtype=np.int)
    mapping_matrix[overlap.argmax(axis=0), range(mapping_matrix.shape[1])] = 1

    corrected_borders = []
    added = np.zeros(len(borders), dtype=np.uint8)
    for i, line in enumerate(mapping_matrix):
        if np.sum(line) > 1:  # misssing separation (even multiple almost impossible case)
            current = 1
            total = np.sum(line)
            for j in np.where(line == 1)[0]:
                if current == 1:
                    begin = min((borders[i][0], averaged_borders[j][0]))
                    corrected_borders.append([begin, averaged_borders[j][1]])
                elif current < total:
                    corrected_borders.append([averaged_borders[j][0], averaged_borders[j][1]])
                else:
                    end = max((borders[i][1], averaged_borders[j][1]))
                    corrected_borders.append([end, borders[i][1]])
                current += 1
            added[i] = 1  # added border from original borders
        elif np.sum(line) == 0:  # extra peak
            # label that added to exclude
            added[i] = 1

    for j, column in enumerate(mapping_matrix.T):
        if np.sum(column) > 1:  # redundant separation
            begin, end = None, None
            for i in np.where(column == 1)[0]:
                if begin is None and end is None:
                    begin, end = borders[i]
                else:
                    begin = np.min((begin, borders[i][0]))
                    end = np.max((end, borders[i][1]))
                assert added[i] != 1, '"many-to-many" case here must be impossible!'
                added[i] = 1
            corrected_borders.append([begin, end])
        elif np.sum(column) == 0:  # missed peak
            # added averaged borders
            corrected_borders.append(averaged_borders[j])

    # add the ramaining ("one-to-one") cases
    for i in np.where(added == 0)[0]:
        corrected_borders.append(borders[i])

    # sort corrected borders
    corrected_borders.sort()
    return corrected_borders


def border_correction(component, borders):
    """
    https://cs.stackexchange.com/questions/10713/algorithm-to-return-largest-subset-of-non-intersecting-intervals
    :param component: a groupedROI object
    :param borders: dict - key is a sample name, value is a n_borders x 2 matrix;
        predicted, corrected and transformed to normal values borders
    :return: None (in-place correction)
    """
    # to do: to average not borders, but predictions of CNNs
    n_samples = len(component.samples)
    scan_borders = defaultdict(list)  # define borders in shifted scans (size n_samples x n_borders*2 (may vary))
    for k, sample in enumerate(component.samples):
        scan_begin, _ = component.rois[k].scan
        shift = component.shifts[k]
        shifted_borders = []
        for border in borders[sample]:
            shifted_borders.append([border[0] + scan_begin + shift, border[1] + scan_begin + shift])
        scan_borders[sample] = shifted_borders

    # border correction within the similarity group
    labels = np.unique(component.grouping)
    for label in labels:
        # find total begin and end in one similarity group
        total_begin, total_end = None, None
        # and maximum number of peaks
        max_number_of_peaks = 0
        for i, sample in enumerate(component.samples):
            # to do: it would be better to have mapping from group to samples and numbers
            if component.grouping[i] == label:
                shift = component.shifts[i]
                if total_begin is None and total_end is None:
                    total_begin, total_end = component.rois[i].scan
                    total_begin += shift
                    total_end += shift
                else:
                    begin, end = component.rois[i].scan
                    total_begin = min((total_begin, begin + shift))
                    total_end = max((total_end, end + shift))
                max_number_of_peaks = max((max_number_of_peaks, len(borders[sample])))

        # find averaged integration domains
        averaged_domain = np.zeros(total_end - total_begin)
        samples_in_group = 0
        for i, sample in enumerate(component.samples):
            # to do: it would be better to have mapping from group to samples and numbers
            if component.grouping[i] == label:
                samples_in_group += 1
                for border in scan_borders[sample]:
                    averaged_domain[border[0] - total_begin:border[1] - total_begin] += 1
        averaged_domain = averaged_domain / samples_in_group

        # calculate number of peaks within similarity group
        averaged_domain = averaged_domain > 0.5  # to do: adjustable parameter?
        number_of_peaks = 0
        averaged_borders = []
        begin = 0 if averaged_domain[0] else - 1
        # to do: think about peak wide and peak minimum points
        # to do: the following code is almost the exact copy
        # of the part of border_prediction function. Separate function?
        for n in range(len(averaged_domain) - 1):
            if averaged_domain[n + 1] and not averaged_domain[n]:  # peak begins
                begin = n + 1
            elif not averaged_domain[n + 1] and begin != -1:  # peak ends
                number_of_peaks += 1
                averaged_borders.append([begin + total_begin, n + 2 + total_begin])  # to do: why n+2?
                begin = -1
        if begin != -1:
            number_of_peaks += 1
            averaged_borders.append([begin + total_begin, len(averaged_domain) + 1 + total_begin])  # to do: why n+2?
            begin = -1

        while number_of_peaks > max_number_of_peaks:  # need to merge some borders
            # to do: rethink this idea
            # compute 'many-to-one' cases
            counter = np.zeros(number_of_peaks)
            for i, sample in enumerate(component.samples):
                # to do: it would be better to have mapping from group to samples and numbers
                if component.grouping[i] == label:
                    mapping_matrix = np.zeros((len(scan_borders[sample]), len(averaged_borders)), dtype=np.int)
                    for k, border in enumerate(scan_borders[sample]):
                        for j, avg_border in enumerate(averaged_borders):
                            mapping_matrix[k, j] += border_intersection(border, avg_border)
                    for line in mapping_matrix:
                        if np.sum(line) > 1:
                            for j in np.where(line == 1):
                                counter[j] += 1
            counter_order = np.argsort(counter)
            assert np.abs(counter_order[-1] - counter_order[-2]) == 1, "almost impossible case :)"
            mergeable = min((counter_order[-1], counter_order[-2]))
            averaged_borders[mergeable][1] = averaged_borders[mergeable + 1][1]
            averaged_borders.pop(mergeable + 1)
            number_of_peaks -= 1

        # finally border correctrion
        for i, sample in enumerate(component.samples):
            # to do: it would be better to have mapping from group to samples and numbers
            if component.grouping[i] == label:
                scan_borders[sample] = border2average_correction(scan_borders[sample], averaged_borders)
                # to do: add border2borders_correction

        # change initial borders (reverse shift of scan_borders)
        for k, sample in enumerate(component.samples):
            scan_begin, _ = component.rois[k].scan
            shift = component.shifts[k]
            shifted_borders = []
            for border in scan_borders[sample]:
                shifted_borders.append([max((border[0] - scan_begin - shift, 0)),
                                        max((1, min((border[1] - scan_begin - shift, len(component.rois[k].i)))))])
            borders[sample] = shifted_borders


class Feature:
    def __init__(self, samples, rois, borders, shifts,
                 intensities, mz, rtmin, rtmax,
                 mzrtgroup, similarity_group):
        # common information
        self.samples = samples
        self.rois = rois
        self.borders = borders
        self.shifts = shifts
        # feature specific information
        self.intensities = intensities
        self.mz = mz
        self.rtmin = rtmin
        self.rtmax = rtmax
        # extra information
        self.mzrtgroup = mzrtgroup  # from the same or separate groupedROI object
        self.similarity_group = similarity_group

    def __len__(self):
        return len(self.samples)

    def append(self, sample, roi, border, shift,
               intensity, mz, rtmin, rtmax):
        if self.samples:
            self.mz = (self.mz * len(self) + mz) / (len(self) + 1)
            self.rtmin = min((self.rtmin, rtmin))
            self.rtmax = max((self.rtmax, rtmax))
        else:  # feature is empty
            self.mz = mz
            self.rtmin = rtmin
            self.rtmax = rtmax

        self.samples.append(sample)
        self.rois.append(roi)
        self.borders.append(border)
        self.shifts.append(shift)
        self.intensities.append(intensity)

    def extend(self, feature):
        if self.samples:
            self.mz = (self.mz * len(self) + feature.mz * len(feature)) / (len(self) + len(feature))
            self.rtmin = min((self.rtmin, feature.rtmin))
            self.rtmax = max((self.rtmax, feature.rtmax))
        else:  # feature is empty
            self.mz = feature.mz
            self.rtmin = feature.rtmin
            self.rtmax = feature.rtmax

        self.samples.extend(feature.samples)
        self.rois.extend(feature.rois)
        self.borders.extend(feature.borders)
        self.shifts.extend(feature.shifts)
        self.intensities.extend(feature.intensities)

    def plot(self, ax, shifted=True, show_legend=True):
        """
        Visualize Feature object
        """
        name2label = {}
        label2class = {}
        labels = set()
        for sample in self.samples:
            label = os.path.basename(os.path.dirname(sample))
            labels.add(label)
            name2label[sample] = label

        for i, label in enumerate(labels):
            label2class[label] = i

        m = len(labels)
        for sample, roi, shift, border in sorted(zip(self.samples, self.rois, self.shifts, self.borders),
                                                 key=lambda zipped: zipped[0]):
            y = roi.i
            if shifted:
                x = np.linspace(roi.scan[0] + shift, roi.scan[1] + shift, len(y))
            else:
                x = np.linspace(roi.scan[0], roi.scan[1], len(y))
            label = label2class[name2label[sample]]
            c = [label / m, 0.0, (m - label) / m]
            ax.plot(x, y, color=c)
            ax.fill_between(x[border[0]:border[1]], y[border[0]:border[1]], color=c,
                            alpha=0.5, label=os.path.basename(sample))
        if show_legend:
            ax.legend(loc='best')


def build_features(component, borders, initial_group):
    """
    Integrate peaks within similarity components and build features
    :param component: a groupedROI object
    :param borders: dict - key is a sample name, value is a (n_borders x 2) matrix;
        predicted, corrected and transformed to normal values borders
    :param initial_group: a number of mzrt group
    :return: None (in-place correction)
    """
    rtdiff = (component.rois[0].rt[1] - component.rois[0].rt[0])
    scandiff = (component.rois[0].scan[1] - component.rois[0].scan[0])
    frequency = scandiff / rtdiff

    features = []
    labels = np.unique(component.grouping)
    for label in labels:
        # compute number of peaks
        peak_number = None
        for i, sample in enumerate(component.samples):
            # to do: it would be better to have mapping from group to samples and numbers
            if component.grouping[i] == label:
                peak_number = len(borders[sample])

        for p in range(peak_number):
            # build feature
            intensities = []
            samples = []
            rois = []
            feature_borders = []
            shifts = []
            rtmin, rtmax, mz = None, None, None
            for i, sample in enumerate(component.samples):
                # to do: it would be better to have mapping from group to samples and numbers
                if component.grouping[i] == label:
                    assert len(borders[sample]) == peak_number
                    begin, end = borders[sample][p]
                    intensity = np.sum(component.rois[i].i[begin:end])
                    intensities.append(intensity)
                    samples.append(sample)
                    rois.append(component.rois[i])
                    feature_borders.append(borders[sample][p])
                    shifts.append(component.shifts[i])
                    if mz is None:
                        mz = component.rois[i].mzmean
                        rtmin = component.rois[i].rt[0] + begin / frequency
                        rtmax = component.rois[i].rt[0] + end / frequency
                    else:
                        mz = (mz * i + component.rois[i].mzmean) / (i + 1)
                        rtmin = min((rtmin, component.rois[i].rt[0] + begin / frequency))
                        rtmax = max((rtmax, component.rois[i].rt[0] + end / frequency))
            features.append(Feature(samples, rois, feature_borders, shifts,
                                    intensities, mz, rtmin, rtmax,
                                    initial_group, label))
    # to do: there are a case, when borders are empty
    # assert len(features) != 0
    return features


def calculate1dios(s1, s2):
    """
    Calculate intersection over smallest
    """
    res = 0
    i_b = max((s1[0], s2[0]))  # intersection begin
    i_e = min((s1[1], s2[1]))  # intersection end
    if i_b < i_e:
        smallest = min((s1[1] - s1[0], s2[1] - s2[0]))
        res = (i_e - i_b) / smallest
    return res


def collapse_mzrtgroup(mzrtgroup, code):
    """
    Collapse features from the same component based on peaks similarities
    :param mzrtgroup: list of Feature objects from the same component
    :param code: a number (code) of mzrtgroup
    :return: new list of collapsed Feature objects
    """
    label2idx = defaultdict(list)
    for idx, feature in enumerate(mzrtgroup):
        label2idx[feature.similarity_group].append(idx)
    unique_labels = list(set(label for label in label2idx))  # to do: not the best way :)

    # find most intense peaks in each feature
    idx2basepeak = dict()  # feature id in mzrtgroup to basepeak (np.array)
    for idx, feature in enumerate(mzrtgroup):
        n = 0
        for k in range(1, len(feature)):
            if feature.intensities[k] > feature.intensities[n]:
                n = k
        b, e = feature.borders[n]
        basepeak = np.array(feature.rois[n].i[b:e])
        idx2basepeak[idx] = {'peak': basepeak, 'rt': (feature.rtmin, feature.rtmax)}


    similarity_values = np.zeros((len(mzrtgroup), len(mzrtgroup), 2))
    feature_n = 0
    for i, label in enumerate(unique_labels):
        for idx in label2idx[label]:  # iter over features in one similarity group (from the same ROI)
            base_peak = idx2basepeak[idx]['peak']
            base_rt = idx2basepeak[idx]['rt']
            similarity_values[feature_n][idx] = (1, 1)

            for j, comp_label in enumerate(unique_labels[i + 1:]):
                for jdx in label2idx[comp_label]:
                    comp_peak = idx2basepeak[jdx]['peak']
                    comp_rt = idx2basepeak[jdx]['rt']
                    # calculate 'iou'
                    inter = calculate1dios(base_rt, comp_rt)
                    # calculate cross-correlation
                    conv_vector = np.convolve(base_peak[::-1], comp_peak, mode='full')
                    corr_vector = conv2correlation(base_peak, comp_peak, conv_vector)
                    corr = np.max(corr_vector)
                    if inter > 0.8 or corr > 0.8:
                        similarity_values[feature_n][jdx] = [inter, corr]
            feature_n += 1

    # to do: completely rewrite it
    # from similarity_values construct mapping matrix
    mapping_matrix = np.eye(len(mzrtgroup), dtype=np.int)
    for i, label in enumerate(unique_labels):
        for j, comp_label in enumerate(unique_labels[i + 1:]):
            submatrix = similarity_values[label2idx[label]][:, label2idx[comp_label]]
            raws, columns, _ = submatrix.shape
            if raws > columns:
                ones_position = np.arange(columns)
                score = 0
                for c, r in enumerate(ones_position):
                    if submatrix[r, c, 0] > 0.8:
                        score += 2
                    else:
                        score += submatrix[r, c, 0] + submatrix[r, c, 1]
                # to do: write your own iterator which forbids cross shifts
                for position_candidate in permutations(np.arange(raws), columns):
                    flag_exit = False
                    for c in range(columns - 1):
                        if position_candidate[c + 1] < position_candidate[c]:
                            r1 = position_candidate[c]
                            r2 = position_candidate[c + 1]
                            if submatrix[r1, c, 0] + submatrix[r1, c, 1] != 0 and \
                               submatrix[r2, c + 1, 0] + submatrix[r2, c + 1, 1] != 0:
                                flag_exit = True
                                break
                    if flag_exit:
                        continue
                    candidate_score = 0
                    for c, r in enumerate(position_candidate):
                        if submatrix[r, c, 0] > 0.8:
                            candidate_score += 2
                        else:
                            candidate_score += submatrix[r, c, 0] + submatrix[r, c, 1]
                    if candidate_score > score:
                        score = candidate_score
                        ones_position = position_candidate
                for c, r in enumerate(ones_position):
                    if submatrix[r, c, 0] + submatrix[r, c, 1] != 0:
                        mapping_matrix[label2idx[label][r], label2idx[comp_label][c]] += 1
            else:
                ones_position = np.arange(raws)
                score = 0
                for r, c in enumerate(ones_position):
                    if submatrix[r, c, 0] > 0.8:
                        score += 2
                    else:
                        score += submatrix[r, c, 0] + submatrix[r, c, 1]
                # to do: write your own iterator which forbids cross shifts
                for position_candidate in permutations(np.arange(columns), raws):
                    flag_exit = False
                    for r in range(raws - 1):
                        if position_candidate[r + 1] < position_candidate[r]:
                            c1 = position_candidate[r]
                            c2 = position_candidate[r + 1]
                            if submatrix[r, c1, 0] + submatrix[r, c1, 1] != 0 and \
                                    submatrix[r+1, c2, 0] + submatrix[r+1, c2, 1] != 0:
                                flag_exit = True
                                break
                    if flag_exit:
                        continue
                    candidate_score = 0
                    for r, c in enumerate(position_candidate):
                        if submatrix[r, c, 0] > 0.8:
                            candidate_score += 2
                        else:
                            candidate_score += submatrix[r, c, 0] + submatrix[r, c, 1]
                    if candidate_score > score:
                        score = candidate_score
                        ones_position = position_candidate
                for r, c in enumerate(ones_position):
                    if submatrix[r, c, 0] + submatrix[r, c, 1] != 0:
                        mapping_matrix[label2idx[label][r], label2idx[comp_label][c]] += 1
    for j, columns in enumerate(mapping_matrix.T):
        if np.sum(columns) > 1:
            for i in np.where(columns == 1)[0][1:]:
                mapping_matrix[i, j] = 0

    # create 'new' features
    new_features = []
    for line in mapping_matrix:
        if np.sum(line) != 0:
            feature = Feature([], [], [], [], [], None, None, None, code, None)
            for idx in np.where(line != 0)[0]:
                feature.extend(mzrtgroup[idx])
            new_features.append(feature)

    return new_features


def feature_collapsing(features):
    """
    Collapse features from the same component based on peaks similarities
    with the use of 'collapse_mzrtgroup'
    :param features: list of Feature objects
    :return: new list of collapsed Feature objects
    """
    new_features = []
    group_number = 0
    mzrtgroup = []
    for feature in features:
        if feature.mzrtgroup == group_number:
            mzrtgroup.append(feature)
        else:
            # assert feature.mzrtgroup == group_number + 1  # to do: there are a case, when borders are empty
            new_features.extend(collapse_mzrtgroup(mzrtgroup, group_number))
            mzrtgroup = [feature]
            group_number = feature.mzrtgroup
    new_features.extend(collapse_mzrtgroup(mzrtgroup, group_number))
    return new_features
