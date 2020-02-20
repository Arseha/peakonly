import json
import pymzml
import numpy as np
from tqdm import tqdm


def construct_ROI(roi_dict):
    """
    Construct an ROI object from dict
    :param roi: a dict with 'description' (not necessary),
                            'code' (basically the name of file, not necessary),
                            'label' (annotated class),
                            'number of peaks' (quantity of peaks within ROI),
                            'begins' (a list of scan numbers),
                            'ends' (a list of scan numbers),
                            'intersections' (a list of scan numbers),
                            'scan' (first and last scan of ROI),
                            'rt',
                            'intensity',
                            'mz'
    """
    return ROI(roi_dict['scan'], roi_dict['rt'], roi_dict['intensity'], roi_dict['mz'], np.mean(roi_dict['mz']))


class ROI:
    def __init__(self, scan, rt, i, mz, mzmean):
        self.scan = scan
        self.rt = rt
        self.i = i
        self.mz = mz
        self.mzmean = mzmean

    def __repr__(self):
        return 'mz = {:.4f}, rt = {:.2f} - {:.2f}'.format(self.mzmean, self.rt[0], self.rt[1])

    def save_annotated(self, path, label, code=None, number_of_peaks=0, begins=None, ends=None,
                       intersections=None, description=None):
        roi = dict()
        roi['code'] = code
        roi['label'] = label
        roi['number of peaks'] = number_of_peaks
        if begins is None:
            roi['begins'] = []
        else:
            roi['begins'] = begins
        if ends is None:
            roi['ends'] = []
        else:
            roi['ends'] = ends
        if intersections is None:
            roi['intersections'] = []
        else:
            roi['intersections'] = intersections
        roi['description'] = description

        roi['scan'] = self.scan
        roi['rt'] = self.rt
        roi['intensity'] = self.i
        roi['mz'] = self.mz

        with open(path, 'w') as jsonfile:
            json.dump(roi, jsonfile)


class ProcessROI(ROI):
    def __init__(self, scan, rt, i, mz, mzmean):
        super().__init__(scan, rt, i, mz, mzmean)
        self.points = 1


def get_closest(mzmean, mz, pos):
    if pos == len(mzmean):
        res = pos - 1
    elif pos == 0:
        res = pos
    else:
        res = pos if (mzmean[pos] - mz) < (mz - mzmean[pos - 1]) else pos - 1
    return res


def get_ROIs(path, delta_mz=0.005, required_points=15, dropped_points=3, pbar=None):
    '''
    :param path: path to mzml file
    :param delta_mz:
    :param required_points:
    :param dropped_points: can be zero points
    :param pbar: an pyQt5 progress bar to visualize
    :return: ROIs - a list of ROI objects found in current file
    '''
    # read all scans in mzML file
    run = pymzml.run.Reader(path)
    scans = []
    for scan in run:
        scans.append(scan)

    ROIs = []  # completed ROIs
    process_ROIs = []  # processed ROIs

    # initialize a processed data
    number = 1  # number of processed scan
    init_scan = scans[0]
    start_time = init_scan.scan_time[0]
    for mz, i in zip(init_scan.mz, init_scan.i):
        process_ROIs.append(ProcessROI([1, 1],
                                       [start_time, start_time],
                                       [i],
                                       [mz],
                                       mz))
    mzmean = np.copy(init_scan.mz)

    for scan in tqdm(scans):
        if number == 1:  # already processed scan
            number += 1
            continue
        # expand ROI
        for n, mz in enumerate(scan.mz):
            pos = np.searchsorted(mzmean, mz)
            closest = get_closest(mzmean, mz, pos)
            if abs(process_ROIs[closest].mzmean - mz) < delta_mz:
                roi = process_ROIs[closest]
                if roi.scan[1] == number:
                    # ROIs is already extended (two peaks in one mz window)
                    roi.mzmean = (roi.mzmean * roi.points + mz) / (roi.points + 1)
                    mzmean[closest] = roi.mzmean
                    roi.points += 1
                    roi.mz[-1] = (roi.i[-1]*roi.mz[-1] + scan.i[n]*mz) / (roi.i[-1] + scan.i[n])
                    roi.i[-1] = (roi.i[-1] + scan.i[n])
                else:
                    roi.mzmean = (roi.mzmean * roi.points + mz) / (roi.points + 1)
                    mzmean[closest] = roi.mzmean
                    roi.points += 1
                    roi.mz.append(mz)
                    roi.i.append(scan.i[n])
                    roi.scan[1] = number  # show that we extended the roi
                    roi.rt[1] = scan.scan_time[0]
            else:
                time = scan.scan_time[0]
                process_ROIs.insert(pos, ProcessROI([number, number],
                                       [time, time],
                                       [scan.i[n]],
                                       [mz],
                                       mz))
                mzmean = np.insert(mzmean, pos, mz)
        # Check and cleanup
        to_delete = []
        for n, roi in enumerate(process_ROIs):
            if roi.scan[1] < number <= roi.scan[1] + dropped_points:
                # insert 'zero' in the end
                roi.mz.append(roi.mzmean)
                roi.i.append(0)
            elif roi.scan[1] != number:
                to_delete.append(n)
                if roi.points >= required_points:
                    ROIs.append(ROI(
                        roi.scan,
                        roi.rt,
                        roi.i,
                        roi.mz,
                        roi.mzmean
                    ))
        for n in to_delete[::-1]:
            process_ROIs.pop(n)
        mzmean = np.delete(mzmean, to_delete)
        number += 1
        if pbar is not None:
            pbar.setValue(int(100 * number / len(scans)))
    # expand constructed roi
    for roi in ROIs:
        for n in range(dropped_points):
            # insert in the begin
            roi.i.insert(0, 0)
            roi.mz.insert(0, roi.mzmean)
        # change scan numbers (necessary for future matching)
        roi.scan = (roi.scan[0] - dropped_points, roi.scan[1] + dropped_points)
    return ROIs
