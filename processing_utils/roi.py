import json
import pymzml
import numpy as np
from tqdm import tqdm
from bintrees import FastAVLTree


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

    def save_annotated(self, path, code=None, label=0, number_of_peaks=0, peaks_labels=None, borders=None,
                             description=None):
        roi = dict()
        roi['code'] = code
        roi['label'] = label
        roi['number of peaks'] = number_of_peaks
        roi["peaks' labels"] = [] if peaks_labels is None else peaks_labels
        roi['borders'] = [] if borders is None else borders
        roi['description'] = description

        roi['rt'] = self.rt
        roi['scan'] = self.scan
        roi['intensity'] = list(map(float, self.i))
        roi['mz'] = list(map(float, self.mz))

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
        if scan.ms_level == 1:
            scans.append(scan)

    ROIs = []  # completed ROIs
    process_ROIs = FastAVLTree()  # processed ROIs

    # initialize a processed data
    number = 1  # number of processed scan
    init_scan = scans[0]
    start_time = init_scan.scan_time[0]

    min_mz = max(init_scan.mz)
    max_mz = min(init_scan.mz)
    for mz, i in zip(init_scan.mz, init_scan.i):
        if i != 0:
            process_ROIs[mz] = ProcessROI([1, 1],
                                          [start_time, start_time],
                                          [i],
                                          [mz],
                                          mz)
            min_mz = min(min_mz, mz)
            max_mz = max(max_mz, mz)

    for scan in tqdm(scans):
        if number == 1:  # already processed scan
            number += 1
            continue
        # expand ROI
        for n, mz in enumerate(scan.mz):
            if scan.i[n] != 0:
                ceiling_mz, ceiling_item = None, None
                floor_mz, floor_item = None, None
                if mz < max_mz:
                    _, ceiling_item = process_ROIs.ceiling_item(mz)
                    ceiling_mz = ceiling_item.mzmean
                if mz > min_mz:
                    _, floor_item = process_ROIs.floor_item(mz)
                    floor_mz = floor_item.mzmean
                # choose closest
                if ceiling_mz is None and floor_mz is None:
                    continue
                elif ceiling_mz is None:
                    closest_mz, closest_item = floor_mz, floor_item
                elif floor_mz is None:
                    closest_mz, closest_item = ceiling_mz, ceiling_item
                else:
                    if ceiling_mz - mz > mz - floor_mz:
                        closest_mz, closest_item = floor_mz, floor_item
                    else:
                        closest_mz, closest_item = ceiling_mz, ceiling_item

                if abs(closest_item.mzmean - mz) < delta_mz:
                    roi = closest_item
                    if roi.scan[1] == number:
                        # ROIs is already extended (two peaks in one mz window)
                        roi.mzmean = (roi.mzmean * roi.points + mz) / (roi.points + 1)
                        roi.points += 1
                        roi.mz[-1] = (roi.i[-1]*roi.mz[-1] + scan.i[n]*mz) / (roi.i[-1] + scan.i[n])
                        roi.i[-1] = (roi.i[-1] + scan.i[n])
                    else:
                        roi.mzmean = (roi.mzmean * roi.points + mz) / (roi.points + 1)
                        roi.points += 1
                        roi.mz.append(mz)
                        roi.i.append(scan.i[n])
                        roi.scan[1] = number  # show that we extended the roi
                        roi.rt[1] = scan.scan_time[0]
                else:
                    time = scan.scan_time[0]
                    process_ROIs[mz] = ProcessROI([number, number],
                                                  [time, time],
                                                  [scan.i[n]],
                                                  [mz],
                                                  mz)
        # Check and cleanup
        to_delete = []
        for mz, roi in process_ROIs.items():
            if roi.scan[1] < number <= roi.scan[1] + dropped_points:
                # insert 'zero' in the end
                roi.mz.append(roi.mzmean)
                roi.i.append(0)
            elif roi.scan[1] != number:
                to_delete.append(mz)
                if roi.points >= required_points:
                    ROIs.append(ROI(
                        roi.scan,
                        roi.rt,
                        roi.i,
                        roi.mz,
                        roi.mzmean
                    ))
        process_ROIs.remove_items(to_delete)
        min_mz, _ = process_ROIs.min_item()
        max_mz, _ = process_ROIs.max_item()
        number += 1
    # add final rois
    for mz, roi in process_ROIs.items():
        if roi.points >= required_points:
            for n in range(dropped_points - (number - 1 - roi.scan[1])):
                # insert 'zero' in the end
                roi.mz.append(roi.mzmean)
                roi.i.append(0)
            ROIs.append(ROI(
                        roi.scan,
                        roi.rt,
                        roi.i,
                        roi.mz,
                        roi.mzmean
                        ))
    # expand constructed roi
    for roi in ROIs:
        for n in range(dropped_points):
            # insert in the begin
            roi.i.insert(0, 0)
            roi.mz.insert(0, roi.mzmean)
        # change scan numbers (necessary for future matching)
        roi.scan = (roi.scan[0] - dropped_points, roi.scan[1] + dropped_points)
        assert roi.scan[1] - roi.scan[0] == len(roi.i) - 1
    return ROIs
