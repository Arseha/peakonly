import pymzml
import numpy as np
from tqdm import tqdm


class ROI:
    def __init__(self, scan, rt, i, mz, mzmean):
        self.scan = scan
        self.rt = rt
        self.i = i
        self.mz = mz
        self.mzmean = mzmean


# TO DO: rewrite using BST ('bintrees' package should be nice or implement it in Cython)
# TO DO: When use scan.scan_time_in_minutes() every time gets new value???
# (problem solved: don't use scan_time_in_minutes() :))
def get_ROIs(path, delta_mz=0.005, required_points=15, dropped_points=3):
    '''
    :param path: path to mzml file
    :param delta_mz:
    :param required_points:
    :param dropped_points: can be zero points
    :return: ROIs - a list of ROI objects found in current file
    '''
    # read all scans in mzML file
    run = pymzml.run.Reader(path)
    scans = []
    for scan in run:
        scans.append(scan)

    ROIs = []  # completed ROIs

    # TO DO: rewrite this part (till the end :))
    # initialize a processed data
    number = 1  # number of processed scan
    init_scan = scans[0]
    ROI_mzmean = np.copy(init_scan.mz)
    ROI_begin = np.ones_like(ROI_mzmean)
    ROI_end = np.ones_like(ROI_mzmean)
    ROI_RTbegin = np.zeros_like(ROI_mzmean)
    ROI_RTend = np.zeros_like(ROI_mzmean)
    ROI_Npoints = np.ones_like(ROI_mzmean)  # number of points in ROI (zeros excluded)

    ROI_mz = []
    for mz in init_scan.mz:
        ROI_mz.append([mz])

    ROI_intens = []
    for i in init_scan.i:
        ROI_intens.append([i])

    for scan in tqdm(scans):
        number += 1
        # expand ROI
        for n, mz in enumerate(scan.mz):
            pos = np.searchsorted(ROI_mzmean, mz)
            if pos - 1 >= 0 and np.abs(ROI_mzmean[pos - 1] - mz) < delta_mz:
                if ROI_end[pos - 1] == number:
                    # ROIs is already extended (two peaks in one mz window)
                    ROI_mz[pos - 1][-1] = (ROI_mz[pos - 1][-1] + mz) / 2
                    ROI_mzmean[pos - 1] = np.mean(ROI_mz[pos - 1])  # TO DO: recalculate mean
                    ROI_intens[pos - 1][-1] = (ROI_intens[pos - 1][-1] + scan.i[n])  # TO DO: sum or mean?
                else:
                    ROI_mz[pos - 1].append(mz)
                    ROI_mzmean[pos - 1] = np.mean(ROI_mz[pos - 1])  # TO DO: recalculate mean
                    ROI_intens[pos - 1].append(scan.i[n])
                    ROI_end[pos - 1] = number  # show that we extended the roi
                    ROI_RTend[pos - 1] = scan.scan_time[0]
                    ROI_Npoints[pos - 1] += 1
            elif pos < len(ROI_mzmean) and np.abs(ROI_mzmean[pos] - mz) < delta_mz:
                if ROI_end[pos] == number:
                    # ROIs is already extended (two peaks in one mz window)
                    ROI_mz[pos][-1] = (ROI_mz[pos][-1] + mz) / 2
                    ROI_mzmean[pos] = np.mean(ROI_mz[pos])  # TO DO: recalculate mean
                    ROI_intens[pos][-1] = (ROI_intens[pos][-1] + scan.i[n])  # TO DO: sum or mean?
                else:
                    ROI_mz[pos].append(mz)
                    ROI_mzmean[pos] = np.mean(ROI_mz[pos])  # TO DO: recalculate mean
                    ROI_intens[pos].append(scan.i[n])
                    ROI_end[pos] = number  # show that we extended that roi
                    ROI_RTend[pos] = scan.scan_time[0]
                    ROI_Npoints[pos] += 1
            else:
                ROI_mz.insert(pos, [mz])
                ROI_mzmean = np.insert(ROI_mzmean, pos, mz)
                ROI_intens.insert(pos, [scan.i[n]])
                ROI_begin = np.insert(ROI_begin, pos, number)
                ROI_end = np.insert(ROI_end, pos, number)
                ROI_RTbegin = np.insert(ROI_RTbegin, pos, scan.scan_time[0])
                ROI_RTend = np.insert(ROI_RTend, pos, scan.scan_time[0])
                ROI_Npoints = np.insert(ROI_Npoints, pos, 1)

        # Check and cleanup
        to_delete = []
        for n, end in enumerate(ROI_end):
            if end < number and end + dropped_points >= number:
                # insert in the end
                ROI_mz[n].append(ROI_mzmean[n])
                ROI_intens[n].append(0)
            elif end != number:
                to_delete.append(n)
                if ROI_Npoints[n] >= required_points:
                    ROIs.append(ROI(
                        [ROI_begin[n], ROI_end[n]],
                        [ROI_RTbegin[n], ROI_RTend[n]],
                        ROI_intens[n],
                        ROI_mz[n],
                        ROI_mzmean[n]
                    ))
        for n in to_delete[::-1]:
            ROI_mz.pop(n)
            ROI_intens.pop(n)
        ROI_mzmean = np.delete(ROI_mzmean, to_delete)
        ROI_begin = np.delete(ROI_begin, to_delete)
        ROI_end = np.delete(ROI_end, to_delete)
        ROI_RTbegin = np.delete(ROI_RTbegin, to_delete)
        ROI_RTend = np.delete(ROI_RTend, to_delete)
        ROI_Npoints = np.delete(ROI_Npoints, to_delete)

    # TO DO: solve problem with "ParseError: junk after document element"
    # TO DO: some code available in draft.py (problem solved saving all the scans in 'scans')
    # expand constructed roi
    for roi in ROIs:
        for n in range(dropped_points):
            # insert in the begin
            roi.i.insert(0, 0)
            roi.mz.insert(0, roi.mzmean)
    return ROIs
