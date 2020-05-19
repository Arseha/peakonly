# distutils: language = c++
import pymzml
from libcpp.map cimport map
from libcpp.vector cimport vector
from cython.operator cimport dereference, postincrement, postdecrement
from processing_utils.roi import ROI

cdef struct cROI:
    int scan_begin
    int scan_end
    float rt_begin
    float rt_end
    vector[float] i
    vector[float] mz
    float mz_mean
    int points  # calculate number of non_zero points
    
cdef struct MsScan:
    vector[float] i
    vector[float] mz
    float rt
    

def get_ROIs(str path, float delta_mz=0.005, int required_points=15, int dropped_points=3, progress_callback=None):
    # read all scans in mzML file
    run = pymzml.run.Reader(path)
    cdef vector[MsScan] scans
    for scan in run:
        if scan.ms_level == 1:
            scans.push_back(MsScan(scan.i, scan.mz, scan.scan_time[0]))
            
    cdef vector[cROI] rois  # completed ROIs (vector)
    cdef map[float, cROI] process_rois  # processing ROIs (map)
    
    # initialize a processed data
    cdef MsScan init_scan = scans[0]
    cdef float start_time = init_scan.rt

    cdef float min_mz = min(init_scan.mz)
    cdef float max_mz = max(init_scan.mz)
    cdef cROI new_roi
    for n in range(init_scan.i.size()):
        if init_scan.i[n] != 0:
            new_roi = cROI(0, 0, start_time, start_time, vector[float](),
                          vector[float](), init_scan.mz[n], 1)
            new_roi.i.push_back(init_scan.i[n])
            new_roi.mz.push_back(init_scan.mz[n])
            process_rois[init_scan.mz[n]] = new_roi
            
    
    cdef float ceiling_mz  # the closest m/z not less than the given
    cdef cROI* ceiling
    cdef map[float, cROI].iterator ceiling_it
    cdef float floor_mz  #  the closest m/z not greater than the given
    cdef cROI* floor
    cdef map[float, cROI].iterator floor_it
    cdef float closest_mz  # the closest m/z
    cdef cROI* closest

    cdef cROI* roi
    cdef float mz
    cdef MsScan current_scan
    
    cdef map[float, cROI].iterator map_it
    
    for number in range(1, scans.size()):
        current_scan = scans[number]
        for n in range(current_scan.i.size()):
            if current_scan.i[n] != 0:
                mz = current_scan.mz[n]
                # find ceiling_it and floor_it
                floor_it = process_rois.lower_bound(mz)
                if floor_it != process_rois.end() and floor_it != process_rois.begin():
                    ceiling_it = postdecrement(floor_it)
                else:
                    ceiling_it = floor_it
                    floor_it = process_rois.end()
                # get ceiling and floor
                if ceiling_it != process_rois.end():
                    ceiling = &dereference(ceiling_it).second
                    ceiling_mz = ceiling.mz_mean
                if floor_it != process_rois.end():
                    floor = &dereference(floor_it).second
                    floor_mz = floor.mz_mean
                # getting closest roi (if possible)
                if ceiling_it == process_rois.end() and floor_it == process_rois.end():  # process_rois is empty?
                    new_roi = cROI(number, number, current_scan.rt, current_scan.rt, vector[float](),
                              vector[float](), mz, 1)
                    new_roi.i.push_back(current_scan.i[n])
                    new_roi.mz.push_back(current_scan.mz[n])
                    process_rois[mz] = new_roi
                elif ceiling_it == process_rois.end():
                    closest_mz = floor_mz
                    closest = floor
                elif floor_it == process_rois.end():
                    closest_mz = ceiling_mz
                    closest = ceiling
                else:
                    if ceiling_mz - mz > mz - floor_mz:
                        closest_mz = floor_mz
                        closest = floor
                    else:
                        closest_mz = ceiling_mz
                        closest = ceiling
                # expanding existing roi or creates a new one
                if abs(closest_mz - mz) < delta_mz:
                    roi = closest
                    if roi.scan_end == number:
                        # ROIs is already extended (two peaks in one mz window) (almost not possible)
                        roi.mz_mean = 0.9 * roi.mz_mean + 0.1 * mz
                        roi.points += 1
                        roi.mz[roi.mz.size() - 1] = ((roi.i[roi.mz.size() - 1]*roi.mz[roi.mz.size() - 1] +
                                                      current_scan.i[n]*mz) / (roi.i[roi.mz.size() - 1]
                                                                               + current_scan.i[n]))
                        roi.i[roi.i.size() - 1] = roi.i[roi.i.size() - 1] + current_scan.i[n]
                    else:
                        roi.mz_mean = 0.9 * roi.mz_mean + 0.1 * mz
                        roi.points += 1
                        roi.mz.push_back(mz)
                        roi.i.push_back(current_scan.i[n])
                        roi.scan_end = number 
                        roi.rt_end = current_scan.rt
                else:
                    new_roi = cROI(number, number, current_scan.rt, current_scan.rt, vector[float](),
                              vector[float](), mz, 1)
                    new_roi.i.push_back(current_scan.i[n])
                    new_roi.mz.push_back(current_scan.mz[n])
                    process_rois[mz] = new_roi
        # Check and cleanup
        map_it = process_rois.begin()
        while map_it != process_rois.end():
            roi = &dereference(map_it).second
            mz = roi.mz_mean
            if roi.scan_end < number <= roi.scan_end + dropped_points:
                # insert 'zero' in the end
                roi.mz.push_back(mz)
                roi.i.push_back(0)
                postincrement(map_it)
            elif roi.scan_end != number:
                if roi.points >= required_points:
                    new_roi = dereference(map_it).second
                    rois.push_back(new_roi)
                process_rois.erase(postincrement(map_it))
            else:
                postincrement(map_it)
        if progress_callback is not None and not number % 10:
            progress_callback.emit(int(number * 100 / scans.size()))
    # add final rois
    map_it = process_rois.begin()
    while map_it != process_rois.end():
        roi = &dereference(map_it).second
        if roi.points >= required_points:
            for n in range(dropped_points - (scans.size() - 1 - roi.scan_end)):
                roi.mz.push_back(roi.mz_mean)
                roi.i.push_back(0)
            rois.push_back(dereference(roi))
        postincrement(map_it)
    # expand constructed roi and creating python object
    cdef vector[cROI].iterator roi_it = rois.begin()
    python_rois = []
    while roi_it != rois.end():
        roi = &dereference(roi_it)
        roi.i.insert(roi.i.begin(), dropped_points, 0)
        roi.mz.insert(roi.mz.begin(), dropped_points, roi.mz_mean)
        # change scan numbers (necessary for future matching)
        roi.scan_begin = roi.scan_begin - dropped_points
        roi.scan_end = roi.scan_end + dropped_points

        python_rois.append(ROI([roi.scan_begin, roi.scan_end], [roi.rt_begin, roi.rt_end],
                               roi.i, roi.mz, roi.mz_mean))
        postincrement(roi_it)
    return python_rois
