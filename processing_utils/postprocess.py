import pymzml
import numpy as np
import pandas as pd
from tqdm import tqdm


class ResultTable:
    def __init__(self, files, features):
        n_features = len(features)
        n_files = len(files)
        self.files = {k: v for v, k in enumerate(files)}
        self.intensities = np.zeros((n_files, n_features))
        self.mz = np.zeros(n_features)
        self.rtmin = np.zeros(n_features)
        self.rtmax = np.zeros(n_features)
        # fill in intensities values
        for i, feature in enumerate(features):
            self.mz[i] = feature.mz
            self.rtmin[i] = feature.rtmin
            self.rtmax[i] = feature.rtmax
            for j, sample in enumerate(feature.samples):
                self.intensities[self.files[sample], i] = feature.intensities[j]

    def fill_zeros(self, delta_mz):
        print('zero filling...')
        for file, k in tqdm(self.files.items()):
            # read all scans in mzML file
            run = pymzml.run.Reader(file)
            scans = []
            for scan in run:
                scans.append(scan)

            begin_time = scans[0].scan_time[0]
            end_time = scans[-1].scan_time[0]
            frequency = len(scans) / (end_time - begin_time)
            for m, intensity in enumerate(self.intensities[k]):
                if intensity == 0:
                    mz = self.mz[m]
                    begin = int((self.rtmin[m] - begin_time) * frequency) - 1
                    end = int((self.rtmax[m] - begin_time) * frequency) + 1
                    for scan in scans[begin:end]:
                        pos = np.searchsorted(scan.mz, mz)
                        if pos < len(scan.mz) and mz - delta_mz < scan.mz[pos] < mz + delta_mz:
                            self.intensities[k, m] += scan.i[pos]
                        if pos >= 1 and mz - delta_mz < scan.mz[pos - 1] < mz + delta_mz:
                            self.intensities[k, m] += scan.i[pos - 1]

    def to_csv(self, path):
        df = pd.DataFrame()
        df['mz'] = self.mz
        df['rtmin'] = self.rtmin / 60
        df['rtmax'] = self.rtmax / 60
        for file, k in self.files.items():
            df[file] = self.intensities[k]
        df.to_csv(path)
