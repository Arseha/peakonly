import sys
import torch
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm
from processing_utils.roi import get_ROIs
from models.models import Classifier, Integrator


def preprocess(signal, points=256):
    interpolate = interp1d(np.arange(len(signal)), signal, kind='linear')
    signal = interpolate(np.arange(points) / (points - 1) * (len(signal) - 1))
    signal = torch.tensor(signal / np.max(signal), dtype=torch.float32)
    return signal.view(1, 1, -1)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('''Run script in the following format:
    python3 run.py path_to_file delta_mz roi_minimum_points peak_minimum_points''')
        exit()
    path = sys.argv[1]
    delta_mz = float(sys.argv[2])
    required_points = int(sys.argv[3])
    peak_minimum_points = int(sys.argv[4])
    ROIs = get_ROIs(path, delta_mz, required_points)  # get ROIs from raw spectrum

    # load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classify = Classifier()
    classify.load_state_dict(torch.load('data/Classifier', map_location=device))
    classify.eval()
    integrate = Integrator()
    integrate.load_state_dict(torch.load('data/Integrator', map_location=device))
    integrate.eval()

    threshold = 0.5
    split_threshold = 0.95
    # prob_threshold = 0.9
    points = 256
    peaks = {'mz': [], 'mzmin': [], 'mzmax': [], 'rtmin': [],
             'rtmax': [], 'intensity': []}
    for roi in tqdm(ROIs):
        signal = preprocess(roi.i)
        proba = classify(signal)[0].softmax(0)
        label = int(np.argmax(proba.cpu().detach().numpy()))
        if label == 1:  # ROI contains at least one peak
            logits = integrate(signal).sigmoid()
            splitter = logits[0, 0, :]
            domain = (((1 - splitter) * logits[0, 1, :]) > threshold).cpu().detach().numpy()
            splitter = splitter.cpu().detach().numpy()
            # create peaks
            process_peaks = {'mz': [], 'mzmin': [], 'mzmax': [], 'rtmin': [],
                             'rtmax': [], 'intensity': [], 'begins': [], 'ends': []}
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
                        process_peaks['begins'].append(begin)
                        process_peaks['ends'].append(n + 2)
                        begin = np.max((int((begin + 1) * len(roi.i) // points - 1), 0))
                        end = int((n + 2) * len(roi.i) // points - 1) + 1
                        process_peaks['intensity'].append(np.sum(roi.i[begin:end]))
                        process_peaks['mz'].append(np.mean(roi.mz[begin:end]))
                        process_peaks['mzmin'].append(np.min(roi.mz[begin:end]))
                        process_peaks['mzmax'].append(np.max(roi.mz[begin:end]))
                        process_peaks['rtmin'].append(roi.rt[0] + (roi.rt[1] - roi.rt[0]) / len(roi.i) * begin)
                        process_peaks['rtmax'].append(roi.rt[0] + (roi.rt[1] - roi.rt[0]) / len(roi.i) * (end - 1))
                    begin = -1
                    peak_wide = 0
            # delete the smallest peak if there is no splitter between them
            n = 0
            while n < number_of_peaks - 1:
                if not np.any(splitter[process_peaks['ends'][n]:process_peaks['begins'][n + 1]] > split_threshold):
                    smallest = n if process_peaks['intensity'][n] < process_peaks['intensity'][n + 1] else n + 1
                    for _, v in process_peaks.items():
                        v.pop(smallest)
                    number_of_peaks -= 1
                else:
                    n += 1
            # add process_peaks into peaks
            for key in peaks:
                peaks[key].extend(process_peaks[key])
    
    df = pd.DataFrame.from_dict(peaks)
    df.to_csv('../results.csv')
    print('Processing is done!')
