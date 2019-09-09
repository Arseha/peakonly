import os
import json
from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d


class PeakGenerator:
    def __init__(self, sigma=None):
        '''
        Class for generation of assymetrical peaks (as two parts of gauss)
        sigma - range for min/max dispersion for gauss
        '''
        if sigma is None:
            sigma = [3, 16]
        self.sigma = sigma

    def __call__(self):
        '''
        Generates assymetrical peak, returns peak
        '''
        sigma_up = np.random.uniform(low=self.sigma[0], high=self.sigma[1])
        sigma_down = np.random.uniform(low=self.sigma[0], high=self.sigma[1])

        a = int(3 * sigma_up)  # the summit of the peak
        points = 3 * sigma_up + 3 * sigma_down
        x = np.arange(points)
        peak = np.zeros_like(x)
        peak[:a] = np.exp(-(x[:a] - a) ** 2 / (2 * sigma_up ** 2))
        peak[a:] = np.exp(-(x[a:] - a) ** 2 / (2 * sigma_down ** 2))
        return peak


class ROIGenerator:
    def __init__(self, mode='classification', length=256,
                 sigma=None, max_peaks=None):
        '''
        Generates data for model training.
        mode - 'classification'/'segmentation'/'detection'
        length - total number of points in spectrum
        sigma - range for min/max dispersion for gauss
        noise_prob - probability of noise
        max_peaks - maximum number of peaks in spectrum
        '''
        if sigma is None:
            sigma = [3, 16]
        self.sigma = sigma
        self.mode = mode
        self.length = length
        self.peak_gen = PeakGenerator(sigma)
        self.max_peaks = max_peaks

    def __call__(self):
        '''
        Generates annotated ROI
        '''
        if self.mode == 'classification':
            label = np.random.choice([0, 1, 2], p=[3 / 8, 3 / 8, 2 / 8])
            gen_p = 0.4
            bad_shape_p = 0.5
            double_p = 0
        elif self.mode == 'integration':
            label = 1
            gen_p = 0.5
            bad_shape_p = 0.2
            double_p = 0.3

        ROI = {'label': label, 'number of peaks': 0, 'begins': [],
               'ends': [], 'intersections': [], 'intensity': []}
        if label == 0:  # 0 stands for noise
            points = np.random.randint(low=self.length // 16, high=self.length)
            ROI['intensity'] = [0.] * points
            self._postprocessing(ROI)
        elif label == 1:  # 1 stands for peak
            shape = np.random.choice(['bad', 'good', 'double'], p=[bad_shape_p, 1 - bad_shape_p - double_p, double_p])
            if shape == 'bad':
                shape = np.random.choice(['noise', 'small'])
                if shape == 'small':
                    ROI['number of peaks'] = 1
                    ROI['begins'] = [3]
                    points = np.random.randint(low=self.length // 16, high=self.length // 8)
                    ROI['ends'] = [points - 3]
                    sigma = (points - 6) / 6

                    x = np.arange(points)
                    ROI['intensity'] = np.exp(-(x - points / 2) ** 2 / (2 * sigma ** 2))

                    ROI['intensity'] -= 0.15 * np.random.randn(points)
                    ROI['intensity'] -= np.min(ROI['intensity'])
                    ROI['intensity'] /= np.max(ROI['intensity'])
                    ROI['intensity'][:2] = 0
                    ROI['intensity'][-2:] = 0
                    self._postprocessing(ROI, full=False)
                elif shape == 'noise':
                    ROI['number of peaks'] = 1
                    points = np.random.randint(low=self.length // 2, high=self.length)
                    sigma = points / 32
                    a = np.random.randint(low=sigma * 3 + 1, high=points - 1 - sigma * 3)
                    ROI['begins'] = [int(a - 3 * sigma)]
                    ROI['ends'] = [int(a + 3 * sigma)]

                    x = np.arange(points)
                    ROI['intensity'] = 0.2 * np.exp(-(x - points // 2) ** 2 / (points // 2) ** 2)  # background noise
                    ROI['intensity'] += np.exp(-(x - a) ** 2 / (2 * sigma ** 2))

                    ROI['intensity'] -= 0.15 * np.random.randn(points)
                    ROI['intensity'] -= np.min(ROI['intensity'])
                    ROI['intensity'] /= np.max(ROI['intensity'])
                    self._postprocessing(ROI, full=False)
            elif shape == 'double':  # double peaks
                ROI['number of peaks'] = 2
                # first peak
                up1 = np.random.uniform(low=self.sigma[0], high=self.sigma[1] // 2)
                down1 = np.random.uniform(low=self.sigma[0], high=self.sigma[1] // 2)
                a1 = np.random.uniform(low=up1 * 3.5, high=np.min((self.length - down1 * 3.5, 0.6 * self.length)))
                a1_point = int(a1)

                # second peak
                up2 = np.random.uniform(low=1, high=self.sigma[1] / 2)
                down2 = np.random.uniform(low=1, high=self.sigma[1] / 2)
                a2 = np.random.uniform(low=a1 + 1.5 * (down1 + up2), high=self.length - down2 * 3.5)
                a2_point = int(a2)
                alpha = np.random.uniform(low=0.5, high=2)

                # value generation
                x = np.arange(self.length)
                ROI['intensity'] = np.zeros_like(x, dtype='float32')
                ROI['intensity'][:a1_point] = np.exp(-(x[:a1_point] - a1) ** 2 / (2 * up1 ** 2)) + alpha * np.exp(
                    -(x[:a1_point] - a2) ** 2 / (2 * up2 ** 2))
                ROI['intensity'][a1_point:a2_point] = np.exp(-(x[a1_point:a2_point] - a1) ** 2 / (2 * down1 ** 2)) + alpha * np.exp(
                    -(x[a1_point:a2_point] - a2) ** 2 / (2 * up2 ** 2))
                ROI['intensity'][a2_point:] = np.exp(-(x[a2_point:] - a1) ** 2 / (2 * down1 ** 2)) + alpha * np.exp(
                    -(x[a2_point:] - a2) ** 2 / (2 * down2 ** 2))

                # noise
                noise_ampl = np.random.uniform(low=0.05, high=0.25)
                ROI['intensity'] += noise_ampl * np.random.randn(self.length) / 2
                ROI['intensity'] += np.abs(np.min(ROI['intensity']))
                ROI['intensity'] = ROI['intensity'] / np.max(ROI['intensity'])

                # intersection point and begin/end points
                ROI['intersections'] = [np.argmin(ROI['intensity'][a1_point:a2_point]) + a1_point]
                begin1 = int(a1 - 3*up1)
                if int(a2 - 3*up2) > int(a1 + 3*down1):
                    end1 = int(a1 + 3*down1)
                    begin2 = int(a2 - 3*up2)
                    ROI['intersections'] = [(begin2 - end1) // 2 + end1]
                else:
                    ROI['intersections'] = [np.argmin(ROI['intensity'][a1_point:a2_point]) + a1_point]
                    end1 = ROI['intersections'][0]
                    begin2 = ROI['intersections'][0]
                end2 = int(a2 + 3*down2)
                ROI['begins'] = [begin1, begin2]
                ROI['ends'] = [end1, end2]
                self._postprocessing(ROI, full=False)
            else:
                peaks = [self.peak_gen()]
                current_length = len(peaks[-1])
                gen = np.random.choice([True, False], p=[gen_p, 1 - gen_p])
                while gen and current_length < self.length:
                    peaks.append(self.peak_gen())
                    current_length += len(peaks[-1])
                    gen = np.random.choice([True, False], p=[gen_p, 1 - gen_p])

                # if length is too big
                if current_length > self.length:
                    current_length -= len(peaks[-1])
                    peaks.pop()

                # stitching
                self._stitching(ROI, peaks)
                self._postprocessing(ROI)
        else:  # 2 stands for 'hard to say'
            assert label == 2, 'label={}, but there is only 3 classes'.format(label)
            points = np.random.randint(low=self.length // 16, high=self.length // 8)
            shape = np.random.choice(['linear', 'flattened gauss'])
            if shape == 'linear':
                linear = interp1d([3, points - 3], [0, 1])
                ROI['intensity'] = np.zeros(points)
                ROI['intensity'][3:points - 3] = linear(np.arange(3, points - 3))

                ROI['intensity'][3:-3] += 0.2 * np.random.randn(points - 6)
                ROI['intensity'][ROI['intensity'] < 0] = 0
            elif shape == 'flattened gauss':
                sigma = (points - 6) / 6
                a = points // 2

                # create gauss
                x = np.arange(points)
                ROI['intensity'] = np.exp(-(x - a) ** 2 / (2 * sigma ** 2))

                begin = np.random.randint(low=int(a - 2.5 * sigma), high=int(a - 2 * sigma))
                end = np.random.randint(low=int(a + 2 * sigma), high=int(a + 2.5 * sigma))

                linear = interp1d([begin, end], [ROI['intensity'][begin], ROI['intensity'][end]])
                ROI['intensity'][begin:end] = linear(np.arange(begin, end))

                ROI['intensity'] /= np.max(ROI['intensity'])
                ROI['intensity'][3:-3] += 0.2 * np.random.randn(points - 6)
                ROI['intensity'][ROI['intensity'] < 0] = 0

            self._postprocessing(ROI)
        return ROI

    def _stitching(self, ROI, peaks):
        '''
        Stitches peaks into one ROI
        '''
        ROI['number of peaks'] = len(peaks)
        ROI['intensity'].extend(peaks[0])
        ROI['begins'].append(0)
        vertices = [np.argmax(peaks[0])]
        for m, peak in enumerate(peaks[1:]):
            tail_length = len(ROI['intensity']) - vertices[-1]
            shift = np.random.randint(
                low=int(2 / 3 * tail_length),
                high=int(4 / 3 * tail_length)
            )
            alpha = np.random.uniform(low=0.5, high=2)
            if shift >= tail_length:
                ROI['ends'].append(vertices[-1] + tail_length - 1)
                ROI['intersections'].append(int(ROI['ends'][-1] + (shift - tail_length) // 2))
                ROI['intensity'].extend([ROI['intensity'][-1]] * (shift - tail_length))
                ROI['begins'].append(len(ROI['intensity']))
                i_shift = alpha * peak[0] - ROI['intensity'][-1]
                for i in peak:
                    ROI['intensity'].append(alpha * i - i_shift)
            else:
                for n, i in enumerate(peak):
                    if n < tail_length - shift:
                        ROI['intensity'][vertices[-1] + shift + n] += alpha * i
                    else:
                        ROI['intensity'].append(alpha * i)
                ROI['intersections'].append(np.argmin(ROI['intensity'][vertices[-1]:vertices[-1] + tail_length]) +
                                            vertices[-1])
                ROI['ends'].append(ROI['intersections'][-1])
                ROI['begins'].append(ROI['intersections'][-1])
            vertices.append(np.argmax(ROI['intensity'][(vertices[-1] + shift):]) + vertices[-1] + shift)
        ROI['ends'].append(len(ROI['intensity']) - 1)

    def _postprocessing(self, ROI, full=True):
        '''
        noise generation, random shifting, normalization, peak suppression
        '''
        if full:
            if ROI['label'] == 0:
                noise_ampl = 0.25
                ROI['intensity'] = np.array(ROI['intensity'])
            elif ROI['label'] == 1:
                # random peak suppression
                if ROI['number of peaks'] > 1:
                    for n in range(ROI['number of peaks']):
                        to_do = False
                        if n == 0 and ROI['ends'][n] + 5 < ROI['begins'][n + 1]:  # first peak
                            to_do = np.random.choice([True, False], p=[0.7, 0.3])
                        elif n == ROI['number of peaks'] - 1 and ROI['ends'][n - 1] + 5 < ROI['begins'][n]:  # last peak
                            to_do = np.random.choice([True, False], p=[0.7, 0.3])
                        elif ROI['ends'][n - 1] + 5 < ROI['begins'][n] and ROI['ends'][n] + 5 < ROI['begins'][n + 1]:
                            to_do = np.random.choice([True, False], p=[0.7, 0.3])

                        if to_do:
                            suppress = np.random.choice(np.logspace(-2, 0, 3))
                            ROI['intensity'][ROI['begins'][n]:ROI['ends'][n] + 1] /= suppress

                min_alpha = 1
                for n in range(ROI['number of peaks']):
                    alpha = np.max(ROI['intensity'][ROI['begins'][n]:ROI['ends'][n] + 1])

                    min_alpha = min((alpha, min_alpha))
                noise_ampl = 0.1 * min_alpha

                # random shifting
                extra_points = np.random.randint(low=int(0.1 * len(ROI['intensity'])),
                                                 high=int(0.5 * len(ROI['intensity'])))
                on_the_edge = np.random.choice([True, False], p=[0.6, 0.4])  # peak begins fastly
                if on_the_edge:
                    shift = 0
                else:
                    shift = np.random.randint(low=0, high=extra_points)
                for n in range(ROI['number of peaks']):
                    ROI['begins'][n] += shift
                    ROI['ends'][n] += shift
                    if n + 1 < ROI['number of peaks']:
                        ROI['intersections'][n] += shift
                tmp_signal = np.zeros(len(ROI['intensity']) + extra_points)
                tmp_signal[shift:len(ROI['intensity']) + shift] = ROI['intensity']
                ROI['intensity'] = tmp_signal

            if ROI['label'] != 2:
                # noise generation and normalization
                ROI['intensity'] += noise_ampl * np.random.randn(len(ROI['intensity']))
                ROI['intensity'] /= np.max(ROI['intensity'])
                ROI['intensity'] = np.abs(ROI['intensity'])

        # interpolate to 'self.length' points
        points = len(ROI['intensity'])
        interpolate = interp1d(np.arange(points), ROI['intensity'], kind='linear')
        ROI['intensity'] = interpolate(np.arange(self.length) / (self.length - 1) * (points - 1))

        ROI['begins'] = np.array(ROI['begins'])
        ROI['ends'] = np.array(ROI['ends'])
        ROI['intersections'] = np.array(ROI['intersections'])
        ROI['begins'] = ROI['begins'] * (self.length - 1) // (points - 1)
        ROI['ends'] = ROI['ends'] * (self.length - 1) // (points - 1)
        ROI['intersections'] = ROI['intersections'] * (self.length - 1) // (points - 1)


def get_mask(ROI):
    mask = np.zeros_like(ROI['intensity'])
    if ROI['number of peaks'] > 1:
        for n, inter in enumerate(ROI['intersections']):
            if ROI['begins'][n + 1] - ROI['ends'][n] > 3:
                mask[ROI['ends'][n]:ROI['begins'][n + 1]] = 1
            else:
                mask[inter - 1:inter + 2] = 1

    integration = np.zeros_like(ROI['intensity'])
    if ROI['label'] == 1:
        for beg, end in zip(np.array(ROI['begins']), np.array(ROI['ends'])):
            integration[int(beg):int(end)] = 1
        if ROI['number of peaks'] > 1:
            for inter in ROI['intersections']:
                integration[inter - 1:inter + 2] = 0
    return (mask, integration), torch.float32


def get_label(ROI):
    return ROI['label'], torch.long


class Reflection:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, signal, mask):
        if np.random.choice([True, False], p=[self.p, 1-self.p]):
            signal = signal[::-1]
            mask = mask[0][::-1], mask[1][::-1]
        return signal, mask


class ROIDataset(Dataset):
    def __init__(self, path, key, mode='classification', length=256,
                 sigma=None, classes=None, augmentation=None, sizeof=2048, gen_p=0.8):
        super().__init__()
        # reading data into RAM
        if mode == 'classification':
            if classes is None:
                classes = (0, 1, 2)  # use all possible classes
            self.classes = classes
            self.data = dict()
            for label in classes:
                self.data[label] = []
            for file in os.listdir(path):
                with open(os.path.join(path, file)) as json_file:
                    roi = json.load(json_file)
                    if roi['label'] in classes:
                        self.data[roi['label']].append(roi)
        elif mode == 'integration':
            self.data = {'uni': [], 'multi': []}
            self.classes = ('uni', 'multi')
            for file in os.listdir(path):
                with open(os.path.join(path, file)) as json_file:
                    roi = json.load(json_file)
                    if roi['label'] == 1 and roi['number of peaks'] == 1:
                        self.data['uni'].append(roi)
                    elif roi['label'] == 1 and roi['number of peaks'] >= 2:
                        self.data['multi'].append(roi)
        else:
            assert False, 'mode should be one of twe: classification/segmentation'
        if gen_p == 0:
            self.data = [w for k, v in self.data.items() for w in v]
        # save needed parameters
        self.path = path
        self.key = key
        self.length = length
        self.augmentation = augmentation
        self.generator = ROIGenerator(mode=mode, sigma=sigma)
        self.sizeof = sizeof
        self.gen_p = gen_p

    def __len__(self):
        if self.gen_p == 0:
            return len(self.data)
        else:
            return self.sizeof

    def _process(self, roi):
        roi = deepcopy(roi)
        points = len(roi['intensity'])
        interpolate = interp1d(np.arange(points), roi['intensity'], kind='linear')
        roi['intensity'] = interpolate(np.arange(self.length) / (self.length - 1.) * (points - 1.))
        roi['begins'] = np.array(roi['begins'])
        roi['ends'] = np.array(roi['ends'])
        roi['intersections'] = np.array(roi['intersections'])
        roi['begins'] = roi['begins'] * (self.length - 1) // (points - 1)
        roi['ends'] = roi['ends'] * (self.length - 1) // (points - 1)
        roi['intersections'] = roi['intersections'] * (self.length - 1) // (points - 1)

        y, dtype = self.key(roi)
        x = roi['intensity']
        if self.augmentation:
            for aug in self.augmentation:
                x, y = aug(x, y)
        return x, y, dtype

    def __getitem__(self, idx):
        if self.gen_p == 0:
           data = self.data[idx]
           x, y, dtype = self._process(data)
        else:
            gen_on = np.random.binomial(1, self.gen_p)
            if gen_on:
                data = self.generator()
                y, dtype = self.key(data)
                x = data['intensity']
            else:
                label = np.random.choice(self.classes)
                data = np.random.choice(self.data[label])  # balanced classes
                x, y, dtype = self._process(data)
        x = torch.tensor(x / np.max(x), dtype=torch.float32).view(1, -1)
        y = torch.tensor(y, dtype=dtype)
        return x, y
