import os
import json
from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d


# to do: Reflection should take a ROI (dict)
class Reflection:
    """
    class that just reflects any signal
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, signal):
        if np.random.choice([True, False], p=[self.p, 1 - self.p]):
            signal = signal[::-1]
        return signal


class ROIDataset(Dataset):
    """
    A dataset for a training
    """
    def __init__(self, path, device, interpolate=False, adaptive_interpolate=False,
                 length=None, augmentations=None, balanced=False, return_roi_code=False):
        """
        :param path: a path to annotated ROIs
        :param device: a device where training will occur (GPU / CPU)
        :param interpolate: bool, if interpolation is needed
        :param adaptive_interpolate: to do: add interpolation to the closest power of 2
        :param length: only needed if 'interpolate' is True
        :param augmentations: roi augmantations
        :param balanced: bool, noise and peaks are returned 50/50
        :param return_roi_code: explicitly return the code of the roi
        """
        super().__init__()
        self.balanced = balanced
        self.device = device
        self.data = {0: [], 1: []}  # a dict from label2roi
        self.interpolate = interpolate
        self.adaptive_interpolate = interpolate
        self.length = length
        self.return_roi_code = return_roi_code
        for file in os.listdir(path):
            with open(os.path.join(path, file)) as json_file:
                roi = json.load(json_file)
                roi['intensity'] = np.array(roi['intensity'])
                roi['borders'] = np.array(roi['borders'])
                if self.interpolate:
                    roi = self._interpolate(roi)

                self.data[roi['label']].append(roi)
        self.augmentations = [] if augmentations is None else augmentations

    def __len__(self):
        if self.balanced:
            return min(len(self.data[0]), len(self.data[1]))
        else:
            return len(self.data[0]) + len(self.data[1])

    @staticmethod
    def _get_mask(roi):
        integration_mask = np.zeros_like(roi['intensity'])
        if roi['number of peaks'] >= 1:
            for b, e in roi['borders']:
                integration_mask[int(b):int(e)] = 1

        intersection_mask = np.zeros_like(roi['intensity'])
        if roi['number of peaks'] >= 2:
            for e, b in zip(roi['borders'][:-1, 1], roi['borders'][1:, 0]):
                if b - e > 5:
                    intersection_mask[e + 1:b] = 1
                else:
                    intersection_mask[e - 1:b + 2] = 1
        return integration_mask, intersection_mask

    def _interpolate(self, roi):
        roi = deepcopy(roi)
        points = len(roi['intensity'])
        interpolate = interp1d(np.arange(points), roi['intensity'], kind='linear')
        roi['intensity'] = interpolate(np.arange(self.length) / (self.length - 1.) * (points - 1.))
        roi['borders'] = np.array(roi['borders'])
        roi['borders'] = roi['borders'] * (self.length - 1) // (points - 1)
        return roi

    def __getitem__(self, idx):
        if self.balanced:
            roi = np.random.choice(self.data[idx % 2])
        else:
            roi = self.data[0][idx] if idx < len(self.data[0]) else self.data[1][idx - len(self.data[0])]

        for aug in self.augmentations:
            roi = deepcopy(roi)
            roi = aug(roi)

        x = roi['intensity']
        x = torch.tensor(x, dtype=torch.float32, device=self.device).view(1, -1)
        x = x / torch.max(x)
        y = torch.tensor(roi['label'], dtype=torch.long, device=self.device)

        integration_mask, intersection_mask = self._get_mask(roi)
        integration_mask = torch.tensor(integration_mask, dtype=torch.float32, device=self.device)
        intersection_mask = torch.tensor(intersection_mask, dtype=torch.float32, device=self.device)

        if self.return_roi_code:
            original_length = len(roi['mz'])
            return x, y, integration_mask, intersection_mask, roi['code'], original_length

        return x, y, integration_mask, intersection_mask
