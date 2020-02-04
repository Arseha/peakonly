import unittest
import numpy as np
from collections import defaultdict

from utils.matching import stitch_component, align_component
from utils.roi import ROI


class MyTestCase(unittest.TestCase):
    def test_stitch_component(self):
        component = defaultdict(list)
        roi1 = ROI(scan=(48, 54),
                   rt=(48, 54),
                   i=[1.]*7,
                   mz=[100] * 7,
                   mzmean=100)
        roi2 = ROI(scan=(25, 30),
                   rt=(25, 30),
                   i=list(np.random.randn(6)),
                   mz=[100] * 6,
                   mzmean=100)
        component['sample'] = [roi1, roi2]
        new_component = stitch_component(component)
        self.assertEqual(roi1.i, list(new_component['sample'][0].i[-7:]))
        self.assertEqual(roi2.i, list(new_component['sample'][0].i[:6]))

    def test_align_component_simple(self):
        component = defaultdict(list)
        roi1_sample1 = ROI(scan=(-5, 1),
                           rt=(5, 10),
                           i=[0, 2, 2, 0, 1, 1, 0],
                           mz=[105] * 7,
                           mzmean=105)
        component['sample1'] = [roi1_sample1]

        roi1_sample2 = ROI(scan=(-6, 0),
                           rt=(5, 10),
                           i=[1, 1, 0, 0.9, 0.8, 0, 0],
                           mz=[105] * 7,
                           mzmean=105)
        component['sample2'] = [roi1_sample2]
        group = align_component(component)
        shifts = dict()
        for sample, shift in zip(group.samples, group.shifts):
            shifts[sample] = shift
        self.assertEqual({'sample1': 0, 'sample2': 2}, shifts)

    def test_align_component_complex(self):
        component = defaultdict(list)
        roi1_sample1 = ROI(scan=(11, 20),
                           rt=(11, 20),
                           i=[0] + [1, 2, 3, 4, 3, 2, 1] + [0] * 2,
                           mz=[100] * 10,
                           mzmean=100)
        component['sample1'] = [roi1_sample1]

        roi1_sample2 = ROI(scan=(12, 19),
                           rt=(12, 19),
                           i=[0] + [1, 2, 3, 2, 3, 2, 1],
                           mz=[100] * 8,
                           mzmean=100)
        roi2_sample2 = ROI(scan=(25, 31),
                           rt=(25, 31),
                           i=[0] + [1, 1, 1, 1, 1] + [0],
                           mz=[100] * 7,
                           mzmean=100)
        component['sample2'] = [roi1_sample2, roi2_sample2]

        roi1_sample3 = ROI(scan=(10, 28),
                           rt=(10, 28),
                           i=[0] + [1, 2, 3, 3, 3, 2, 1] + [0] * 3 + [1, 1, 1, 1, 1] + [0]*3,
                           mz=[100] * 19,
                           mzmean=100)
        component['sample3'] = [roi1_sample3]

        group = align_component(component)
        shifts = dict()
        for sample, shift in zip(group.samples, group.shifts):
            shifts[sample] = shift
        self.assertEqual({'sample1': 0, 'sample2': -1, 'sample3': 1}, shifts)

    def test_align_component_strange(self):
        component = defaultdict(list)
        roi1_sample1 = ROI(scan=(10, 20),
                           rt=(10, 20),
                           i=np.random.randn(11),
                           mz=[100] * 11,
                           mzmean=100)
        component['sample1'] = [roi1_sample1]

        roi1_sample2 = ROI(scan=(100, 108),
                           rt=(100, 108),
                           i=np.random.randn(9)+10,
                           mz=[100] * 9,
                           mzmean=100)
        component['sample2'] = [roi1_sample2]

        group = align_component(component)
        shifts = dict()
        for sample, shift in zip(group.samples, group.shifts):
            shifts[sample] = shift
        self.assertEqual([0, 0], group.shifts)


if __name__ == '__main__':
    unittest.main()
