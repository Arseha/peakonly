import unittest
import numpy as np

from processing_utils import run_utils


class MyTestCase(unittest.TestCase):
    def test_intersection(self):
        interval_1 = [0., 100.]
        interval_2 = [101., 201.]
        interval_3 = [25., 125.]
        interval_4 = [25., 50.]
        # symmetry:
        self.assertEqual(run_utils.intersection(*interval_1, *interval_3),
                         run_utils.intersection(*interval_3, *interval_1))
        self.assertEqual(run_utils.intersection(*interval_1, *interval_4),
                         run_utils.intersection(*interval_4, *interval_1))
        # No intersection:
        self.assertEqual(run_utils.intersection(*interval_1, *interval_2), 0.)
        # Identity:
        self.assertEqual(run_utils.intersection(*interval_1, *interval_1), 1.)
        self.assertEqual(run_utils.intersection(*interval_4, *interval_4), 1.)
        # Intersections:
        self.assertAlmostEqual(run_utils.intersection(*interval_1, *interval_3), 0.75)
        self.assertAlmostEqual(run_utils.intersection(*interval_1, *interval_4), 1.)
        self.assertAlmostEqual(run_utils.intersection(*interval_2, *interval_3), 0.24)

    def test_border2average_correction(self):
        # Issue #10 regression:
        borders = [[231, 453], [460, 477]]
        averaged_borders = [[232, 325], [330, 333], [333, 476]]
        self.assertEqual(run_utils.border2average_correction(borders,
                                                              averaged_borders),
                          [[231, 325], [453, 453], [460, 477]])
        # len(borders) > len(average_borders):
        borders = [[232, 243], [256, 266], [268, 437], [470, 487]]
        averaged_borders = [[228, 243], [344, 432], [474, 488]]
        self.assertEqual(run_utils.border2average_correction(borders,
                                                              averaged_borders),
                          [[232, 243], [268, 437], [470, 487]])
        # no borders:
        borders = []
        averaged_borders = [[228, 243], [344, 432], [474, 488]]
        self.assertEqual(run_utils.border2average_correction(borders,
                                                              averaged_borders),
                          [[228, 243], [344, 432], [474, 488]])
        self.assertEqual(run_utils.border2average_correction(averaged_borders,
                                                              borders),
                          [])


if __name__ == '__main__':
    unittest.main()
