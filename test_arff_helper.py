#!/usr/bin/env python


import unittest
import os
import numpy as np
import arff

from sp_tool.arff_helper import ArffHelper


class ArffHelperTest(unittest.TestCase):

    def test_load(self):
        a = ArffHelper.load(open('test_data/arff_data_example.arff'))

        self.assertEqual(a['metadata']['distance_mm'], 450.0)
        del(a['metadata'])

        b = arff.load(open('test_data/arff_data_example.arff'))
        del (a['description'])
        del (b['description'])

        np.testing.assert_almost_equal(a['data'].tolist(), b['data'], 2)
        # tested data equality, can now delete it
        del (a['data'])
        del (b['data'])

        self.assertEqual(a, b)

    def test_dump(self):
        a = ArffHelper.load(open('test_data/arff_data_example.arff'))
        # close the returned file handle after dump has been completed
        ArffHelper.dump(a, open('test_data/test_dump.arff', 'w')).close()
        b = arff.load(open('test_data/test_dump.arff'))
        c = arff.load(open('test_data/arff_data_example.arff'))

        del (a['metadata'])
        del (a['description'])
        del (b['description'])
        del (c['description'])

        np.testing.assert_almost_equal(a['data'].tolist(), b['data'], 2)
        np.testing.assert_almost_equal(a['data'].tolist(), c['data'], 2)

        del (a['data'])
        del (b['data'])
        del (c['data'])

        self.assertEqual(a, b)
        self.assertEqual(a, c)
        os.remove('test_data/test_dump.arff')

    def test_add_column(self):
        name = 'EM_type'
        dtype = ['UNKNOWN', 'FIX', 'SACCADE', 'SP', 'NOISE']
        a = ArffHelper.load(open('test_data/arff_data_example.arff'))
        b = ArffHelper.add_column(a, name, dtype, 'UNKNOWN')
        a['attributes'].append((name, dtype))
        self.assertEqual(a['attributes'], b['attributes'])

if __name__ == '__main__':
    unittest.main()
