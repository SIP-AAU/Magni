"""
..
    Copyright (c) 2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing unittests of the `magni.cs.indicators`.

Routine listings
----------------
CoherenceTest(unittest.TestCase)
    Tests of the calculate_coherence function.
MutualCoherenceTest(unittest.TestCase)
    Tests of the calculate_mutual_coherence function.
RelativeEnergyTest(unittest.TestCase)
    Tests of the calculate_relative_energy function.

"""

from __future__ import division
import unittest

import numpy as np

import magni


class CoherenceTest(unittest.TestCase):
    """
    Tests of the calculate_coherence function.

    """

    def setUp(self):
        np.random.seed(1512)

        self._Phi, self._Psi = _generate_test_problem(5, 10, range(0, 10, 2))
        self._M = self._Phi.dot(self._Psi)

        for i in range(10):
            self._M[:, i] = self._M[:, i] / np.linalg.norm(self._M[:, i])

        self._M = self._M.T.dot(self._M) - np.eye(10)
        self._func = magni.cs.indicators.calculate_coherence

    def test_default(self):
        Phi, Psi, M = self._Phi, self._Psi, self._M

        true = {0: self._func(Phi, Psi, norm=0),
                1: self._func(Phi, Psi, norm=1),
                2: self._func(Phi, Psi, norm=2),
                np.inf: self._func(Phi, Psi, norm=np.inf)}
        test = self._func(Phi, Psi)

        # print('Expected: {}'.format(true))
        # print('Received: {}'.format(test))

        self.assertTrue(test == true)

    def test_0(self):
        Phi, Psi, M = self._Phi, self._Psi, self._M

        true = np.sum(np.abs(M) > 1e-9) / (M.size - M.shape[0])
        test = self._func(Phi, Psi, norm=0)

        # print('Expected: {}'.format(true))
        # print('Received: {}'.format(test))

        self.assertTrue(np.allclose(test, true))

    def test_1(self):
        Phi, Psi, M = self._Phi, self._Psi, self._M

        true = np.sum(np.abs(M)) / (M.size - M.shape[0])
        test = self._func(Phi, Psi, norm=1)

        # print('Expected: {}'.format(true))
        # print('Received: {}'.format(test))

        self.assertTrue(np.allclose(test, true))

    def test_2(self):
        Phi, Psi, M = self._Phi, self._Psi, self._M

        true = (np.sum(np.abs(M)**2) / (M.size - M.shape[0]))**(1 / 2)
        test = self._func(Phi, Psi, norm=2)

        # print('Expected: {}'.format(true))
        # print('Received: {}'.format(test))

        self.assertTrue(np.allclose(test, true))

    def test_inf(self):
        Phi, Psi, M = self._Phi, self._Psi, self._M

        true = np.max(np.abs(M))
        test = self._func(Phi, Psi, norm=np.inf)

        # print('Expected: {}'.format(true))
        # print('Received: {}'.format(test))

        self.assertTrue(np.allclose(test, true))


class MutualCoherenceTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(1514)

        self._Phi, self._Psi = _generate_test_problem(5, 10, range(0, 10, 2))
        self._M = self._Phi.dot(self._Psi)
        self._func = magni.cs.indicators.calculate_mutual_coherence

    def test_default(self):
        Phi, Psi, M = self._Phi, self._Psi, self._M

        true = {0: self._func(Phi, Psi, norm=0),
                1: self._func(Phi, Psi, norm=1),
                2: self._func(Phi, Psi, norm=2),
                np.inf: self._func(Phi, Psi, norm=np.inf)}
        test = self._func(Phi, Psi)

        # print('Expected: {}'.format(true))
        # print('Received: {}'.format(test))

        self.assertTrue(test == true)

    def test_0(self):
        Phi, Psi, M = self._Phi, self._Psi, self._M

        true = np.sum(np.abs(M) > 1e-9) / M.size
        test = self._func(Phi, Psi, norm=0)

        # print('Expected: {}'.format(true))
        # print('Received: {}'.format(test))

        self.assertTrue(np.allclose(test, true))

    def test_1(self):
        Phi, Psi, M = self._Phi, self._Psi, self._M

        true = np.sum(np.abs(M)) / M.size
        test = self._func(Phi, Psi, norm=1)

        # print('Expected: {}'.format(true))
        # print('Received: {}'.format(test))

        self.assertTrue(np.allclose(test, true))

    def test_2(self):
        Phi, Psi, M = self._Phi, self._Psi, self._M

        true = (np.sum(np.abs(M)**2) / M.size)**(1 / 2)
        test = self._func(Phi, Psi, norm=2)

        # print('Expected: {}'.format(true))
        # print('Received: {}'.format(test))

        self.assertTrue(np.allclose(test, true))

    def test_inf(self):
        Phi, Psi, M = self._Phi, self._Psi, self._M

        true = np.max(np.abs(M))
        test = self._func(Phi, Psi, norm=np.inf)

        # print('Expected: {}'.format(true))
        # print('Received: {}'.format(test))

        self.assertTrue(np.allclose(test, true))


class RelativeEnergyTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(1510)

        self._Phi, self._Psi = _generate_test_problem(5, 10, range(0, 10, 2))
        self._M = self._Phi.dot(self._Psi)
        self._func = magni.cs.indicators.calculate_relative_energy

    def test_default(self):
        Phi, Psi, M = self._Phi, self._Psi, self._M
        w = [np.linalg.norm(M[:, i]) for i in range(M.shape[1])]

        true = {'min': self._func(Phi, Psi, method='min'),
                'diff': self._func(Phi, Psi, method='diff'),
                'mean': self._func(Phi, Psi, method='mean'),
                'std': self._func(Phi, Psi, method='std')}
        test = self._func(Phi, Psi)

        # print('Expected: {}'.format(true))
        # print('Received: {}'.format(test))

        self.assertTrue(test == true)

    def test_min(self):
        Phi, Psi, M = self._Phi, self._Psi, self._M
        w = [np.linalg.norm(M[:, i]) for i in range(M.shape[1])]

        true = np.min(w)
        test = self._func(Phi, Psi, method='min')

        # print('Expected: {}'.format(true))
        # print('Received: {}'.format(test))

        self.assertTrue(np.allclose(test, true))

    def test_diff(self):
        Phi, Psi, M = self._Phi, self._Psi, self._M
        w = [np.linalg.norm(M[:, i]) for i in range(M.shape[1])]

        true = np.max(w) - np.min(w)
        test = self._func(Phi, Psi, method='diff')

        # print('Expected: {}'.format(true))
        # print('Received: {}'.format(test))

        self.assertTrue(np.allclose(test, true))

    def test_mean(self):
        Phi, Psi, M = self._Phi, self._Psi, self._M
        w = [np.linalg.norm(M[:, i]) for i in range(M.shape[1])]

        true = np.mean(w)
        test = self._func(Phi, Psi, method='mean')

        # print('Expected: {}'.format(true))
        # print('Received: {}'.format(test))

        self.assertTrue(np.allclose(test, true))

    def test_std(self):
        Phi, Psi, M = self._Phi, self._Psi, self._M
        w = [np.linalg.norm(M[:, i]) for i in range(M.shape[1])]

        true = np.std(w)
        test = self._func(Phi, Psi, method='std')

        # print('Expected: {}'.format(true))
        # print('Received: {}'.format(test))

        self.assertTrue(np.allclose(test, true))


def _generate_test_problem(n, m, coords):
    Phi = np.zeros((n, m))

    for i, j in enumerate(coords):
        Phi[i, j] = 1

    Psi = np.random.randn(m, m)

    for i in range(m):
        Psi[:, i] = Psi[:, i] / np.linalg.norm(Psi[:, i])

    return Phi, Psi
