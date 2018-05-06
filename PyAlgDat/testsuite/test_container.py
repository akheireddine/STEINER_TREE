#!/usr/bin/env py.test

"""
Test of Container class - a general container data structure.
"""

import unittest

from py_alg_dat import container

class TestContainer(unittest.TestCase):

    """
    Test Container class.
    """

    def setUp(self):
        """
        Setup global test variables.
        """
        self.con1 = container.Container()

    def test_container_get_count(self):
        """
        Test get count method.
        """
        self.assertEqual(0, self.con1.get_count())

    def test_container_is_empty(self):
        """
        Test is empty method.
        """
        self.assertTrue(self.con1.is_empty())
