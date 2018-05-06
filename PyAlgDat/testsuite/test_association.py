#!/usr/bin/env py.test

"""
Test of Association - a DTO class with key and value pair.
"""

import unittest

from py_alg_dat import association

class TestAssociation(unittest.TestCase):
    """
    Test Association class.
    """

    def setUp(self):
        self.assoc1 = association.Association("k1", 1)
        self.assoc2 = association.Association("k2", 2)
        self.assoc3 = association.Association("k1", 1)

    def test_association_get_key(self):
        """
        Test get key method.
        """
        self.assertEqual("k1", self.assoc1.get_key())

    def test_association_get_value(self):
        """
        Test get value method.
        """
        self.assertEqual(1, self.assoc1.get_value())

    def test_association_equal(self):
        """
        Test equal operator.
        """
        self.assertEqual(self.assoc1, self.assoc3)

    def test_association_not_equal(self):
        """
        Test inequal operator.
        """
        self.assertNotEqual(self.assoc1, self.assoc2)

