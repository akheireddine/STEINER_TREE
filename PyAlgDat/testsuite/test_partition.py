#!/usr/bin/env py.test

"""
Test Partition class.
"""

import unittest

from py_alg_dat import partition

class TestEntry(unittest.TestCase):

    """
    Test Partition class.
    """

    def setUp(self):
        pass

    ### Begin test of local class PartitionElement ###

    def test_partition_element_equal(self):
        """
        Test operator "equal" - partition element.
        """
        elem1 = partition.PartitionElement('a')
        elem2 = partition.PartitionElement('a')
        self.assertEqual(elem1, elem2)

    def test_partition_element_not_equal(self):
        """
        Test operator "inequal"- partition element.
        """
        elem1 = partition.PartitionElement('a')
        elem2 = partition.PartitionElement('b')
        self.assertNotEqual(elem1, elem2)

    ### End test of local class PartitionElement ###

    ### Begin test of class Partition ###

    def test_partition_contructor_without_elements(self):
        """
        Test constructor (empty).
        """
        res = partition.Partition()
        ref = {}
        self.assertEqual(ref, res.elems)

    def test_partition_contructor_with_elements(self):
        """
        Test constructor.
        """
        list1 = ['a', 'b', 'c']
        list2 = ['a', 'b', 'c']
        par1 = partition.Partition(list1)
        par2 = partition.Partition(list2)
        self.assertEqual(par1, par2)

    def test_partition_equal(self):
        """
        Test operator "equal".
        """
        list1 = ['a', 'b', 'c']
        list2 = ['a', 'b', 'c']
        par1 = partition.Partition(list1)
        par2 = partition.Partition(list2)
        self.assertEqual(par1, par2)

    def test_partition_not_equal(self):
        """
        Test operator "inequal".
        """
        list1 = ['a', 'b', 'c']
        list2 = ['a', 'b', 'cc']
        par1 = partition.Partition(list1)
        par2 = partition.Partition(list2)
        self.assertNotEqual(par1, par2)

    def test_partition_make_set(self):
        """
        Test method "make_set".
        """
        par1 = partition.Partition()
        par2 = partition.Partition()
        par1.make_set('a')
        par1.make_set('b')
        par1.make_set('c')
        par2.make_set('a')
        par2.make_set('b')
        par2.make_set('c')
        self.assertEqual(par1, par2)

    def test_same_set(self):
        """
        Test method "same_set".
        """
        par = partition.Partition()
        par.make_set('a')
        par.make_set('b')
        par.make_set('c')
        par.union_recursive('a', 'b')
        self.assertTrue(par.same_set('a', 'b'))

    def test_same_set_not(self):
        """
        Test method "same_set" - inverted.
        """
        par = partition.Partition()
        par.make_set('a')
        par.make_set('b')
        par.make_set('c')
        par.union_recursive('a', 'b')
        self.assertFalse(par.same_set('a', 'c'))

    def test_partition_find(self):
        """
        Test method "find".
        """
        par = partition.Partition()
        par.make_set('a')
        par.make_set('b')
        par.make_set('c')
        ref = 'b'
        res = par.find('b')
        self.assertEqual(ref, res)

    def test_partition_find_raise(self):
        """
        Test method "find" - raising exception.
        """
        par = partition.Partition(['a', 'b', 'c'])
        self.assertRaises(KeyError, lambda: par.find('not_in_partition'))

    def test_partition_find_element_recursive(self):
        """
        Test method "find_element" - recursive.
        """
        par = partition.Partition()
        par.make_set('a')
        par.make_set('b')
        par.make_set('c')
        ref = partition.PartitionElement('b')
        res = par.find_element_recursive('b')
        self.assertEqual(ref.parent, res)

    def test_partition_find_element_iterative(self):
        """
        Test method "find_element" - iterative.
        """
        par = partition.Partition()
        par.make_set('a')
        par.make_set('b')
        par.make_set('c')
        ref = partition.PartitionElement('b')
        res = par.find_element_iterative('b')
        self.assertEqual(ref.parent, res)

    def test_partition_union(self):
        """
        Test method "union".
        """
        res = partition.Partition()
        elem_a = 'a'
        elem_b = 'b'
        elem_c = 'c'
        elem_d = 'd'
        elem_e = 'e'
        elem_f = 'f'
        elem_g = 'g'
        elem_h = 'h'
        elem_i = 'i'
        elem_j = 'j'

        res.make_set(elem_a)
        res.make_set(elem_b)
        res.make_set(elem_c)
        res.make_set(elem_d)
        res.make_set(elem_e)
        res.make_set(elem_f)
        res.make_set(elem_g)
        res.make_set(elem_h)
        res.make_set(elem_i)
        res.make_set(elem_j)

        res.union(elem_a, elem_d)
        res.union(elem_h, elem_f)
        res.union(elem_i, elem_j)
        res.union(elem_g, elem_e)
        res.union(elem_f, elem_e)
        res.union(elem_b, elem_c)
        res.union(elem_c, elem_d)
        res.union(elem_d, elem_e)

        p_a = partition.PartitionElement('b')
        p_a.rank = 1
        p_b = partition.PartitionElement('b')
        p_b.rank = 3
        p_c = partition.PartitionElement('b')
        p_c.rank = 0
        p_d = partition.PartitionElement('b')
        p_d.rank = 0
        p_e = partition.PartitionElement('h')
        p_e.rank = 0
        p_f = partition.PartitionElement('h')
        p_f.rank = 0
        p_g = partition.PartitionElement('h')
        p_g.rank = 1
        p_h = partition.PartitionElement('b')
        p_h.rank = 2
        p_i = partition.PartitionElement('i')
        p_i.rank = 1
        p_j = partition.PartitionElement('i')
        p_j.rank = 0

        ref = partition.Partition()
        ref.elems['a'] = p_a
        ref.elems['b'] = p_b
        ref.elems['c'] = p_c
        ref.elems['d'] = p_d
        ref.elems['e'] = p_e
        ref.elems['f'] = p_f
        ref.elems['g'] = p_g
        ref.elems['h'] = p_h
        ref.elems['i'] = p_i
        ref.elems['j'] = p_j
        self.assertEqual(ref.elems, res.elems)

    ### End test of class Partition ###
