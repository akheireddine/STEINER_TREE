#!/usr/bin/env py.test

"""
Test MaxHeap class.
"""

import unittest

from py_alg_dat import max_heap

class TestMaxHeap(unittest.TestCase):

    """
    Test MaxHeap class.
    """

    def setUp(self):
        self.heap1 = max_heap.MaxHeap()
        self.heap1.insert(4)
        self.heap1.insert(1)
        self.heap1.insert(3)
        self.heap1.insert(2)
        self.heap1.insert(16)
        self.heap1.insert(9)
        self.heap1.insert(10)
        self.heap1.insert(14)
        self.heap1.insert(8)
        self.heap1.insert(7)

    def test_build_max_heap_recursive_empty(self):
        """
        Test constructor (empty) "MaxHeap" - recursive.
        """
        ref = []
        res = max_heap.MaxHeap()
        self.assertEqual(ref, res.array)

    def test_build_max_heap_recurxsive(self):
        """
        Test constructor "MaxHeap" - recursive.
        """
        ref = [16, 14, 10, 8, 7, 9, 3, 2, 4, 1]
        res = max_heap.MaxHeap([4, 1, 3, 2, 16, 9, 10, 14, 8, 7])
        self.assertEqual(ref, res.array)

    def test_maxheap_len(self):
        """
        Test operator "len".
        """
        heap = max_heap.MaxHeap([4, 1, 3, 2, 16, 9, 10, 14, 8, 7])
        self.assertEqual(10, len(heap))

    def test_maxheap_equal(self):
        """
        Test operator "equal".
        """
        heap1 = max_heap.MaxHeap([4, 1, 3, 2, 16, 9, 10, 14, 8, 7])
        heap2 = max_heap.MaxHeap([4, 1, 3, 2, 16, 9, 10, 14, 8, 7])
        self.assertEqual(heap1, heap2)

    def test_maxheap_not_equal(self):
        """
        Test operator "inequal".
        """
        heap1 = max_heap.MaxHeap([4, 1, 3, 2, 16, 9, 10, 14, 8, 7])
        heap2 = max_heap.MaxHeap([4, 1, 3, 2, 16, 9, 10, 14, 8, 100])
        self.assertNotEqual(heap1, heap2)

    def test_maxheap_is_empty(self):
        """
        Test method "is_empty".
        """
        heap = max_heap.MaxHeap()
        self.assertTrue(heap.is_empty())

    def test_maxheap_is_empty_not(self):
        """
        Test method "is_empty" - inverted.
        """
        heap = max_heap.MaxHeap([4, 1, 3, 2, 16, 9, 10, 14, 8, 7])
        self.assertFalse(heap.is_empty())

    def test_maxheap_clear(self):
        """
        Test method "clear".
        """
        heap = max_heap.MaxHeap([4, 1, 3, 2, 16, 9, 10, 14, 8, 7])
        heap.clear()
        test1 = len(heap) == 0
        test2 = heap.array == []
        test = test1 and test2
        self.assertTrue(test)

    def test_maxheap_parent(self):
        """
        Test method "parent".
        """
        heap = max_heap.MaxHeap([4, 1, 3, 2, 16, 9, 10, 14, 8, 7])
        index = 5
        parent_index = heap.parent(index)
        parent_value = heap.array[parent_index]
        test1 = parent_index == 2
        test2 = parent_value == 10
        test = test1 and test2
        self.assertTrue(test)

    def test_maxheap_parent_not(self):
        """
        Test method "parent" - inverted.
        """
        heap = max_heap.MaxHeap([4, 1, 3, 2, 16, 9, 10, 14, 8, 7])
        index = 2
        parent_index = heap.parent(index)
        parent_value = heap.array[parent_index]
        test1 = parent_index == 0
        test2 = parent_value == 16
        test = test1 and test2
        self.assertTrue(test)

    def test_maxheap_left_child(self):
        """
        Test method "left_child".
        """
        heap = max_heap.MaxHeap([4, 1, 3, 2, 16, 9, 10, 14, 8, 7])
        index = 1
        left_index = heap.left_child(index)
        left_value = heap.array[left_index]
        test1 = left_index == 3
        test2 = left_value == 8
        test = test1 and test2
        self.assertTrue(test)

    def test_maxheap_left_child_not(self):
        """
        Test method "left_child" - inverted.
        """
        heap = max_heap.MaxHeap([4, 1, 3, 2, 16, 9, 10, 14, 8, 7])
        index = 2
        left_index = heap.left_child(index)
        left_value = heap.array[left_index]
        test1 = left_index == 3
        test2 = left_value == 8
        test = test1 and test2
        self.assertFalse(test)

    def test_maxheap_right_child(self):
        """
        Test method "right_child".
        """
        heap = max_heap.MaxHeap([4, 1, 3, 2, 16, 9, 10, 14, 8, 7])
        index = 1
        right_index = heap.right_child(index)
        right_value = heap.array[right_index]
        test1 = right_index == 4
        test2 = right_value == 7
        test = test1 and test2
        self.assertTrue(test)

    def test_maxheap_right_child_not(self):
        """
        Test method "right_child" - inverted.
        """
        heap = max_heap.MaxHeap([4, 1, 3, 2, 16, 9, 10, 14, 8, 7])
        index = 2
        right_index = heap.right_child(index)
        right_value = heap.array[right_index]
        test1 = right_index == 4
        test2 = right_value == 7
        test = test1 and test2
        self.assertFalse(test)

    def test_maxheap_is_leaf(self):
        """
        Test method "is_lead".
        """
        heap = max_heap.MaxHeap([4, 1, 3, 2, 16, 9, 10, 14, 8, 7])
        t_1 = heap.is_leaf(0) == False
        t_2 = heap.is_leaf(1) == False
        t_3 = heap.is_leaf(2) == False
        t_4 = heap.is_leaf(3) == False
        t_5 = heap.is_leaf(4) == False
        t_6 = heap.is_leaf(5) == True
        t_7 = heap.is_leaf(6) == True
        t_8 = heap.is_leaf(7) == True
        t_9 = heap.is_leaf(8) == True
        t_10 = heap.is_leaf(9) == True
        test = t_1 and t_2 and t_3 and t_4 and t_5 and t_6 and t_7 and t_8 and t_9 and t_10
        self.assertTrue(test)

    def test_maxheap_is_max_heap(self):
        """
        Test method "is_max_heap".
        """
        heap = max_heap.MaxHeap([4, 1, 3, 2, 16, 9, 10, 14, 8, 7])
        self.assertTrue(heap.is_max_heap())

    def test_maxheap_is_max_heap_not(self):
        """
        Test method "is_max_heap" - inverted.
        """
        heap = max_heap.MaxHeap()
        heap.array = [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]
        self.assertFalse(heap.is_max_heap())

    def test_maxheap_insert(self):
        """
        Test method "insert".
        """
        ref = max_heap.MaxHeap([4, 1, 3, 2, 16, 9, 10, 14, 8, 7])
        res = max_heap.MaxHeap()
        res.insert(4)
        res.insert(16)
        res.insert(9)
        res.insert(2)
        res.insert(1)
        res.insert(10)
        res.insert(3)
        res.insert(14)
        res.insert(8)
        res.insert(7)
        self.assertEqual(res, ref)

    def test_maxheap_remove(self):
        """
        Test method "remove".
        """
        ref = max_heap.MaxHeap([16, 14, 9, 8, 7, 1, 3, 2, 4])
        heap = max_heap.MaxHeap([4, 1, 3, 2, 16, 9, 10, 14, 8, 7])
        heap.remove(2)
        test1 = heap.is_max_heap() == True
        test2 = ref.array == heap.array
        test = test1 and test2
        self.assertTrue(test)

    def test_maxheap_build_maxheap_recursive(self):
        """
        Test method "build_max_heap" - recursive.
        """
        ref = [16, 14, 10, 8, 7, 9, 3, 2, 4, 1]
        res = [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]
        heap = max_heap.MaxHeap()
        heap.build_max_heap(res)
        self.assertEqual(res, ref)

    def test_maxheap_build_maxheap_iterative(self):
        """
        Test method "build_max_heap" - iterative.
        """
        ref = [16, 14, 10, 8, 7, 9, 3, 2, 4, 1]
        res = [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]
        heap = max_heap.MaxHeap()
        heap.build_max_heap(res, False)
        self.assertEqual(res, ref)

    def test_maxheap_heap_sort(self):
        """
        Test method "heap_sort".
        """
        ref = [16, 14, 10, 9, 8, 7, 4, 3, 2, 1]
        self.heap1.heap_sort()
        self.assertEqual(ref, self.heap1.array)

    def test_maxheap_extract_max(self):
        """
        Test method "heap_extract_max".
        """
        self.assertEqual(16, self.heap1.heap_extract_max())

    def test_maxheap_increase_key(self):
        """
        Test method "heap_increase_key".
        """
        self.heap1.heap_increase_key(5, 50)
        self.assertEqual(self.heap1.array[0], 50)

    def test_maxheap_merge(self):
        """
        Test method "heap_merge".
        """
        ref = [16, 14, 10, 8, 9, 7, 3, 2, 4, 1]
        heap1 = max_heap.MaxHeap([16, 14, 10, 8])
        heap2 = max_heap.MaxHeap([7, 9, 3, 2, 4, 1])
        heap = heap1.heap_merge(heap2)
        test1 = ref == heap.array
        test2 = heap.is_max_heap()
        test = test1 and test2
        self.assertTrue(test)
