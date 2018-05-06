#!/usr/bin/env py.test

"""
Test MinHeap class.
"""

import unittest

from py_alg_dat import min_heap

class TestMinHeap(unittest.TestCase):

    """
    Test MinHeap class.
    """

    def setUp(self):
        self.heap1 = min_heap.MinHeap()
        self.heap1.insert(1)
        self.heap1.insert(2)
        self.heap1.insert(3)
        self.heap1.insert(17)
        self.heap1.insert(19)
        self.heap1.insert(36)
        self.heap1.insert(7)
        self.heap1.insert(25)
        self.heap1.insert(100)

    def test_build_minheap_recursive_empty(self):
        """
        Test constructor (empty) "MinHeap" - recursive.
        """
        ref = []
        res = min_heap.MinHeap()
        self.assertEqual(ref, res.array)

    def test_build_minheap_recursive(self):
        """
        Test constructor "MinHeap" - recursive.
        """
        ref = [1, 2, 3, 19, 17, 7, 25, 36, 100]
        res = min_heap.MinHeap([100, 36, 25, 19, 17, 7, 3, 2, 1])
        self.assertEqual(ref, res.array)

    def test_minheap_len(self):
        """
        Test operator "len".
        """
        heap = min_heap.MinHeap([100, 36, 25, 19, 17, 7, 3, 2, 1])
        self.assertEqual(9, len(heap))

    def test_minheap_equal(self):
        """
        Test operator "equal".
        """
        heap = min_heap.MinHeap([1, 2, 3, 17, 19, 36, 7, 25, 100])
        self.assertEqual(heap, self.heap1)

    def test_minheap_not_equal(self):
        """
        Test operator "inequal".
        """
        heap1 = min_heap.MinHeap([1, 2, 3, 17, 19, 36, 7, 25, 100])
        heap2 = min_heap.MinHeap([1, 2, 3, 17, 19, 36, 7, 25, 200])
        self.assertNotEqual(heap1, heap2)

    def test_minheap_is_empty(self):
        """
        Test method "is_empty".
        """
        heap = min_heap.MinHeap()
        self.assertTrue(heap.is_empty())

    def test_minheap_is_empty_not(self):
        """
        Test method "is_empty" - inverted.
        """
        self.assertFalse(self.heap1.is_empty())

    def test_minheap_clear(self):
        """
        Test method "clear".
        """
        heap = min_heap.MinHeap([100, 36, 25, 19, 17, 7, 3, 2, 1])
        heap.clear()
        test1 = len(heap) == 0
        test2 = heap.array == []
        test = test1 and test2
        self.assertTrue(test)

    def test_minheap_parent(self):
        """
        Test method "parent".
        """
        heap = min_heap.MinHeap([100, 36, 25, 19, 17, 7, 3, 2, 1])
        index = 7
        parent_index = heap.parent(index)
        parent_value = heap.array[parent_index]
        test1 = parent_index == 3
        test2 = parent_value == 19
        test = test1 and test2
        self.assertTrue(test)

    def test_minheap_parent_not(self):
        """
        Test method "parent" - inverted.
        """
        heap = min_heap.MinHeap([100, 36, 25, 19, 17, 7, 3, 2, 1])
        index = 6
        parent_index = heap.parent(index)
        parent_value = heap.array[parent_index]
        test1 = parent_index == 3
        test2 = parent_value == 19
        test = test1 and test2
        self.assertFalse(test)

    def test_minheap_left_child(self):
        """
        Test method "left_child".
        """
        heap = min_heap.MinHeap([100, 36, 25, 19, 17, 7, 3, 2, 1])
        index = 3
        left_index = heap.left_child(index)
        left_value = heap.array[left_index]
        test1 = left_index == 7
        test2 = left_value == 36
        test = test1 and test2
        self.assertTrue(test)

    def test_minheap_left_child_not(self):
        """
        Test method "left_child" - inverted.
        """
        heap = min_heap.MinHeap([100, 36, 25, 19, 17, 7, 3, 2, 1])
        index = 2
        left_index = heap.left_child(index)
        left_value = heap.array[left_index]
        test1 = left_index == 7
        test2 = left_value == 36
        test = test1 and test2
        self.assertFalse(test)

    def test_minheap_right_child(self):
        """
        Test method "right_child".
        """
        heap = min_heap.MinHeap([100, 36, 25, 19, 17, 7, 3, 2, 1])
        index = 1
        right_index = heap.right_child(index)
        right_value = heap.array[right_index]
        test1 = right_index == 4
        test2 = right_value == 17
        test = test1 and test2
        self.assertTrue(test)

    def test_minheap_right_child_not(self):
        """
        Test method "right_child" - inverted.
        """
        heap = min_heap.MinHeap([100, 36, 25, 19, 17, 7, 3, 2, 1])
        index = 2
        right_index = heap.right_child(index)
        right_value = heap.array[right_index]
        test1 = right_index == 6
        test2 = right_value == 25
        test = test1 and test2
        self.assertTrue(test)

    def test_minheap_is_leaf(self):
        """
        Test method "is_leaf".
        """
        heap = min_heap.MinHeap([100, 36, 25, 19, 17, 7, 3, 2, 1])
        t_1 = heap.is_leaf(0) == False
        t_2 = heap.is_leaf(1) == False
        t_3 = heap.is_leaf(2) == False
        t_4 = heap.is_leaf(3) == False
        t_5 = heap.is_leaf(4) == True
        t_6 = heap.is_leaf(5) == True
        t_7 = heap.is_leaf(6) == True
        t_8 = heap.is_leaf(7) == True
        t_9 = heap.is_leaf(8) == True
        test = t_1 and t_2 and t_3 and t_4 and t_5 and t_6 and t_7 and t_8 and t_9
        self.assertTrue(test)

    def test_minheap_is_heap_not(self):
        """
        Test method "is_min_heap" - inverted.
        """
        heap = min_heap.MinHeap()
        heap.array = [100, 36, 25, 19, 17, 7, 3, 2, 1]
        self.assertFalse(heap.is_min_heap())

    def test_minheap_insert(self):
        """
        Test method "insert".
        """
        ref = min_heap.MinHeap([100, 36, 25, 19, 17, 7, 3, 2, 1])
        res = min_heap.MinHeap()
        res.insert(1)
        res.insert(2)
        res.insert(3)
        res.insert(19)
        res.insert(17)
        res.insert(7)
        res.insert(25)
        res.insert(36)
        res.insert(100)
        self.assertEqual(res, ref)

    def test_minheap_remove(self):
        """
        Test method "remove".
        """
        ref = [1, 2, 3, 36, 17, 7, 25, 100]
        heap = min_heap.MinHeap([100, 36, 25, 19, 17, 7, 3, 2, 1])
        heap.remove(3)
        test1 = heap.is_min_heap() == True
        test2 = ref == heap.array
        test = test1 and test2
        self.assertTrue(test)

    def test_minheap_build_minheap_recursive(self):
        """
        Test method "build_min_heap" - recursive.
        """
        ref = [1, 2, 3, 4, 7, 8, 9, 10, 14, 16]
        res = [1, 2, 3, 4, 7, 8, 9, 10, 14, 16]
        heap = min_heap.MinHeap()
        heap.build_min_heap(res)
        self.assertEqual(res, ref)

    def test_minheap_build_minheap_iterative(self):
        """
        Test method "build_min_heap" - iterative.
        """
        ref = [1, 2, 3, 4, 7, 8, 9, 10, 14, 16]
        res = [1, 2, 3, 4, 7, 8, 9, 10, 14, 16]
        heap = min_heap.MinHeap()
        heap.build_min_heap(res, False)
        self.assertEqual(res, ref)

    def test_minheap_heap_sort(self):
        """
        Test method "heap_sort".
        """
        ref = [1, 2, 3, 7, 17, 19, 25, 36, 100]
        self.heap1.heap_sort()
        self.assertEqual(ref, self.heap1.array)

    def test_minheap_extract_min(self):
        """
        Test method "heap_extract_min".
        """
        self.assertEqual(1, self.heap1.heap_extract_min())

    def test_minheap_increase_key(self):
        """
        Test method "heap_increase_key".
        """
        self.heap1.heap_increase_key(5, 50)
        self.assertEqual(self.heap1.array[5], 50)

    def test_minheap_merge(self):
        """
        Test method "heap_merge".
        """
        ref = [1, 7, 2, 17, 36, 25, 3, 19, 100]
        heap1 = min_heap.MinHeap([100, 36, 25, 19])
        heap2 = min_heap.MinHeap([17, 7, 3, 2, 1])
        heap = heap1.heap_merge(heap2)
        test1 = ref == heap.array
        test2 = heap.is_min_heap()
        test = test1 and test2
        self.assertTrue(test)

