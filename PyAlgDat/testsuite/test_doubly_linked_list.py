#!/usr/bin/env py.test

"""
Test doubly linked list.
"""

import copy
import unittest

from py_alg_dat import doubly_linked_list

class TestDoublyLinkedList(unittest.TestCase):

    """
    Test doubly linked list.
    """

    def setUp(self):
        self.list1 = doubly_linked_list.DoublyLinkedList()
        self.list1.append('a')
        self.list1.append('b')
        self.list1.append('c')
        self.list1.append('d')
        self.list1.append('e')

        self.list2 = doubly_linked_list.DoublyLinkedList()
        self.list2.prepend('a')
        self.list2.prepend('b')
        self.list2.prepend('c')
        self.list2.prepend('d')
        self.list2.prepend('e')

    ### Begin test of local class DoublyLinkedListElement ###

    def test_doubly_linked_list_element_equal(self):
        """
        Test operator "equal".
        """
        a_list = doubly_linked_list.DoublyLinkedList()
        elem1 = doubly_linked_list.DoublyLinkedListElement(a_list, 'a', None, None)
        elem2 = doubly_linked_list.DoublyLinkedListElement(a_list, 'a', None, None)
        self.assertEqual(elem1, elem2)

    def test_doubly_linked_list_element_not_equal(self):
        """
        Test operator "inequal".
        """
        a_list = doubly_linked_list.DoublyLinkedList()
        elem1 = doubly_linked_list.DoublyLinkedListElement(a_list, 'a', None, None)
        elem2 = doubly_linked_list.DoublyLinkedListElement(a_list, 'b', None, None)
        self.assertNotEqual(elem1, elem2)

    def test_doubly_linked_list_element_copy_equal(self):
        """
        Test operator "copy".
        """
        a_list = doubly_linked_list.DoublyLinkedList()
        elem = doubly_linked_list.DoublyLinkedListElement(a_list, 'a', None, None)
        e_copy = copy.copy(elem)
        self.assertEqual(elem, e_copy)

    def test_doubly_linked_list_element_copy_not_equal(self):
        """
        Test operator "copy".
        """
        a_list = doubly_linked_list.DoublyLinkedList()
        elem = doubly_linked_list.DoublyLinkedListElement(a_list, 'a', None, None)
        e_copy = copy.copy(elem)
        elem.data = 'aa'
        self.assertNotEqual(elem, e_copy)

    def test_doubly_linked_list_element_get_data(self):
        """
        Test method "get_data".
        """
        a_list = doubly_linked_list.DoublyLinkedList()
        elem = doubly_linked_list.DoublyLinkedListElement(a_list, 'a', None, None)
        self.assertEqual('a', elem.get_data())

    def test_doubly_linked_list_element_get_previous(self):
        """
        Test method "get_previous".
        """
        a_list = doubly_linked_list.DoublyLinkedList()
        elem1 = doubly_linked_list.DoublyLinkedListElement(a_list, 'a', None, None)
        elem2 = doubly_linked_list.DoublyLinkedListElement(a_list, 'b', elem1, None)
        self.assertEqual(elem1, elem2.get_previous())

    def test_doubly_linked_list_element_get_next(self):
        """
        Test method "get_next".
        """
        a_list = doubly_linked_list.DoublyLinkedList()
        elem2 = doubly_linked_list.DoublyLinkedListElement(a_list, 'b', None, None)
        elem1 = doubly_linked_list.DoublyLinkedListElement(a_list, 'a', None, elem2)
        self.assertEqual(elem2, elem1.get_next())

    def test_doubly_linked_list_element_insert_at(self):
        """
        Test method "insert_at".
        """
        a_list = doubly_linked_list.DoublyLinkedList()
        elem1 = doubly_linked_list.DoublyLinkedListElement(a_list, 'a', None, None)
        elem1.insert_after('b')
        elem2 = elem1.get_next()
        elem2.insert_after('c')
        elem3 = elem2.get_next()
        elem3.insert_after('d')
        elem4 = elem3.get_next()
        elem4.insert_after('e')
        elem5 = elem4.get_next()
        elem5.insert("test")
        self.assertEqual('test', elem1.get_next().get_next().get_next().get_next().get_data())

    def test_doubly_linked_list_element_insert_before(self):
        """
        Test method "insert_before".
        """
        a_list = doubly_linked_list.DoublyLinkedList()
        elem = doubly_linked_list.DoublyLinkedListElement(a_list, 'b', None, None)
        elem.insert_before('a')
        self.assertEqual('a', elem.get_previous().get_data())

    def test_doubly_linked_list_element_insert_between(self):
        """
        Test method "insert_between".
        """
        a_list = doubly_linked_list.DoublyLinkedList()
        elem1 = doubly_linked_list.DoublyLinkedListElement(a_list, 'a', None, None)
        elem1.insert_after('c')
        elem2 = elem1.get_next()
        elem1.insert_between('b', elem2)
        self.assertEqual('b', elem1.get_next().get_data())

    def test_doubly_linked_list_element_insert_after(self):
        """
        Test method "insert_after".
        """
        a_list = doubly_linked_list.DoublyLinkedList()
        elem = doubly_linked_list.DoublyLinkedListElement(a_list, 'a', None, None)
        elem.insert_after('b')
        self.assertEqual('b', elem.get_next().get_data())

    def test_doubly_linked_list_element_remove(self):
        """
        Test method "remove".
        """
        a_list = doubly_linked_list.DoublyLinkedList()
        elem1 = doubly_linked_list.DoublyLinkedListElement(a_list, 'a', None, None)
        elem1.insert_after('b')
        elem2 = elem1.get_next()
        elem2.insert_after('c')
        elem3 = elem2.get_next()
        elem3.insert_after('d')
        elem2.remove()
        self.assertEqual('c', elem1.get_next().get_data())

    ### End test of local class DoublyLinkedListElement ###

    ### Begin test of class DoublyLinkedList ###

    def test_doubly_linked_list_len(self):
        """
        Test operator "len".
        """
        self.assertEqual(5, len(self.list1))

    def test_doubly_linked_list_contains_true(self):
        """
        Test operator "contains".
        """
        result = 'a' in self.list1
        self.assertTrue(result)

    def test_doubly_linked_list_contains_false(self):
        """
        Test operator "contains" - inverted.
        """
        result = 'aa' in self.list1
        self.assertFalse(result)

    def test_doubly_linked_list_get_item(self):
        """
        Test operator "get_item".
        """
        elem = doubly_linked_list.DoublyLinkedListElement(self.list1, 'a', None, None)
        self.assertEqual(elem, self.list1[0])

    def test_doubly_linked_list_get_item_raise(self):
        """
        Test operator "get_item".
        """
        self.assertRaises(IndexError, lambda: self.list1[10])

    def test_doubly_linked_list_iter_(self):
        """
        Test operator "iterator".
        """
        tmp1 = []
        tmp2 = []
        tmp1.append(self.list1[0])
        tmp1.append(self.list1[1])
        tmp1.append(self.list1[2])
        tmp1.append(self.list1[3])
        tmp1.append(self.list1[4])
        for i in self.list1:
            tmp2.append(i)
        self.assertEqual(tmp1, tmp2)

    def test_doubly_linked_list_get_head(self):
        """
        Test method "get_head".
        """
        self.assertEqual('a', self.list1.get_head().get_data())

    def test_doubly_linked_list_get_tail(self):
        """
        Test method "get_tail".
        """
        self.assertEqual('e', self.list1.get_tail().get_data())

    def test_doubly_linked_list_is_empty(self):
        """
        Test method "is_empty".
        """
        self.assertFalse(self.list1.is_empty())

    def test_doubly_linked_list_get_first(self):
        """
        Test method "get_first".
        """
        self.assertEqual('a', self.list1.get_first())

    def test_doubly_linked_list_get_last(self):
        """
        Test method "get_last".
        """
        self.assertEqual('e', self.list1.get_last())

    def test_doubly_linked_list_clear(self):
        """
        Test method "clear".
        """
        a_list = doubly_linked_list.DoublyLinkedList()
        a_list.append('a')
        a_list.append('b')
        a_list.clear()
        self.assertTrue(a_list.get_head() == None and a_list.get_tail() == None and len(a_list) == 0)

    def test_doubly_linked_list_append(self):
        """
        Test method "append".
        """
        a_list = doubly_linked_list.DoublyLinkedList()
        a_list.append('a')
        a_list.append('b')
        a_list.append('c')
        self.assertEqual('c', a_list.get_last())

    def test_doubly_linked_list_prepend(self):
        """
        Test method "prepend".
        """
        a_list = doubly_linked_list.DoublyLinkedList()
        a_list.prepend('a')
        a_list.prepend('b')
        a_list.prepend('c')
        self.assertEqual('c', a_list.get_first())

    def test_doubly_linked_list_insert_at_index(self):
        """
        Test method "insert_at_index".
        """
        a_list = doubly_linked_list.DoublyLinkedList()
        a_list.append('a')
        a_list.append('b')
        a_list.append('c')
        a_list.insert_at('aa', 0)
        a_list.insert_at('bb', 1)
        a_list.insert_at('cc', 2)
        elem0 = a_list[0].get_data()
        elem1 = a_list[1].get_data()
        elem2 = a_list[2].get_data()
        self.assertTrue('aa' == elem0 and 'bb' == elem1 and 'cc' == elem2)

    def test_doubly_linked_list_insert_before_element(self):
        """
        Test method "insert_before_element".
        """
        a_list = doubly_linked_list.DoublyLinkedList()
        a_list.append('a')
        a_list.append('b')
        a_list.append('c')
        elem2 = a_list[2]
        a_list.insert_before_element('cc', elem2)
        self.assertEqual('cc', a_list[2].get_data())

    def test_doubly_linked_list_insert_between_elements(self):
        """
        Test method "insert_between_elements".
        """
        a_list = doubly_linked_list.DoublyLinkedList()
        a_list.append('a')
        a_list.append('c')
        elem0 = a_list[0]
        elem1 = a_list[1]
        a_list.insert_between_elements('bb', elem0, elem1)
        self.assertEqual('bb', a_list[1].get_data())

    def test_doubly_linked_list_insert_after_element(self):
        """
        Test method "insert_after_element".
        """
        a_list = doubly_linked_list.DoublyLinkedList()
        a_list.append('a')
        a_list.append('b')
        a_list.append('c')
        elem2 = a_list[2]
        a_list.insert_after_element('cc', elem2)
        self.assertEqual('cc', a_list[3].get_data())

    def test_doubly_linked_list_remove(self):
        """
        Test method "remove".
        """
        self.list2.remove('e')
        self.assertEqual('a', self.list2.get_last())

    ### End test of class DoublyLinkedList ###
