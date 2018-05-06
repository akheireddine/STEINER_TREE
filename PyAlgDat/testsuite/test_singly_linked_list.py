#!/usr/bin/env py.test

"""
Test SinglyLinkedList class.
"""

import copy
import unittest

from py_alg_dat import singly_linked_list

class TestSinglyLinkedList(unittest.TestCase):

    """
    Test SinglyLinkedList class.
    """

    def setUp(self):
        self.list1 = singly_linked_list.SinglyLinkedList()
        self.list1.append('b')
        self.list1.append('c')
        self.list1.append('d')

    ### Begin test of local class SinglyLinkedListElement ###

    def test_singly_linked_list_element_equal(self):
        """
        Test operator (list element) "equal".
        """
        a_list = singly_linked_list.SinglyLinkedList()
        elem1 = singly_linked_list.SinglyLinkedListElement(a_list, 'a', None)
        elem2 = singly_linked_list.SinglyLinkedListElement(a_list, 'a', None)
        self.assertEqual(elem1, elem2)

    def test_singly_linked_list_element_not_equal(self):
        """
        Test operator (list element) "equal" - inverted.
        """
        a_list = singly_linked_list.SinglyLinkedList()
        elem1 = singly_linked_list.SinglyLinkedListElement(a_list, 'a', None)
        elem2 = singly_linked_list.SinglyLinkedListElement(a_list, 'b', None)
        self.assertNotEqual(elem1, elem2)

    def test_singly_linked_list_element_copy_equal(self):
        """
        Test operator (list element) "copy".
        """
        a_list = singly_linked_list.SinglyLinkedList()
        elem = singly_linked_list.SinglyLinkedListElement(a_list, 'a', None)
        e_copy = copy.copy(elem)
        self.assertEqual(elem, e_copy)

    def test_singly_linked_list_element_copy_not_equal(self):
        """
        Test operator (list element) "copy" - inverted.
        """
        a_list = singly_linked_list.SinglyLinkedList()
        elem = singly_linked_list.SinglyLinkedListElement(a_list, 'a', None)
        e_copy = copy.copy(elem)
        elem.data = 'aa'
        self.assertNotEqual(elem, e_copy)

    def test_singly_linked_list_element_get_data(self):
        """
        Test method (list element) "get_data".
        """
        a_list = singly_linked_list.SinglyLinkedList()
        elem = singly_linked_list.SinglyLinkedListElement(a_list, 'a', None)
        self.assertEqual('a', elem.get_data())

    def test_singly_linked_list_element_get_next(self):
        """
        Test method (list element) "get_next".
        """
        a_list = singly_linked_list.SinglyLinkedList()
        elem2 = singly_linked_list.SinglyLinkedListElement(a_list, 'b', None)
        elem1 = singly_linked_list.SinglyLinkedListElement(a_list, 'a', elem2)
        self.assertEqual(elem2, elem1.get_next())

    def test_singly_linked_list_element_insert_empty(self):
        """
        Test method (list element) "insert".
        """
        a_list = singly_linked_list.SinglyLinkedList()
        elem1 = singly_linked_list.SinglyLinkedListElement(a_list, 'a', None)

        elem1.insert('b')

        test1 = a_list.get_head() == a_list[0]
        test2 = a_list.get_tail() == a_list[0]
        test3 = a_list[0].get_next() == None
        test4 = len(a_list) == 1
        test = test1 and test2 and test3 and test4
        self.assertTrue(test)

    def test_singly_linked_list_element_insert_head(self):
        """
        Test method (list element) "insert".
        """
        a_list = singly_linked_list.SinglyLinkedList()
        elem2 = singly_linked_list.SinglyLinkedListElement(a_list, 'c', None)
        elem1 = singly_linked_list.SinglyLinkedListElement(a_list, 'b', elem2)
        elem0 = singly_linked_list.SinglyLinkedListElement(a_list, 'a', elem1)

        a_list.append(elem0.get_data())
        a_list.append(elem1.get_data())
        a_list.append(elem2.get_data())
        elem0.insert('aa')

        test1 = a_list.get_head() == a_list[0]
        test2 = a_list.get_tail() == a_list[len(a_list) - 1]
        test3 = a_list[0].get_next() == elem1
        test4 = len(a_list) == 3
        test = test1 and test2 and test3 and test4
        self.assertTrue(test)

    def test_singly_linked_list_element_insert_middle(self):
        """
        Test method (list element) "insert".
        """
        a_list = singly_linked_list.SinglyLinkedList()
        elem5 = singly_linked_list.SinglyLinkedListElement(a_list, 'e', None)
        elem4 = singly_linked_list.SinglyLinkedListElement(a_list, 'd', elem5)
        elemx = singly_linked_list.SinglyLinkedListElement(a_list, 'cc', elem4)
        elem3 = singly_linked_list.SinglyLinkedListElement(a_list, 'c', elem4)
        elem2 = singly_linked_list.SinglyLinkedListElement(a_list, 'b', elem3)
        elem1 = singly_linked_list.SinglyLinkedListElement(a_list, 'a', elem2)

        a_list.append(elem1.get_data())
        a_list.append(elem2.get_data())
        a_list.append(elem3.get_data())
        a_list.append(elem4.get_data())
        a_list.append(elem5.get_data())

        elem3.insert('cc')

        res = []
        res.append(a_list[0])
        res.append(a_list[1])
        res.append(a_list[2])
        res.append(a_list[3])
        res.append(a_list[4])

        ref = []
        ref.append(elem1)
        ref.append(elem2)
        ref.append(elemx)
        ref.append(elem4)
        ref.append(elem5)

        t_1 = a_list.get_head() == a_list[0]
        t_2 = a_list.get_tail() == a_list[4]
        t_3 = a_list[0].get_next() == a_list[1]
        t_4 = a_list[1].get_next() == a_list[2]
        t_5 = a_list[2].get_next() == a_list[3]
        t_6 = a_list[3].get_next() == a_list[4]
        t_7 = a_list[4].get_next() == None
        t_8 = len(a_list) == 5
        t_9 = ref == res
        test = t_1 and t_2 and t_3 and t_4 and t_5 and t_6 and t_7 and t_8 and t_9
        self.assertTrue(test)

    def test_singly_linked_list_element_insert_tail(self):
        """
        Test method (list element) "insert".
        """
        a_list = singly_linked_list.SinglyLinkedList()
        elem1 = singly_linked_list.SinglyLinkedListElement(a_list, 'a', None)
        a_list.append(elem1.get_data())

        elem1.insert('aa')

        test1 = a_list.get_head() == a_list[0]
        test2 = a_list.get_tail() == a_list[0]
        test3 = a_list[0].get_next() == None
        test4 = len(a_list) == 1
        test = test1 and test2 and test3 and test4
        self.assertTrue(test)

    def test_singly_linked_list_element_insert_before_first_one(self):
        """
        Testing inserting a linked list element into a linked
        list. In this test the linked list contains a single
        element prior to the insertion of the second element
        and the new element is inserted before the first element.

        Before inserting:
        list = [b]
        After inserting:
        list = [a] -> [b]
        """
        a_list = singly_linked_list.SinglyLinkedList()
        elem1 = singly_linked_list.SinglyLinkedListElement(a_list, 'b', None)
        elemx = singly_linked_list.SinglyLinkedListElement(a_list, 'a', elem1)
        a_list.append(elem1.get_data())

        elem1.insert_before('a')

        res = []
        res.append(a_list[0])
        res.append(a_list[1])

        ref = []
        ref.append(elemx)
        ref.append(elem1)

        t_1 = a_list.get_head() == a_list[0]
        t_2 = a_list.get_tail() == a_list[1]
        t_3 = a_list[0].get_next() == a_list[1]
        t_4 = a_list[1].get_next() == None
        t_5 = len(a_list) == 2
        t_6 = ref == res
        test = t_1 and t_2 and t_3 and t_4 and t_5 and t_6
        self.assertTrue(test)

    def test_singly_linked_list_element_insert_before_first_two(self):
        """
        Testing inserting a linked list element into a linked
        list. In this test the linked list contains two elements
        prior to the insertion of the third element and the new
        element is inserted before the first element.

        Before inserting:
        list = [b] -> [c]
        After inserting:
        list = [a] -> [b] -> [c]
        """
        a_list = singly_linked_list.SinglyLinkedList()
        elem2 = singly_linked_list.SinglyLinkedListElement(a_list, 'c', None)
        elem1 = singly_linked_list.SinglyLinkedListElement(a_list, 'b', elem2)
        elemx = singly_linked_list.SinglyLinkedListElement(a_list, 'a', elem1)

        a_list.append(elem1.get_data())
        a_list.append(elem2.get_data())

        elem1.insert_before('a')

        res = []
        res.append(a_list[0])
        res.append(a_list[1])
        res.append(a_list[2])

        ref = []
        ref.append(elemx)
        ref.append(elem1)
        ref.append(elem2)

        t_1 = a_list.get_head() == a_list[0]
        t_2 = a_list.get_tail() == a_list[2]
        t_3 = a_list[0].get_next() == a_list[1]
        t_4 = a_list[1].get_next() == a_list[2]
        t_5 = len(a_list) == 3
        t_6 = ref == res
        test = t_1 and t_2 and t_3 and t_4 and t_5 and t_6
        self.assertTrue(test)

    def test_singly_linked_list_element_insert_before_middle(self):
        """
        Testing inserting a linked list element into a linked
        list. In this test the linked list contains three elements
        prior to the insertion of the fourh element and the new
        element is inserted before the third element.

        Before inserting:
        list = [a] -> [b] -> [d]
        After inserting:
        list = [a] -> [b] -> [c] -> [d]
        """
        a_list = singly_linked_list.SinglyLinkedList()
        elem2 = singly_linked_list.SinglyLinkedListElement(a_list, 'd', None)
        elemx = singly_linked_list.SinglyLinkedListElement(a_list, 'c', elem2)
        elem1 = singly_linked_list.SinglyLinkedListElement(a_list, 'b', elemx)
        elem0 = singly_linked_list.SinglyLinkedListElement(a_list, 'a', elem1)

        a_list.append(elem0.get_data())
        a_list.append(elem1.get_data())
        a_list.append(elem2.get_data())

        elem2.insert_before('c')

        res = []
        res.append(a_list[0])
        res.append(a_list[1])
        res.append(a_list[2])
        res.append(a_list[3])

        ref = []
        ref.append(elem0)
        ref.append(elem1)
        ref.append(elemx)
        ref.append(elem2)

        t_1 = a_list.get_head() == a_list[0]
        t_2 = a_list.get_tail() == a_list[3]
        t_3 = a_list[0].get_next() == a_list[1]
        t_4 = a_list[1].get_next() == a_list[2]
        t_5 = a_list[2].get_next() == a_list[3]
        t_6 = a_list[3].get_next() == None
        t_7 = len(a_list) == 4
        t_8 = ref == res
        test = t_1 and t_2 and t_3 and t_4 and t_5 and t_6 and t_7 and t_8
        self.assertTrue(test)

    def test_singly_linked_list_element_insert_after_first_one(self):
        """
        Testing inserting a linked list element into a linked
        list. In this test the linked list contains a single
        element prior to the insertion of the second element
        and the new element is inserted after this element.

        Before inserting:
        list = [a]
        After inserting:
        list = [a] -> [b]
        """
        a_list = singly_linked_list.SinglyLinkedList()
        elem0 = singly_linked_list.SinglyLinkedListElement(a_list, 'a', None)
        elemx = singly_linked_list.SinglyLinkedListElement(a_list, 'b', None)

        a_list.append(elem0.get_data())

        elem0.insert_after('b')

        res = []
        res.append(a_list[0])
        res.append(a_list[1])

        ref = []
        ref.append(elem0)
        ref.append(elemx)

        t_1 = a_list.get_head() == a_list[0]
        t_2 = a_list.get_tail() == a_list[1]
        t_3 = a_list[0].get_next() == a_list[1]
        t_4 = a_list[1].get_next() == None
        t_5 = len(a_list) == 2
        t_6 = ref == res
        test = t_1 and t_2 and t_3 and t_4 and t_5 and t_6
        self.assertTrue(test)

    def test_singly_linked_list_element_insert_after_first_two(self):
        """
        Testing inserting a linked list element into a linked
        list. In this test the linked list contains two elements
        prior to the insertion of the third element and the new
        element is inserted after the second element.

        Before inserting:
        list = [a] -> [b]
        After inserting:
        list = [a] -> [b] -> [c]
        """
        a_list = singly_linked_list.SinglyLinkedList()
        elemx = singly_linked_list.SinglyLinkedListElement(a_list, 'c', None)
        elem1 = singly_linked_list.SinglyLinkedListElement(a_list, 'b', elemx)
        elem0 = singly_linked_list.SinglyLinkedListElement(a_list, 'a', elem1)

        a_list.append(elem0.get_data())
        a_list.append(elem1.get_data())

        elem1.insert_after('c')

        res = []
        res.append(a_list[0])
        res.append(a_list[1])
        res.append(a_list[2])

        ref = []
        ref.append(elem0)
        ref.append(elem1)
        ref.append(elemx)

        t_1 = a_list.get_head() == a_list[0]
        t_2 = a_list.get_tail() == a_list[2]
        t_3 = a_list[0].get_next() == a_list[1]
        t_4 = a_list[1].get_next() == a_list[2]
        t_5 = a_list[2].get_next() == None
        t_6 = len(a_list) == 3
        t_7 = ref == res
        test = t_1 and t_2 and t_3 and t_4 and t_5 and t_6 and t_7
        self.assertTrue(test)

    def test_singly_linked_list_element_insert_after_middle(self):
        """
        Testing inserting a linked list element into a linked
        list. In this test the linked list contains three elements
        prior to the insertion of the fourh element and the new
        element is inserted after the third element.

        Before inserting:
        list = [a] -> [b] -> [c]
        After inserting:
        list = [a] -> [b] -> [c] -> [d]
        """
        a_list = singly_linked_list.SinglyLinkedList()
        elem2 = singly_linked_list.SinglyLinkedListElement(a_list, 'd', None)
        elemx = singly_linked_list.SinglyLinkedListElement(a_list, 'c', elem2)
        elem1 = singly_linked_list.SinglyLinkedListElement(a_list, 'b', elemx)
        elem0 = singly_linked_list.SinglyLinkedListElement(a_list, 'a', elem1)

        a_list.append(elem0.get_data())
        a_list.append(elem1.get_data())
        a_list.append(elem2.get_data())

        elem1.insert_after('c')

        res = []
        res.append(a_list[0])
        res.append(a_list[1])
        res.append(a_list[2])
        res.append(a_list[3])

        ref = []
        ref.append(elem0)
        ref.append(elem1)
        ref.append(elemx)
        ref.append(elem2)

        t_1 = a_list.get_head() == a_list[0]
        t_2 = a_list.get_tail() == a_list[3]
        t_3 = a_list[0].get_next() == a_list[1]
        t_4 = a_list[1].get_next() == a_list[2]
        t_5 = a_list[2].get_next() == a_list[3]
        t_6 = len(a_list) == 4
        t_7 = ref == res
        test = t_1 and t_2 and t_3 and t_4 and t_5 and t_6 and t_7
        self.assertTrue(test)

    def test_singly_linked_list_element_remove_first_one(self):
        """
        Testing removing a linked list element from a linked
        list. In this test the linked list contains a single
        element prior to removing this element.

        Before removing:
        list = [a]
        After removing:
        list = [None]
        """
        a_list = singly_linked_list.SinglyLinkedList()
        elem0 = singly_linked_list.SinglyLinkedListElement(a_list, 'a', None)

        a_list.append(elem0.get_data())

        elem0.remove()

        res = []
        ref = []
        t_1 = len(a_list) == 0
        t_2 = ref == res
        test = t_1 and t_2
        self.assertTrue(test)

    def test_singly_linked_list_element_remove_first_two(self):
        """
        Testing removing a linked list element from a linked
        list. In this test the linked list contains two
        elements prior to removing the first element.

        Before removing:
        list = [a] -> [b]
        After removing:
        list = [b]
        """
        a_list = singly_linked_list.SinglyLinkedList()
        elem1 = singly_linked_list.SinglyLinkedListElement(a_list, 'b', None)
        elem0 = singly_linked_list.SinglyLinkedListElement(a_list, 'a', elem1)

        a_list.append(elem0.get_data())
        a_list.append(elem1.get_data())

        elem0.remove()

        res = []
        res.append(a_list[0])

        ref = []
        ref.append(elem1)

        t_1 = a_list.get_head() == a_list[0]
        t_2 = a_list.get_tail() == a_list[0]
        t_3 = elem1.get_next() == None
        t_4 = len(a_list) == 1
        t_5 = ref == res
        test = t_1 and t_2 and t_3 and t_4 and t_5
        self.assertTrue(test)

    def test_singly_linked_list_element_remove_middle(self):
        """
        Testing removing a linked list element from a linked
        list. In this test the linked list contains five
        elements prior to removing. The element being removed
        is the third element.

        Before removing:
        list = [a] -> [b] -> [c] -> [d] -> [e]
        After removing:
        list = [a] -> [b] -> [d] -> [e]
        """
        a_list = singly_linked_list.SinglyLinkedList()
        elem5 = singly_linked_list.SinglyLinkedListElement(a_list, 'e', None)
        elem4 = singly_linked_list.SinglyLinkedListElement(a_list, 'd', elem5)
        elem3 = singly_linked_list.SinglyLinkedListElement(a_list, 'c', elem4)
        elem2 = singly_linked_list.SinglyLinkedListElement(a_list, 'b', elem3)
        elem1 = singly_linked_list.SinglyLinkedListElement(a_list, 'a', elem2)

        a_list.append(elem1.get_data())
        a_list.append(elem2.get_data())
        a_list.append(elem3.get_data())
        a_list.append(elem4.get_data())
        a_list.append(elem5.get_data())

        elem3.remove()

        res = []
        res.append(a_list[0])
        res.append(a_list[1])
        res.append(a_list[2])
        res.append(a_list[3])

        ref = []
        ref.append(elem1)
        ref.append(elem2)
        ref.append(elem4)
        ref.append(elem5)

        t_1 = a_list.get_head() == a_list[0]
        t_2 = a_list.get_tail() == a_list[3]
        t_3 = a_list[0].get_next() == a_list[1]
        t_4 = a_list[1].get_next() == a_list[2]
        t_5 = a_list[2].get_next() == a_list[3]
        t_6 = a_list[3].get_next() == None
        t_7 = len(a_list) == 4
        t_8 = ref == res
        test = t_1 and t_2 and t_3 and t_4 and t_5 and t_6 and t_7 and t_8
        self.assertTrue(test)

    def test_singly_linked_list_element_remove_end(self):
        """
        Testing removing a linked list element from a linked
        list. In this test the linked list contains five
        elements prior to removing. The element being removed
        is the last element.

        Before removing:
        list = [a] -> [b] -> [c] -> [d] -> [e]
        After removing:
        list = [a] -> [b] -> [c] -> [d]
        """
        a_list = singly_linked_list.SinglyLinkedList()
        elem4 = singly_linked_list.SinglyLinkedListElement(a_list, 'e', None)
        elem3 = singly_linked_list.SinglyLinkedListElement(a_list, 'd', elem4)
        elem2 = singly_linked_list.SinglyLinkedListElement(a_list, 'c', elem3)
        elem1 = singly_linked_list.SinglyLinkedListElement(a_list, 'b', elem2)
        elem0 = singly_linked_list.SinglyLinkedListElement(a_list, 'a', elem1)

        a_list.append(elem0.get_data())
        a_list.append(elem1.get_data())
        a_list.append(elem2.get_data())
        a_list.append(elem3.get_data())
        a_list.append(elem4.get_data())

        elem4.remove()

        res = []
        res.append(a_list[0])
        res.append(a_list[1])
        res.append(a_list[2])
        res.append(a_list[3])

        ref = []
        ref.append(elem0)
        ref.append(elem1)
        ref.append(elem2)
        ref.append(elem3)

        t_1 = a_list.get_head() == a_list[0]
        t_2 = a_list.get_tail() == a_list[3]
        t_3 = a_list[0].get_next() == a_list[1]
        t_4 = a_list[1].get_next() == a_list[2]
        t_5 = a_list[2].get_next() == a_list[3]
        t_6 = a_list[3].get_next() == None
        t_7 = len(a_list) == 4
        t_8 = ref == res
        test = t_1 and t_2 and t_3 and t_4 and t_5 and t_6 and t_7 and t_8
        self.assertTrue(test)

    def test_singly_linked_list_element_remove_not_present(self):
        """
        Testing removing a linked list element from a linked
        list. In this test the linked list contains two
        elements prior to removing. The element which should
        be removed is not in the list, so nothing should be
        removed and the pointers should be intact.

        Before removing:
        list = [a] -> [b]
        After removing:
        list = [a] -> [b]
        """

        a_list = singly_linked_list.SinglyLinkedList()
        elem2 = singly_linked_list.SinglyLinkedListElement(a_list, 'c', None)
        elem1 = singly_linked_list.SinglyLinkedListElement(a_list, 'b', elem2)
        elem0 = singly_linked_list.SinglyLinkedListElement(a_list, 'a', elem1)

        a_list.append(elem0.get_data())
        a_list.append(elem1.get_data())

        elem2.remove()

        res = []
        res.append(a_list[0])
        res.append(a_list[1])

        ref = []
        ref.append(elem0)
        ref.append(elem1)
        t_1 = a_list.get_head() == a_list[0]
        t_2 = a_list.get_tail() == a_list[1]
        t_3 = a_list[0].get_next() == a_list[1]
        t_4 = a_list[1].get_next() == None
        t_5 = len(a_list) == 2
        t_6 = ref == res
        test = t_1 and t_2 and t_3 and t_4 and t_5 and t_6
        self.assertTrue(test)

    ### End test of local class SinglyLinkedListElement ###

    ### Begin test of class SinglyLinkedList ###

    def test_singly_linked_list_len(self):
        """
        Test operator "len".
        """
        self.assertEqual(3, len(self.list1))

    def test_singly_linked_list_equal(self):
        """
        Test operator "equal".
        """
        a_list1 = singly_linked_list.SinglyLinkedList()
        a_list2 = singly_linked_list.SinglyLinkedList()
        a_list1.append('a')
        a_list1.append('b')
        a_list1.append('c')
        a_list2.append('a')
        a_list2.append('b')
        a_list2.append('c')
        self.assertEqual(a_list1, a_list2)

    def test_singly_linked_list_not_equal(self):
        """
        Test operator "equal" - inverted.
        """
        a_list1 = singly_linked_list.SinglyLinkedList()
        a_list2 = singly_linked_list.SinglyLinkedList()
        a_list1.append('a')
        a_list1.append('b')
        a_list1.append('c')
        a_list2.append('a')
        a_list2.append('b')
        a_list2.append('d')
        self.assertNotEqual(a_list1, a_list2)

    def test_singly_linked_list_copy_not_equal(self):
        """
        Test operator "copy" - inverted.
        """
        a_list1 = singly_linked_list.SinglyLinkedList()
        a_list1.append('a')
        a_list1.append('b')
        a_list1.append('c')
        a_list2 = copy.copy(a_list1)
        a_list1[len(a_list1) - 1] = 'cc'
        self.assertNotEqual(a_list1, a_list2)

    def test_singly_linked_list_copy_equal(self):
        """
        Test operator "copy".
        """
        a_list1 = singly_linked_list.SinglyLinkedList()
        a_list1.append('a')
        a_list1.append('b')
        a_list1.append('c')
        a_list2 = copy.copy(a_list1)
#         print ""
#         print l1
#         print l2
#         print len( l1 )
#         print len( l2 )
#         print l1[0]
#         print l2[0]
#         print l1[1]
#         print l2[1]
#         print l1[2]
#         print l2[2]
#         print l1.get_head()
#         print l2.get_head()
#         print l1.get_tail()
#         # NOTE: it appears that the tail is different!!!
#         print l2.get_tail()
        self.assertEqual(a_list1, a_list2)

    def test_singly_linked_list_contains(self):
        """
        Test operator "contains".
        """
        self.assertTrue('b' in self.list1)

    def test_singly_linked_list_contains_not(self):
        """
        Test operator "contains" - inverted.
        """
        self.assertFalse('bb' in self.list1)

    def test_singly_linked_list_get_item(self):
        """
        Test operator "get_item".
        """
        elem = singly_linked_list.SinglyLinkedListElement(self.list1, 'b', None)
        self.assertEqual(elem, self.list1[0])

    def test_singly_linked_list_get_item_raise(self):
        """
        Test operator "get_item" - raises exception.
        """
        self.assertRaises(IndexError, lambda: self.list1[10])

    def test_singly_linked_list_get_head(self):
        """
        Test method "get_head".
        """
        self.assertEqual('b', self.list1.get_head().get_data())

    def test_singly_linked_list_get_tail(self):
        """
        Test method "get_tail".
        """
        self.assertEqual('d', self.list1.get_tail().get_data())

    def test_singly_linked_list_is_empty(self):
        """
        Test method "is_empty".
        """
        a_list = singly_linked_list.SinglyLinkedList()
        self.assertTrue(a_list.is_empty())

    def test_singly_linked_list_is_empty_not(self):
        """
        Test method "is_empty" - inverted.
        """
        self.assertFalse(self.list1.is_empty())

    def test_singly_linked_list_clear(self):
        """
        Test method "clear".
        """
        a_list = singly_linked_list.SinglyLinkedList()
        a_list.append('a')
        a_list.clear()
        self.assertTrue(a_list.is_empty())

    def test_singly_linked_list_get_first(self):
        """
        Test method "get_first".
        """
        self.assertEqual('b', self.list1.get_first())

    def test_singly_linked_list_get_last(self):
        """
        Test method "get_last".
        """
        self.assertEqual('d', self.list1.get_last())

    def test_singly_linked_list_prepend(self):
        """
        Test method "get_prepend".
        """
        self.list1.prepend('a')
        self.assertEqual('a', self.list1.get_first())

    def test_singly_linked_list_insert_at_empty(self):
        """
        Test method "insert_at".
        """
        a_list = singly_linked_list.SinglyLinkedList()
        a_list.insert_at(0, 'b')
        self.assertEqual('b', a_list[0].get_data())

    def test_singly_linked_list_insert_at_head(self):
        """
        Test method "insert_at".
        """
        a_list = singly_linked_list.SinglyLinkedList()
        a_list.append('a')
        a_list.append('b')
        a_list.append('c')
        a_list.insert_at(0, 'aa')
        t_1 = 'aa' == a_list.get_head().data
        t_2 = a_list[0] == a_list.get_head()
        t_3 = a_list[len(a_list) - 1] == a_list.get_tail()
        test = t_1 and t_2 and t_3
        self.assertTrue(test)

    def test_singly_linked_list_insert_at_middle(self):
        """
        Test method "insert_at".
        """
        a_list = singly_linked_list.SinglyLinkedList()
        a_list.append('a')
        a_list.append('b')
        a_list.append('c')
        a_list.append('d')
        a_list.append('e')
        a_list.insert_at(2, 'cc')
        self.assertEqual('cc', a_list[2].get_data())

    def test_singly_linked_list_insert_at_tail(self):
        """
        Test method "insert_at".
        """
        a_list = singly_linked_list.SinglyLinkedList()
        a_list.append('a')
        a_list.append('b')
        a_list.append('c')
        a_list.insert_at(len(a_list) - 1, 'cc')
        t_1 = a_list[0] == a_list.get_head()
        t_2 = a_list[len(a_list) - 1] == a_list.get_tail()
        t_3 = 'cc' == a_list.get_tail().data
        t_4 = len(a_list) == 3
        test = t_1 and t_2 and t_3 and t_4
        self.assertTrue(test)

    def test_singly_linked_list_insert_before_element(self):
        """
        Test method "insert_before_element".
        """
        a_list = singly_linked_list.SinglyLinkedList()
        a_list.append('a')
        a_list.append('b')
        a_list.append('c')
        elem2 = a_list[2]
        a_list.insert_before_element('cc', elem2)
        self.assertEqual('cc', a_list[2].get_data())

    def test_singly_linked_list_insert_after_element(self):
        """
        Test method "insert_after_element".
        """
        a_list = singly_linked_list.SinglyLinkedList()
        a_list.append('a')
        a_list.append('b')
        a_list.append('c')
        elem2 = a_list[2]
        a_list.insert_after_element('cc', elem2)
        self.assertEqual('cc', a_list[3].get_data())

    def test_singly_linked_list_remove(self):
        """
        Test method "remove".
        """
        self.list1.remove('d')
        self.assertEqual('c', self.list1.get_last())

    ### End test of class SinglyLinkedList ###

