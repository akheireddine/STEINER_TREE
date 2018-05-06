#!/usr/bin/env py.test

"""
Test of ArrayList - a dynamic list.
"""

import unittest
import copy

from py_alg_dat import array_list
from py_alg_dat import association
from py_alg_dat import singly_linked_list

class TestArrayList(unittest.TestCase):

    """
    Testing the ArrayList class.
    """

    def setUp(self):
        """
        Setup global test variables.
        """
        self.array1 = array_list.ArrayList(5)
        self.array1[0] = 1
        self.array1[1] = 2
        self.array1[2] = 3
        self.array1[3] = 4
        self.array1[4] = 5

        self.array2 = array_list.ArrayList(5)
        self.array2[0] = 1
        self.array2[1] = 2
        self.array2[2] = 3
        self.array2[3] = 4
        self.array2[4] = 5

        self.array3 = array_list.ArrayList(2)
        self.array3[0] = 1
        self.array3[1] = 2

    def test_array_list_len(self):
        """
        Test of method returning the length of the array list.
        """
        self.assertEqual(5, len(self.array1))

    def test_array_list_equal(self):
        """
        Test of equal operator.
        """
        self.assertEqual(self.array1, self.array2)

    def test_array_list_not_equal(self):
        """
        Test of inequal operator.
        """
        self.assertNotEqual(self.array1, self.array3)

    def test_array_list_copy(self):
        """
        Test of shallow copy method.
        """
        acopy = copy.copy(self.array1)
        self.assertEqual(acopy, self.array1)

    def test_array_list_get_item(self):
        """
        Test of get method.
        """
        self.assertEqual(3, self.array1[2])

    def test_array_list_get_item_slice(self):
        """
        Test of get method - slice.
        """
        a_array = array_list.ArrayList(5)
        a_array[0] = 10
        a_array[1] = 20
        a_array[2] = 30
        a_array[3] = 40
        a_array[4] = 50
        self.assertEqual(a_array.get_data(), a_array[:])

    def test_array_list_set_item(self):
        """
        Test of set method.
        """
        self.array3[1] = 100
        self.assertEqual(100, self.array3[1])

    def test_array_list_get_offset(self):
        """
        Test of get offset method.
        """
        self.assertEqual(0, self.array1.get_offset(0))

    def test_array_list_get_data(self):
        """
        Test of get data method.
        """
        self.assertEqual([1, 2, 3, 4, 5], self.array1.get_data())

    def test_array_list_get_base_index(self):
        """
        Test of get base index method.
        """
        self.assertEqual(0, self.array1.get_base_index())

    def test_array_list_set_base_index(self):
        """
        Test of set base index method.
        """
        self.array1.set_base_index(1)
        self.assertEqual(1, self.array1.get_base_index())

    def test_array_list_set_len(self):
        """
        Test of set length method.
        """
        self.array1.set_length(3)
        self.assertEqual(3, len(self.array1))

    def test_array_list_remove_at_1(self):
        """
        Test of remove at method.
        """
        a_array = array_list.ArrayList(5)
        a_array[0] = 10
        a_array[1] = 20
        a_array[2] = 30
        a_array[3] = 40
        a_array[4] = 50
        ref = [a_array[0], a_array[1], a_array[3], a_array[4]]
        del a_array[2]
        self.assertEqual(ref, a_array.get_data())

    def test_array_list_remove_at_2(self):
        """
        Test of remove at methhod (association).
        """
        array = array_list.ArrayList(5)
        s_list1 = singly_linked_list.SinglyLinkedList()
        e11 = association.Association("e11", 11)
        e12 = association.Association("e12", 12)
        e13 = association.Association("e13", 13)
        s_list1.append(e11)
        s_list1.append(e12)
        s_list1.append(e13)

        s_list2 = singly_linked_list.SinglyLinkedList()
        e21 = association.Association("e21", 21)
        e22 = association.Association("e22", 22)
        e23 = association.Association("e23", 23)
        s_list2.append(e21)
        s_list2.append(e22)
        s_list2.append(e23)

        s_list3 = singly_linked_list.SinglyLinkedList()
        e31 = association.Association("e31", 31)
        e32 = association.Association("e32", 32)
        e33 = association.Association("e33", 33)
        s_list3.append(e31)
        s_list3.append(e32)
        s_list3.append(e33)

        s_list4 = singly_linked_list.SinglyLinkedList()
        e41 = association.Association("e41", 41)
        e42 = association.Association("e42", 42)
        e43 = association.Association("e43", 43)
        s_list4.append(e41)
        s_list4.append(e42)
        s_list4.append(e43)

        s_list5 = singly_linked_list.SinglyLinkedList()
        e51 = association.Association("e51", 51)
        e52 = association.Association("e52", 52)
        e53 = association.Association("e53", 53)
        s_list5.append(e51)
        s_list5.append(e52)
        s_list5.append(e53)

        array[0] = s_list1
        array[1] = s_list2
        array[2] = s_list3
        array[3] = s_list4
        array[4] = s_list5

        array[0].remove(e11)
        array[0].remove(e12)
        array[0].remove(e13)
        del array[0]

        ref = array_list.ArrayList(4)
        ref[0] = s_list2
        ref[1] = s_list3
        ref[2] = s_list4
        ref[3] = s_list5
        self.assertEqual(ref, array)

