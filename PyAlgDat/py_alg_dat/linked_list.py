#!/usr/bin/python env

# The MIT License (MIT)
#
# Copyright (c) 2015 by Brian Horn, trycatchhorn@gmail.com.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
Provides a data structure used to model a linked list.
"""

__author__ = "Brian Horn"
__copyright__ = "Copyright (c) 2015 Brian Horn"
__credits__ = "Brian Horn"
__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Brian Horn"
__email__ = "trycatchhorn@gmail.com"
__status__ = "Prototype"

from abc import abstractmethod
from py_alg_dat.linked_list_iterator import LinkedListIterator

class LinkedList(object):

    """
    The interface of a linked list.
    """

    def __init__(self):
        """
        Constructs a linked list which initially
        does not contain any elements.
        """
        self.size = 0
        self.head = None
        self.tail = None

    def __str__(self):
        """
        Returns a string representation of this linked list.

        @return: String representation of the linked list.
        @rtype: C{str}
        """
        str_rep = self.__class__.__name__ + ": ["
        ptr = self.head
        while ptr is not None:
            str_rep = str_rep + str(ptr.data)
            if ptr.next is not None:
                str_rep = str_rep + ", "
            ptr = ptr.next
        str_rep = str_rep + "]"
        return str_rep

    def is_equal(self, other):
        """
        Compares this linked list with the specified
        linked list. If the two linked lists are equal
        true is returned, otherwise false is returned.

        The two linked lists are compared by comparing
        the fields: size, head, and tail for equality.
        Additionally, each element in this linked list
        is compared with the element at the same
        position in the specified list and if all
        elements are equal, and the other fields match,
        the two linked lists are equal and true is
        returned, otherwise false is returned.

        @param other: The other linked list.
        @type ohter: L{LinkedList}
        @return: True if the two linked lists are equal, otherwise false.
        @rtype: C{bool}
        """
        if self.size != other.size:
            return False
        elif self.head != other.head:
            return False
        elif self.tail != other.tail:
            return False
        elem_a = self.get_head()
        elem_b = other.get_head()
        while elem_a != None and elem_b != None:
            if elem_a.get_data() != elem_b.get_data():
                return False
            elem_a = elem_a.get_next()
            elem_b = elem_b.get_next()
        return True

    def do_copy(self, other):
        """
        Performs a shallow copy of this linked list. The copy is
        made by copying the individual fields of this linked list
        into a new linked list object, where the linked list type
        is specified by the other parameter.

        NOTE: This method is called by all __copy__ methods in
        the linked list inheritance hierarchy to avoid the
        redundancy introduced if this functionalty was placed
        directly in each of the __copy__ methods in each subclass.

        @param other: Specifies the type of linked list.
        @type other: L{LinkedList}
        @return: A shallow copy of this linked list.
        @rtype: L{LinkedList}
        """
        # Import sub-classes
        from py_alg_dat.singly_linked_list import SinglyLinkedList
        from py_alg_dat.doubly_linked_list import DoublyLinkedList

        result = None
        if type(other) == SinglyLinkedList:
            result = SinglyLinkedList()
        elif type(other) == DoublyLinkedList:
            result = DoublyLinkedList()
        # Check that we have a instance of linked list or one of its subtypes
        if isinstance(result, LinkedList):
            result.size = self.size
            result.head = self.head
            result.tail = self.tail
            ptr = self.get_head()
            i = 0
            while ptr != None:
                result[i] = ptr.data
                ptr = ptr.next
                i += 1
        return result

    def __len__(self):
        """
        Returns the size of this linked list.

        @return: The size of the linked list.
        @rtype: C{int}
        """
        return self.size

    @abstractmethod
    def __contains__(self, item):
        """
        Returns if the specified item is present in this linked list.

        @return: True if the item is present in the linked list, false otherwise.
        @rtype: C{bool}
        """
        pass

    def __getitem__(self, index):
        """
        Returns the linked list element at the specified
        index, if present, in this linked list. If the
        element is not present IndexError is raised.

        @param index: The index of the element to search for in the linked list.
        @type: C{int}
        @return: The specified item, if present, in the linked list.
        @rtype: L{LinkedListElement}
        """
        if index >= 0 and index < self.size:
            ptr = self.head
            i = 0
            while i < self.size:
                if i == index:
                    return ptr
                ptr = ptr.next
                i += 1
        raise IndexError

    @abstractmethod
    def __setitem__(self, index, value):
        """
        Inserts a linked list element with the specified
        value at the specified index in this linked list.
        If the index is out of range IndexError is raised.

        @param index: The index where the element is inserted in the linked list.
        @type: C{int}
        @parm value: The value of the element to be inserted in the linked list.
        @type: L{LinkedListElement}
        """
        pass

    def __iter__(self):
        """
        Returns a linked list iterator enumerating this linked list.

        @return: Linked list iterator enumerating the linked list.
        @rtype: L{LinkedListIterator}
        """
        iterator = LinkedListIterator(self.head)
        return iter(iterator)

    def get_head(self):
        """
        Returns the element at the head of this linked list.

        @return: The head element of the linked list.
        @rtype: L{LinkedListElement}
        """
        return self.head

    def get_tail(self):
        """
        Returns the element at the tail of this linked list.

        @return: The tail element of the linked list.
        @rtype: L{LinkedListElement}
        """
        return self.tail

    def is_empty(self):
        """
        Returns if this linked list is empty.

        @return: True, if the linked list is empty, false otherwise.
        @rtype: C{bool}
        """
        return self.size == 0 and self.head is None and self.tail is None

    def clear(self):
        """
        Clears all elements in this linked list.
        """
        self.size = 0
        self.head = None
        self.tail = None

    def get_first(self):
        """
        Returns the first element in this linked list

        @return: The first element in the linked list.
        @rtype: C{object}
        """
        if self.head is None:
            raise KeyError
        return self.head.data

    def get_last(self):
        """
        Returns the last element in this linked list

        @return: The last element in the linked list.
        @rtype: C{object}
        """
        if self.tail is None:
            raise KeyError
        return self.tail.data

    @abstractmethod
    def prepend(self, item):
        """
        Inserts the specified item at the front of
        this linked list.

        @param item: The item to be inserted in the linked list.
        @type: C{object}
        """
        pass

    @abstractmethod
    def append(self, item):
        """
        Inserts the specified item at the back of
        this linked list.

        @param item: The item to be inserted in the linked list.
        @type: C{object}
        """
        pass

    @abstractmethod
    def insert_at(self, item, index):
        """
        Inserts the specified item at the specified index in
        this linked list. If an element is already present
        at the specified index, this element is replaced by
        the new element. If the linked list is empty prior to
        inserting the new element, the new element is inserted
        in front of the linked list.

        @param item: The item to be inserted in the linked list.
        @type: C{object}
        @param index: The index where the item should be inserted.
        @type: C{int}
        """
        pass

    @abstractmethod
    def insert_before_element(self, item, element):
        """
        Inserts the specified item before the specified element
        in this linked list.

        @param item: The item to be inserted in the linked list.
        @type: C{object}
        @param element: The element where the item should be inserted before.
        @type: L{LinkedListElement}
        """
        pass

    @abstractmethod
    def insert_after_element(self, item, element):
        """
        Inserts the specified item after the specified element
        in this linked list.

        @param item: The item to be inserted in the linked list.
        @type: C{object}
        @param element: The element where the item should be inserted after.
        @type: L{LinkedListElement}
        """
        pass

    @abstractmethod
    def remove(self, item):
        """
        Remove the specified item from this linked list.

        @param item: The item to be removed from the linked list.
        @type: C{object}
        """
        pass










