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
Provides a data structure used to model a doubly linked list.
"""

from py_alg_dat.linked_list import LinkedList

__author__ = "Brian Horn"
__copyright__ = "Copyright (c) 2015 Brian Horn"
__credits__ = "Brian Horn"
__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Brian Horn"
__email__ = "trycatchhorn@gmail.com"
__status__ = "Prototype"

class DoublyLinkedList(LinkedList):

    """
    The interface of a doubly linked list.
    """

    def __init__(self):
        """
        Constructs a doubly linked list which initially
        does not contain any elements.
        """
        super(DoublyLinkedList, self).__init__()

    def __contains__(self, item):
        """
        Returns if the specified item is present in this doubly linked list.

        @return: True if the item is present in the doubly linked list, false otherwise.
        @rtype: C{bool}
        """
        tmp = DoublyLinkedListElement(self, item, None, None)
        for elem in self:
            if elem == tmp:
                return True
        return False

    def __eq__(self, other):
        """
        Compares two doubly linked lists for equality. The comparison
        is done by comparing each element contained in the two lists
        to each other. In order for the two doubly linked list to be
        equal, the two lists must contain the same elements and the
        elements must be located at the same positions within the
        lists.

        NOTE: the comparison is done in the super class, in order to
        avoid having duplicated code in the two '__eq__' methods in
        the classes SinglyLinkedList and DoublyLinkedList.

        @param other: The other doubly linked list.
        @type other: L{DoublyLinkedList}
        @return: True if the double linked lists are equal, false otherwise.
        @rtype: C{bool}
        """
        if isinstance(other, DoublyLinkedList):
            return self.is_equal(other)
        return NotImplemented

    def __ne__(self, other):
        """
        Compares two doubly linked lists for inequality. The comparison
        is done by comparing each element contained in the two lists
        to each other. In order for the two doubly linked list to be
        inequal, the two lists must not contain the same elements.

        @param other: The other doubly linked list.
        @type other: L{DoublyLinkedList}
        @return: True if the double linked lists are inequal, false otherwise.
        @rtype: C{bool}
        """
        return not self == other

    def __copy__(self):
        """
        Performs a shallow copy of this doubly linked list. The copy
        is made by copying the individual fields of this doubly linked
        list into a new doubly linked list object.

        NOTE: the actual copy is made in the super class, in order to
        avoid having duplicated code in the two '__copy__' methods in
        the classes SinglyLinkedList and DoublyLinkedList.

        @return: A shallow copy of this doubly linked list.
        @rtype: L{DoublyLinkedList}
        """
        return self.do_copy(DoublyLinkedList())

    def __setitem__(self, index, value):
        """
        Inserts the specified item at the specified index in
        this doubly linked list.

        @param item: The item to be inserted in the doubly linked list.
        @type: C{object}
        @param index: The index where the item should be inserted.
        @type: C{int}
        """
        self.insert_at(index, value)

    def prepend(self, item):
        """
        Inserts the specified item at the front of
        this doubly linked list.

        @param item: The item to be inserted in the doubly linked list.
        @type: C{object}
        """
        tmp = DoublyLinkedListElement(self, item, None, self.head)
        if self.head is None:
            self.tail = tmp
        else:
            self.head.prev = tmp
        self.head = tmp
        self.size += 1

    def append(self, item):
        """
        Inserts the specified item at the back of
        this doubly linked list.

        @param item: The item to be inserted in the doubly linked list.
        @type: C{object}
        """
        tmp = DoublyLinkedListElement(self, item, self.tail, None)
        if self.head is None:
            self.head = tmp
        else:
            self.tail.next = tmp
        self.tail = tmp
        self.size += 1

    def insert_at(self, item, index):
        """
        Inserts the specified item at the specified index in
        this doubly linked list.

        @param item: The item to be inserted in the doubly linked list.
        @type: C{object}
        @param index: The index where the item should be inserted.
        @type: C{int}
        """
        i = 0
        ptr = self.head
        while ptr is not None and ptr.data is not None:
            if i == index:
                ptr.insert(item)
            ptr = ptr.next
            i += 1

    def insert_before_element(self, item, element):
        """
        Inserts the specified item before the specified element
        in this doubly linked list.

        @param item: The item to be inserted in the doubly linked list.
        @type: C{object}
        @param element: The element where the item should be inserted before.
        @type: L{DoublyLinkedListElement}
        """
        if item is not None and element is not None:
            element.insert_before(item)
            self.size += 1
        else:
            raise IndexError

    def insert_between_elements(self, item, left, right):
        """
        Inserts the specified item between the specified elements
        in this doubly linked list.

        @param item: The item to be inserted in the doubly linked list.
        @type: C{object}
        @param left: The element to the left of where the item should be inserted before.
        @type: L{DoublyLinkedListElement}
        @param right: The element to the right of where the item should be inserted before.
        @type: L{DoublyLinkedListElement}
        """
        if item is not None and left is not None and right is not None:
            left.insert_between(item, right)
            self.size += 1
        else:
            raise IndexError

    def insert_after_element(self, item, element):
        """
        Inserts the specified item aftere the specified element
        in this doubly linked list.

        @param item: The item to be inserted in the doubly linked list.
        @type: C{object}
        @param element: The element where the item should be inserted after.
        @type: L{DoublyLinkedListElement}
        """
        if item is not None and element is not None:
            element.insert_after(item)
            self.size += 1
        else:
            raise IndexError

    def remove(self, item):
        """
        Removes the specified item from this doubly linked list.

        @param item: The item to be removed from the doubly linked list.
        @type: C{object}
        """
        ptr = self.head
        prev_ptr = None
        while ptr is not None and ptr.data is not item:
            prev_ptr = ptr
            ptr = ptr.next
        if ptr is None:
            raise KeyError
        if ptr == self.head:
            self.head = ptr.next
        else:
            prev_ptr.next = ptr.next
        if ptr == self.tail:
            self.tail = prev_ptr
        self.size -= 1

class DoublyLinkedListElement(object):

    """
    Class representing an element in a doubly linked list.

    NOTE: it is possible to insert/delete elements in a doubly
    linked list using this class. However, it is prefered -and
    more convenient to use the DoublyLinkedList class for this.
    """

    def __init__(self, linked_list, data, prev_elem, next_elem):
        """
        Constructs a doubly linked list element with
        the specified values.

        @param list: The doubly linked list containing the element.
        @type: C{DoublyLinkedList}
        @param data: The data contained in the doubly linked list element.
        @type: C{object}
        @param next: The next element in the doubly linked list.
        @type: L{DoublyLinkedListElement}
        """
        self.list = linked_list
        self.data = data
        self.prev = prev_elem
        self.next = next_elem

    def __repr__(self):
        """
        Returns a canonical representation of this doubly linked list element.

        @return: Canonocal representation of the doubly linked list element.
        @rtype: C{str}
        """
        return repr(self.data)

    def __str__(self):
        """
        Returns a string representation of this doubly linked list element.

        @return: String representation of the doubly linked list element.
        @rtype: C{str}
        """
        return str(self.__class__.__name__) + ": " + str(self.data)

    def __hash__(self):
        """
        Returns the hash of this doubly linked list element.

        @return: The hash of the doubly linked list element.
        @rtype: C{int}
        """
        return hash(self.data)

    def __eq__(self, other):
        """
        Compares two doubly linked list elements for equality. The comparison
        is done by comparing the data fields of the two doubly linked list
        elements.

        @param other: The other doubly linked list element.
        @type other: L{DoublyLinkedListElement}
        @return: True if the doubly linked list elements are equal, false otherwise.
        @rtype: C{bool}
        """
        if isinstance(other, DoublyLinkedListElement):
            return self.data == other.data
        return NotImplemented

    def __ne__(self, other):
        """
        Compares two doubly linked list elements for inequality. The comparison
        is done by comparing the data fields of the two doubly linked list
        elements.

        @param other: The other doubly linked list element.
        @type other: L{DoublyLinkedListElement}
        @return: True if the doubly linked list elements are not equal, false otherwise.
        @rtype: C{bool}
        """
        return not self == other

    def __lt__(self, other):
        """
        Implements the 'less than' operator for this doubly linked list
        element.

        @param other: The other doubly linked list element.
        @type other: L{DoublyLinkedListElement}
        @return: True if this doubly linked list element is 'less than' the other, false otherwise.
        @rtype: C{bool}
        """
        return self.data < other.data

    def __le__(self, other):
        """
        Implements the 'less than -or equal' operator for this doubly linked list
        element.

        @param other: The other doubly linked list element.
        @type other: L{DoublyLinkedListElement}
        @return: True if this element is 'less than -or equal' to the other, false otherwise.
        @rtype: C{bool}
        """
        return self.data <= other.data

    def __gt__(self, other):
        """
        Implements the 'greater than' operator for this doubly linked list
        element.

        @param other: The other doubly linked list element.
        @type other: L{DoublyLinkedListElement}
        @return: True if this element is 'greater than' the other, false otherwise.
        @rtype: C{bool}
        """
        return self.data > other.data

    def __ge__(self, other):
        """
        Implements the 'greater than -or equal' operator for this doubly linked list
        element.

        @param other: The other doubly linked list element.
        @type other: L{DoublyLinkedListElement}
        @return: True if this element is 'greater than -or equal' to the other, false otherwise.
        @rtype: C{bool}
        """
        return self.data >= other.data

    def __copy__(self):
        """
        Creates a shallow copy of the doubly linked list element.
        The copy is made by copying all fields of this doubly
        linked list element into a new object holding the new
        doubly linked list element.

        @return: A shallow copy of this doubly linked list element.
        @rtype: L{DoublyLinkedListElement}
        """
        result = DoublyLinkedListElement(None, None, None, None)
        result.list = self.list
        result.data = self.data
        result.prev = self.prev
        result.next = self.next
        return result

    def get_data(self):
        """
        Returns the data contained in this doubly linked list element.

        @return: The data contained in the doubly linked list element.
        @rtype: C{object}
        """
        return self.data

    def get_previous(self):
        """
        Returns the previous doubly linked list element.

        @return: The previous doubly linked list element.
        @rtype: C{DoublyLinkedListElement}
        """
        return self.prev

    def get_next(self):
        """
        Returns the next doubly linked list element.

        @return: The next doubly linked list element.
        @rtype: C{DoublyLinkedListElement}
        """
        return self.next

    def insert(self, item):
        """
        Inserts the specified item into this doubly linked
        list element.

        @param item: The item to be inserted.
        @type: C{object}
        """
        element = DoublyLinkedListElement(self.list, item, None, None)
        element.prev = self.prev
        element.next = self.next
        self.data = element.data

        if self.list.head is None:
            self.list.head = element

        if self.list.head is self:
            self.list.head = element

        if self.list.tail is None:
            self.list.tail = element

        if self.list.tail is self:
            self.list.tail = element

    def insert_before(self, item):
        """
        Inserts a doubly linked list element containing the specified
        item, before this doubly linked list element.

        @param item: The item to be inserted.
        @type: C{object}
        """
        element = DoublyLinkedListElement(self.list, item, None, None)
        element.prev = self.prev
        element.next = self
        if self.list.head is self:
            self.list.head = element
        else:
            self.prev = element
            prev_ptr = self.list.head
            while prev_ptr is not None and prev_ptr.next is not self:
                prev_ptr = prev_ptr.next
            if prev_ptr is not None:
                prev_ptr.next = element

    def insert_between(self, item, right):
        """
        Inserts a doubly linked list element containing the specified
        item, between the specifed doubly linked list element and
        this doubly linked list element.

        @param item: The item to be inserted.
        @type: C{object}
        @param right: The double linked list element to the right of this element.
        @type: C{DoublyLinkedListElement}
        """
        element = DoublyLinkedListElement(self, item, None, None)
        self.next = element
        right.prev = element
        element.prev = self
        element.next = right

    def insert_after(self, item):
        """
        Inserts a doubly linked list element containing the specified
        item, after this doubly linked list element.

        @param item: The item to be inserted.
        @type: C{object}
        """
        element = DoublyLinkedListElement(self.list, item, None, None)
        element.prev = self
        element.next = self.next
        if self.list.tail is self:
            self.list.tail.next = element
        else:
            self.next = element
            prev_ptr = self.list.head
            while prev_ptr is not None and prev_ptr.next is not self:
                prev_ptr = prev_ptr.next
            if prev_ptr is not None:
                prev_ptr.next = element

    def remove(self):
        """
        Removes this doubly linked list element from the doubly linked list.
        """
        if self.next is not None:
            self.next.prev = self.prev
        if self.prev is not None:
            self.prev.next = self.next
        if self.list.head is self:
            self.list.head = self.next
        if self.list.tail is self:
            self.list.tail = self.prev
        del self


