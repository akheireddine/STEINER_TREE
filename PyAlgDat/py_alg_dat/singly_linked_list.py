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
Provides a data structure used to model a singly linked list.
"""

__author__ = "Brian Horn"
__copyright__ = "Copyright (c) 2015 Brian Horn"
__credits__ = "Brian Horn"
__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Brian Horn"
__email__ = "trycatchhorn@gmail.com"
__status__ = "Prototype"

from py_alg_dat.linked_list import LinkedList

class SinglyLinkedList(LinkedList):

    """
    The interface of a singly linked list.
    """

    def __init__(self):
        """
        Constructs a singly linked list, which initially
        does not contain any elements.
        """
        super(SinglyLinkedList, self).__init__()

    def __contains__(self, item):
        """
        Returns if the specified item is present in this linked list.

        @return: True if the item is present in the linked list, false otherwise.
        @rtype: C{bool}
        """
        tmp = SinglyLinkedListElement(self, item, None)
        for elem in self:
            if elem == tmp:
                return True
        return False

    def __eq__(self, other):
        """
        Compares two singly linked lists for equality. The comparison
        is done by comparing each element contained in the two lists
        to each other. In order for the two singly linked list to be
        equal, the two lists must contain the same elements and the
        elements must be located at the same positions within the
        lists.

        NOTE: the comparison is done in the super class, in order to
        avoid having duplicated code in the two '__eq__' methods in
        the classes SinglyLinkedList and SinglyLinkedList.

        @param other: The other singly linked list.
        @type other: L{SinglyLinkedList}
        @return: True if the single linked lists are equal, false otherwise.
        @rtype: C{bool}
        """
        if isinstance(other, SinglyLinkedList):
            return self.is_equal(other)
        return NotImplemented

    def __ne__(self, other):
        """
        Compares two singly linked lists for inequality. The comparison
        is done by comparing each element contained in the two lists
        to each other. In order for the two singly linked list to be
        inequal, the two lists must not contain the same elements.

        @param other: The other singly linked list.
        @type other: L{SinglyLinkedList}
        @return: True if the singly linked lists are inequal, false otherwise.
        @rtype: C{bool}
        """
        return not self == other

    def __setitem__(self, index, value):
        """
        Inserts the specified item at the specified index in
        this singly linked list.

        @param item: The item to be inserted in the singly linked list.
        @type: C{object}
        @param index: The index where the item should be inserted.
        @type: C{int}
        """
        self.insert_at(index, value)

    def __copy__(self):
        """
        Performs a shallow copy of this singly linked list. The copy
        is made by copying the individual fields of this singly linked
        list into a new singly linked list object.

        NOTE: the actual copy is made in the super class, in order to
        avoid having duplicated code in the two '__copy__' methods in
        the classes SinglyLinkedList and DoublyLinkedList.

        @return: A shallow copy of this singly linked list.
        @rtype: L{SinglyLinkedList}
        """
        return self.do_copy(SinglyLinkedList())

    def prepend(self, item):
        """
        Inserts the specified item at the front of
        this singly linked list.

        @param item: The item to be inserted in the singly linked list.
        @type: C{object}
        """
        tmp = SinglyLinkedListElement(self, item, self.head)
        if self.head is None:
            self.tail = tmp
        self.head = tmp
        self.size += 1

    def append(self, item):
        """
        Inserts the specified item at the back of
        this singly linked list.

        @param item: The item to be inserted in the singly linked list.
        @type: C{object}
        """
        tmp = SinglyLinkedListElement(self, item, None)
        if self.head is None:
            self.head = tmp
        else:
            self.tail.next = tmp
        self.tail = tmp
        self.size += 1

    def insert_at(self, index, item):
        """
        Inserts the specified item at the specified index in
        this singly linked list. If an element is already present
        at the specified index, this element is replaced by the
        new element. If the singly linked list is empty prior to
        inserting the new element, the new element is inserted
        in front of the singly linked list.

        @param item: The item to be inserted in the singly linked list.
        @type: C{object}
        @param index: The index where the item should be inserted.
        @type: C{int}
        """
        ptr = self.head
        if ptr is None:
            self.head = SinglyLinkedListElement(self, item, None)
            self.tail = self.head
            self.size += 1
            return
        i = 0
        while ptr is not None and ptr.data is not None:
            if i == index:
                ptr.insert(item)
            ptr = ptr.next
            i += 1

    def insert_before_element(self, item, element):
        """
        Inserts the specified item before the specified element
        in this singly linked list.

        @param item: The item to be inserted in the singly linked list.
        @type: C{object}
        @param element: The element where the item should be inserted before.
        @type: L{SinglyLinkedListElement}
        """
        if item is not None and element is not None:
            element.insert_before(item)
        else:
            raise IndexError

    def insert_after_element(self, item, element):
        """
        Inserts the specified item aftere the specified element
        in this singly linked list.

        @param item: The item to be inserted in the singly linked list.
        @type: C{object}
        @param element: The element where the item should be inserted after.
        @type: L{SinglyLinkedListElement}
        """
        if item is not None and element is not None:
            element.insert_after(item)
        else:
            raise IndexError

    def remove(self, item):
        """
        Remove the specified item from this singly linked list.

        @param item: The item to be removed from the singly linked list.
        @type: C{object}
        """
        tmp = SinglyLinkedListElement(self, item, None)
        tmp.remove()

class SinglyLinkedListElement(object):

    """
    Class representing an element in a singly linked list.

    NOTE: it is possible to insert/delete elements in a
    singly linked list using this class. However, it is prefered
    -and more convenient to use the SinglyLinkedList class for this.
    """

    def __init__(self, linked_list, data, next_elem):
        """
        Constructs a singly linked list element with
        the specified values.

        @param list: The singly linked list containing the element.
        @type: C{SinglyLinkedList}
        @param data: The data contained in the singly linked list element.
        @type: C{object}
        @param next: The next element in the singly linked list.
        @type: L{SinglyLinkedListElement}
        """
        self.list = linked_list
        self.data = data
        self.next = next_elem

    def __repr__(self):
        """
        Returns a canonical representation of this singly linked list element.

        @return: Canonocal representation of the singly linked list element.
        @rtype: C{str}
        """
        return repr(self.data)

    def __str__(self):
        """
        Returns a string representation of this singly linked list element.

        @return: String representation of the singly linked list element.
        @rtype: C{str}
        """
        return str(self.__class__.__name__) + ": " + str(self.data)

    def __hash__(self):
        """
        Returns the hash of this singly linked list element.

        @return: The hash of the singly linked list element.
        @rtype: C{int}
        """
        return hash(self.data)

    def __eq__(self, other):
        """
        Compares two singly linked list elements for equality. The comparison
        is done by comparing the data fields of the two singly linked list
        elements.

        @param other: The other singly linked list element.
        @type other: L{SinglyLinkedListElement}
        @return: True if the singly linked list elements are equal, false otherwise.
        @rtype: C{bool}
        """
        if isinstance(other, SinglyLinkedListElement):
            return self.data == other.data
        return NotImplemented

    def __ne__(self, other):
        """
        Compares two singly linked list elements for inequality. The comparison
        is done by comparing the data fields of the two singly linked list
        elements.

        @param other: The other singly linked list element.
        @type other: L{SinglyLinkedListElement}
        @return: True if the singly linked list elements are not equal, false otherwise.
        @rtype: C{bool}
        """
        return not self == other

    def __lt__(self, other):
        """
        Implements the 'less than' operator for this singly linked list
        element.

        @param other: The other singly linked list element.
        @type other: L{SinglyLinkedListElement}
        @return: True if this singly linked list element is 'less than' the other, false otherwise.
        @rtype: C{bool}
        """
        return self.data < other.data

    def __le__(self, other):
        """
        Implements the 'less than -or equal' operator for this singly linked list
        element.

        @param other: The other singly linked list element.
        @type other: L{SinglyLinkedListElement}
        @return: True if this element is 'less than -or equal' to the other, false otherwise.
        @rtype: C{bool}
        """
        return self.data <= other.data

    def __gt__(self, other):
        """
        Implements the 'greater than' operator for this singly linked list
        element.

        @param other: The other singly linked list element.
        @type other: L{SinglyLinkedListElement}
        @return: True if this element is 'greater than' the other, false otherwise.
        @rtype: C{bool}
        """
        return self.data > other.data

    def __ge__(self, other):
        """
        Implements the 'greater than -or equal' operator for this singly linked list
        element.

        @param other: The other singly linked list element.
        @type other: L{SinglyLinkedListElement}
        @return: True if this element is 'greater than -or equal' to the other, false otherwise.
        @rtype: C{bool}
        """
        return self.data >= other.data

    def __copy__(self):
        """
        Creates a shallow copy of the singly linked list element.
        The copy is made by copying all fields of this singly
        linked list element into a new object holding the new
        singly linked list element.

        @return: A shallow copy of this singly linked list element.
        @rtype: L{SinglyLinkedListElement}
        """
        result = SinglyLinkedListElement(None, None, None)
        result.list = self.list
        result.data = self.data
        result.next = self.next
        return result

    def get_data(self):
        """
        Returns the data contained in this singly linked list element.

        @return: The data contained in the singly linked list element.
        @rtype: C{object}
        """
        return self.data

    def get_next(self):
        """
        Returns the next singly linked list element.

        @return: The next singly linked list element.
        @rtype: C{SinglyLinkedListElement}
        """
        return self.next

    def insert(self, item):
        """
        Inserts the specified item into this singly linked list element.
        The element already present is overwritten by the new element. If
        the list is empty before insertion the new element is inserted at
        the front of the linked list.

        @param item: The item to be inserted.
        @type: C{object}
        """
        new_element = SinglyLinkedListElement(self.list, item, self.next)
        # The singly linked list is empty
        if self.list.head == None:
            self.list.head = new_element
            self.list.tail = new_element
            self.list.size += 1
            return

        # The singly linked list contains one element
        elif self.list.size == 1:
            if self.list.head == self:
                self.list.head = new_element
            if self.list.tail == self:
                self.list.tail = new_element
            return
        else:
            if self.list.head == self:
                self.list.head = new_element
            if self.list.tail == self:
                self.list.tail = new_element

            ptr = self.list.head
            while ptr is not None:
                if ptr.next == self:
                    if ptr.next == self.list.tail:
                        self.list.tail = new_element
                    ptr.next = new_element
                ptr = ptr.next

    def insert_before(self, item):
        """
        Inserts a singly linked list element containing the specified
        item before this singly linked list element.

        @param item: The item to be inserted.
        @type: C{object}
        """
        new_element = SinglyLinkedListElement(self.list, item, None)
        current_element = self.list.head
        was_inserted = False
        # Insertion happens before the first element
        if self == current_element:
            # This is the case where there is only one element in the list
            if current_element.next == None:
                self.list.head = new_element
                self.list.head.next = current_element
                self.list.tail = self.list.head.next
                self.list.size += 1
                return
            # Here there are more than one element in the list
            new_element.next = current_element
            self.list.head = new_element
            self.list.size += 1
            return

        while True:
            if current_element == None:
                was_inserted = False
                break
            # Check if it is the next element we are looking for
            next_element = current_element.next
            if next_element != None:
                if self == next_element:
                    # Found the right slot - insert the new element
                    new_element.next = next_element
                    current_element.next = new_element
                    was_inserted = True
                    break
            current_element = current_element.next
        if was_inserted:
            self.list.size += 1

    def insert_after(self, item):
        """
        Inserts a singly linked list element containing the specified
        item after this singly linked list element.

        @param item: The item to be inserted.
        @type: C{object}
        """
        new_element = SinglyLinkedListElement(self.list, item, None)

        if self == self.list.tail:
            self.list.tail = new_element

        current_element = self.list.head
        was_inserted = False
        # Insertion happens after the first element
        if self == current_element:
            # This is the case where there is only one element in the list
            if current_element.next == None:
                current_element.next = new_element
                self.next = new_element
                self.list.tail = current_element.next
                self.list.size += 1
                return
            # Here there are more than one element in the list
            current_element = current_element.next
            self.list.head = current_element
            self.list.size += 1
            return

        while True:
            if current_element == None:
                was_inserted = False
                break
            # Check if it is the next element we are looking for
            next_element = current_element.next
            if next_element != None:
                if self == next_element:
                    # Found the right slot - insert the new element
                    new_element.next = next_element.next
                    next_element.next = new_element
                    was_inserted = True
                    break
            current_element = current_element.next
        if was_inserted:
            self.list.size += 1

    def remove(self):
        """
        Removes this singly linked list element from the singly linked list.
        """
        current_element = self.list.head
        was_removed = False
        # The first element is being removed
        if self == current_element:
            # This is the case where there is only one element in the list
            if current_element.next == None:
                self.list.head = None
                self.list.tail = self.list.head
                self.list.size -= 1
                return
            # Here there are more than one element in the list
            current_element = current_element.next
            self.list.head = current_element
            self.list.size -= 1
            return

        while True:
            if current_element == None:
                was_removed = False
                break
            # Check if it is the next element we are looking for
            next_element = current_element.next
            if next_element != None:
                if self == next_element:
                    if self == self.list.tail:
                        self.list.tail = current_element
                    # Found the right one, loop around the node
                    next_next_element = next_element.next
                    current_element.next = next_next_element
                    next_element = None
                    was_removed = True
                    break
            current_element = current_element.next
        if was_removed:
            self.list.size -= 1


