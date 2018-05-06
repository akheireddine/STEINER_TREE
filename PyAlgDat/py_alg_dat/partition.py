#!/usr/bin/env python

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
Data structure implementing a partition as a disjoint set
using Union-Find with rank and path compression.
"""

__author__ = "Brian Horn"
__copyright__ = "Copyright (c) 2015 Brian Horn"
__credits__ = "Brian Horn"
__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Brian Horn"
__email__ = "trycatchhorn@gmail.com"
__status__ = "Prototype"

class Partition(object):

    """
    The interface for a partition implemented as a
    disjoint set data structure.
    """

    def __init__(self, start=None):
        """
        Constructs a partition backed by a
        dictionary. If no start parameter is
        given, an empty partition is created.
        Otherwise, singleton sets is created
        for each of the elements in the start
        parameter array, each singleton set
        is inserted into the dictionary.

        @param start: The elements in the partition after initialization.
        @type: C{list}
        """
        self.elems = {}
        if start == None:
            start = []
        for elem in start:
            self.make_set(elem)

    def same_set(self, elem1, elem2):
        """
        Returns if the two specified elements are
        in the same set.

        @param elem1: The key of the first element in the partition.
        @type elem1: C{str}
        @param elem2: The key of the second element in the partition.
        @type elem2: C{str}
        @return: True if the elements are in the same set, false otherwise.
        @rtype: C{bool}
        """
        return self.find_element_iterative(elem1) == self.find_element_iterative(elem2)

    def make_set(self, elem):
        """
        Creates -and inserts the specified element into
        this partition.

        @param elem: The key of the element to be inserted in the partition.
        @type elem: C{str}
        """
        if elem is None:
            # Raise exception instead of return
            return False
        self.elems[elem] = PartitionElement(elem)
        return True

    def __repr__(self):
        """
        Returns the canonical representation of this partition.

        @return: Canonical string representation of the partition.
        @rtype: C{str}
        """
        return repr(self.elems)

    def __eq__(self, other):
        """
        Compares two partitions for equality. The comparison
        is done by comparing the dictionaries backing the two
        partitions.

        @param other: The other partition.
        @type other: L{Partition}
        @return: True if the partitions are equal, false otherwise.
        @rtype: C{bool}
        """
        if isinstance(other, Partition):
            return self.elems == other.elems
        return NotImplemented

    def __ne__(self, other):
        """
        Compares two partitions for inequality. The comparison
        is done by comparing the dictionaries backing the two
        partitions.

        @param other: The other partition.
        @type other: L{Partition}
        @return: True if the partitions are not equal, false otherwise.
        @rtype: C{bool}
        """
        return not self == other

    def find(self, elem):
        """
        Returns the element with the specified key
        in this partition, if the element is present.
        If the element is not contained in the partition
        a KeyError exception is raised.

        @param elem: The key to search for in the partition.
        @type elem: C{str}
        @return: The element with the specified key.
        @rtype: C{str}
        @raises: KeyError if the element is not in the partition.
        @type: C{KeyError}
        """
        if elem not in self.elems:
            raise KeyError
        return self.find_element_recursive(elem)

    def find_element_recursive(self, elem):
        """
        Returns the element with the specified key
        in this partition. The search is carried
        out by recursively comparing the specified
        element with its parent element.

        @param elem: The key to search for in the partition.
        @type elem: C{str}
        @return: The element with the specified key.
        @rtype: C{str}
        """
        if elem not in self.elems:
            return elem
        key_elem = self.elems[elem]
        if elem == key_elem.parent:
            return elem
        key_elem.parent = self.find_element_recursive(key_elem.parent)
        return key_elem.parent

    def find_element_iterative(self, elem):
        """
        Returns the element with the specified key
        in this partition. The search is carried
        out by iteratively comparing the specified
        element with its parent element.

        @param elem: The key to search for in the partition.
        @type elem: C{str}
        @return: The element with the specified key.
        @rtype: C{str}
        """
        root = elem
        while root != self.elems[elem]:
            root = self.elems[root]
        while elem != root:
            newp = self.elems[elem]
            self.elems[elem] = root
            elem = newp
        return self.elems[root.parent].parent

    def union(self, elem1, elem2):
        """
        Merges the two specified elements so they belong
        to the same set. If the two elements already are
        in the same set nothing is done.

        @param elem1: The key of the first element in the partition.
        @type elem1: C{str}
        @param elem2: The key of the second element in the partition.
        @type elem2: C{str}
        """
        one_link = self.elems.get(self.find(elem1))
        two_link = self.elems.get(self.find(elem2))

        if one_link == two_link or one_link == None or two_link == None:
            return None
        if one_link.rank > two_link.rank:
            two_link.parent = one_link.parent
        elif one_link.rank < two_link.rank:
            one_link.parent = two_link.parent
        else:
            two_link.parent = one_link.parent
            one_link.rank += 1

    def union_recursive(self, elem_x, elem_y):
        """
        Merges the two specified elements so they belong
        to the same set. If the two elements already are
        in the same set nothing is done.

        NOTE: This version uses a recursive version of
        the find method.

        @param elem1: The key of the first element in the partition.
        @type elem1: C{str}
        @param elem2: The key of the second element in the partition.
        @type elem2: C{str}
        """
        one_link = self.elems.get(self.find_element_recursive(elem_x))
        two_link = self.elems.get(self.find_element_recursive(elem_y))

        if one_link == two_link or one_link == None or two_link == None:
            return None
        if one_link.rank > two_link.rank:
            two_link.parent = one_link.parent
        elif one_link.rank < two_link.rank:
            one_link.parent = two_link.parent
        else:
            two_link.parent = one_link.parent
            one_link.rank += 1

    def union_iterative(self, elem_x, elem_y):
        """
        Merges the two specified elements so they belong
        to the same set. If the two elements already are
        in the same set nothing is done.

        NOTE: This version uses an iterative version of
        the find method.

        @param elem1: The key of the first element in the partition.
        @type elem1: C{str}
        @param elem2: The key of the second element in the partition.
        @type elem2: C{str}
        """
        one_link = self.elems.get(self.find_element_iterative(elem_x))
        two_link = self.elems.get(self.find_element_iterative(elem_y))

        if one_link == two_link or one_link == None or two_link == None:
            return None
        if one_link.rank > two_link.rank:
            two_link.parent = one_link.parent
        elif one_link.rank < two_link.rank:
            one_link.parent = two_link.parent
        else:
            two_link.parent = one_link.parent
            one_link.rank += 1

class PartitionElement(object):

    """
    Class representing an element in a partition.
    """

    def __init__(self, parent):
        """
        Constructs an element in the partition.
        Initially, the element has itself as
        parent and its rank is zero.
        """
        self.parent = parent
        self.rank = 0

    def __repr__(self):
        """
        Returns a canonical representation of this partition element.

        @return: Canonocal representation of the partition element.
        @rtype: C{str}
        """
        return repr((self.parent, self.rank))

    def __eq__(self, other):
        """
        Compares two partition elements for equality. The comparison
        is done by comparing the parent fields and the rank fields
        of the two partition elements.

        @param other: The other partition element.
        @type other: L{PartitionElement}
        @return: True if the elements are equal, false otherwise.
        @rtype: C{bool}
        """
        if isinstance(other, PartitionElement):
            return self.parent == other.parent and self.rank == other.rank
        return NotImplemented

    def __ne__(self, other):
        """
        Compares two partition elements for inequality. The comparison
        is done by comparing the parent fields and the rank fields
        of the two partition elements.

        @param other: The other partition element.
        @type other: L{PartitionElement}
        @return: True if the elements are not equal, false otherwise.
        @rtype: C{bool}
        """
        return not self == other

    def __lt__(self, other):
        """
        Implements the 'less than' operator for this partition
        element.

        @param other: The other partition element.
        @type other: L{PartitionElement}
        @return: True if this element is 'less than' the other, false otherwise.
        @rtype: C{bool}
        """
        return self.parent < other.parent and self.rank < other.rank

    def __le__(self, other):
        """
        Implements the 'less than -or equal' operator for this partition
        element.

        @param other: The other partition element.
        @type other: L{PartitionElement}
        @return: True if this element is 'less than -or equal' to the other, false otherwise.
        @rtype: C{bool}
        """
        return self.parent <= other.parent and self.rank <= other.rank

    def __gt__(self, other):
        """
        Implements the 'greater than' operator for this partition
        element.

        @param other: The other partition element.
        @type other: L{PartitionElement}
        @return: True if this element is 'greater than' the other, false otherwise.
        @rtype: C{bool}
        """
        return self.parent > other.parent and self.rank > other.rank

    def __ge__(self, other):
        """
        Implements the 'greater than -or equal' operator for this partition
        element.

        @param other: The other partition element.
        @type other: L{PartitionElement}
        @return: True if this element is 'greater than -or equal' to the other, false otherwise.
        @rtype: C{bool}
        """
        return self.parent >= other.parent and self.rank >= other.rank

    def __hash__(self):
        """
        Returns the hash of this partition element.

        @return: The hash of the partition element.
        @rtype: C{int}
        """
        return hash((self.parent, self.rank))
