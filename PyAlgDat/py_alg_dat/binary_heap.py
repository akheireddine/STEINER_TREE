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
Provides a general data structure used to model a binary heap.
"""

__author__ = "Brian Horn"
__copyright__ = "Copyright (c) 2015 Brian Horn"
__credits__ = "Brian Horn"
__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Brian Horn"
__email__ = "trycatchhorn@gmail.com"
__status__ = "Prototype"

class BinaryHeap(object):

    """
    Implements a binary-heap acting as a base class for the
    more specific max-heap and min-heap classes.
    """

    def __init__(self, start=None):
        """
        Constructs a binary-heap from the specified array or
        creates an empty binary-heap.
        """
        self.array = start or []

    def __repr__(self):
        """
        Returns the canonical representation of this binary-heap.

        @return: The canonical string representation of the binary-heap.
        @rtype: C{string}
        """
        return '%s' % self.array

    def __str__(self):
        """
        Returns a string representation of this binary-heap.

        @return: The string representation of the binary-heap.
        @rtype: C{string}
        """
        start = 0
        length = 1
        str_value = ""
        while start < len(self.array):
            str_value += "\n"
            str_value += str(self.array[start:start + length])
            start += length
            length *= 2
        return str_value

    def __eq__(self, other):
        """
        Compares two binary-heaps for equality. The comparison
        is done by comparing the list field of the two binary-heaps.

        @param other: The other binary-heap.
        @type other: C{BinaryHeap}
        @return: True if the binary-heaps are equal, false otherwise.
        @rtype: C{bool}
        """
        return self.array == other.array

    def __ne__(self, other):
        """
        Compares two binary-heaps for inequality. The comparison
        is done by comparing the list field of the two binary-heaps.

        @param other: The other binary-heap.
        @type other: C{BinaryHeap}
        @return: True if the binary-heaps are not equal, false otherwise.
        @rtype: C{bool}
        """
        return not self == other

    def __len__(self):
        """
        Returns the size of the binary-heap.

        @return: The binary-heap size.
        @rtype: C{int}
        """
        return len(self.array)

    def is_empty(self):
        """
        Checks whether or not the binary-heap is empty.

        @return: True if the binary-heap is empty, false otherwise.
        @rtype: C{bool}
        """
        return not self.array

    def clear(self):
        """
        Removes all elements from the binary-heap.
        """
        self.array = []

    @staticmethod
    def parent(i):
        """
        Parent will be at math.floor((i - 1) / 2). Since integer division,
        in Python, simulates the floor function, we do not explicity use it
        because the answer will already be rounded down.

        @param i: The index of the node from which the parent index should be calculated.
        @type: C{int}
        @return: The parent index of the node i.
        @rtype: C{int}
        """
        return (i - 1) / 2

    @staticmethod
    def left_child(i):
        """
        Given index i, compute the index of i's left child.

        @param i: The index of the node from which the left child index should be calculated.
        @type: C{int}
        @return: The left child index of the node i.
        @rtype: C{int}
        """
        return 2 * i + 1

    @staticmethod
    def right_child(i):
        """
        Given index i, compute the index of i's right child.

        @param i: The index of the node from which the right child index should be calculated.
        @type: C{int}
        @return: The right index of the node i.
        @rtype: C{int}
        """
        return 2 * i + 2

    def is_leaf(self, i):
        """
        Given index i, determine if the index key at index i is a leaf.

        @param i: The index of the key which should be checked if it is a leaf node.
        @type: C{int}
        @return: True if the key at index i is a leaf, false otherwise.
        @rtype: C{bool}
        """
        return i < len(self.array) and i >= len(self.array) / 2

    def swap(self, index_a, index_b):
        """
        Swaps the heap elements at the specified indices.

        @param index_a, The index of the first element.
        @type: C{int}
        @param index_b, The index of the second element.
        @type: C{int}
        """
        if index_a < 0 and index_a > len(self.array) - 1:
            return
        elif index_b < 0 and index_b > len(self.array) - 1:
            return
        else:
            self.array[index_a], self.array[index_b] = self.array[index_b], self.array[index_a]

