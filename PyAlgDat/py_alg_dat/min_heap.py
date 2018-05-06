#!/usr/bin/env python

# The MIT License (MIT)
#
# Copyright (C) 2015 by Brian Horn, trycatchhorn@gmail.com.
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
Provides a general data structure used to model a minimum heap.
"""

__author__ = "Brian Horn"
__copyright__ = "Copyright (c) 2015 Brian Horn"
__credits__ = "Brian Horn"
__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Brian Horn"
__email__ = "trycatchhorn@gmail.com"
__status__ = "Prototype"

from py_alg_dat.binary_heap import BinaryHeap

class MinHeap(BinaryHeap):

    """
    Implements a min-heap with the property that the data item stored
    in each node is less than or equal to the data items stored
    in its children.
    """

    def __init__(self, start=None):
        """
        Constructs a min-heap from the specified array or
        creates an empty min-heap.
        """
        super(MinHeap, self).__init__(start)
        if start == None:
            start = []
        self.build_min_heap(self.array)

    def __repr__(self):
        """
        Returns the canonical representation of this min-heap.

        @return: The canonical string representation of the min-heap.
        @rtype: C{string}
        """
        return '%s' % self.array

    def __str__(self):
        """
        Returns a string representation of this min-heap.

        @return: The string representation of the min-heap.
        @rtype: C{string}
        """
        start = 0
        length = 1
        str_rep = ""
        while start < len(self.array):
            str_rep += "\n"
            str_rep += str(self.array[start:start + length])
            start += length
            length *= 2
        return str_rep

    def __eq__(self, other):
        """
        Compares two min-heaps for equality. The comparison
        is done by comparing the list field of the two min-heaps.

        @param other: The other min-heap.
        @type other: C{MinHeap}
        @return: True if the min-heaps are equal, false otherwise.
        @rtype: C{bool}
        """
        return self.array == other.array

    def __ne__(self, other):
        """
        Compares two min-heaps for inequality. The comparison
        is done by comparing the list field of the two min-heaps.

        @param other: The other min-heap.
        @type other: C{MinHeap}
        @return: True if the min-heaps are not equal, false otherwise.
        @rtype: C{bool}
        """
        return not self == other

    def is_min_heap(self):
        """
        Determines if this heap is a min-heap.

        @return: True if this heap is a min-heap, false otherwise.
        @rtype: C{bool}
        """
        number_of_elements = len(self.array)
        for i in xrange(number_of_elements):
            left_index = self.left_child(i)
            right_index = self.right_child(i)
            if left_index < number_of_elements and self.array[left_index] < self.array[i]:
                return False
            if right_index < number_of_elements and self.array[right_index] < self.array[i]:
                return False
        return True

    def insert(self, key):
        """
        Inserts an element into the min-heap.

        Time complexity: O(log(n)).

        @param key: The key of the node to be inserted into the min-heap.
        @type: C{int}
        """
        self.array.append(key)
        self.propagate_up(len(self.array) - 1)

    def remove(self, i):
        """
        Removes the element at the specified position from the
        min-heap.

        Time complexity: O(log(n)).

        @param i: The index of the node to be removed from the heap.
        @type: C{int}
        """
        if i < 0 or i > len(self.array) - 1:
            return None
        if i == len(self.array) - 1:
            pass
        else:
            self.swap(i, len(self.array) - 1)
            while i > 0 and self.array[self.parent(i)] > self.array[i]:
                self.swap(self.array(i), self.parent(i))
                i = self.parent(i)
            if len(self.array) != 0:
                self.propagate_down(i)
        self.array.pop()

    def propagate_up(self, i):
        """
        Compares node at index i with parent node and swaps it with
        the parent node if the node is smaller that the parent node.

        Time complexity: O(log(n)).

        @param i: The index of the node from which propergation starts.
        @type: C{int}
        """
        if i <= 0 and i > len(self.array) - 1:
            return
        while i != 0 and self.array[i] < self.array[self.parent(i)]:
            self.swap(i, self.parent(i))
            i = self.parent(i)

    def propagate_down(self, i):
        """
        Compares node at index i with the left node and moves the node
        at index i down the heap by successively exchanging the node
        with the smaller of its two children. The operation continues
        until the node reaches a position where it is less than both
        its children, or, failing that, until it reaches a leaf.

        Time complexity: O(log(n)).

        @param i: The index of the node from which propergation starts.
        @type: C{int}
        """
        if i <= 0 and i > len(self.array) - 1:
            return
        while not self.is_leaf(i):
            left_index = self.left_child(i)
            if left_index < len(self.array) - 1:
                if self.array[left_index + 1] > self.array[left_index]:
                    left_index += 1
            if self.array[left_index] >= self.array[i]:
                return
            self.swap(i, left_index)
            i = left_index

    def min_heapify_recursive(self, holder, i):
        """
        Responsible for maintaining the min-heap property of the min-heap.
        This function assumes that the subtree located at the left
        and right child satisfies the min-heap property. But the
        tree at index (current node) does not. Note: this function
        uses recursion, which for large inputs could cause a
        decrease in runtime performance compared to its iterative
        counterpart.

        Time complexity: O(log(n)).

        @param holder: The array backing the min-heap.
        @type: C{list}
        @param i: The index of the node from which the min-heap property should be maintained.
        @type: C{int}
        """
        left_index = self.left_child(i)
        right_index = self.right_child(i)
        smallest = i
        if left_index < len(holder) and holder[left_index] < holder[i]:
            smallest = left_index
        if right_index < len(holder) and holder[right_index] < holder[smallest]:
            smallest = right_index
        if smallest != i:
            holder[i], holder[smallest] = holder[smallest], holder[i]
            self.min_heapify_recursive(holder, smallest)

    def min_heapify_iterative(self, holder, i):
        """
        Responsible for maintaining the min-heap property of the min-heap.
        This function assumes that the subtree located at the left
        and right child satisfies the min-heap property. But the
        tree at index (current node) does not. Note: this function
        uses iteration, which for large inputs could give an increase
        in runtime performance compared to its recursive counterpart.

        Time complexity: O(log(n)).

        @param holder: The array backing the min-heap.
        @type: C{list}
        @param i: The index of the node from which the min-heap property should be maintained.
        @type: C{int}
        """
        left_index = self.left_child(i)
        while left_index < len(holder):
            right_index = left_index + 1
            if right_index == len(holder):
                if holder[left_index] < holder[i]:
                    holder[left_index], holder[i] = holder[i], holder[left_index]
                return
            choice = right_index
            if holder[left_index] < holder[right_index]:
                choice = left_index
            if holder[choice] > holder[i]:
                return
            holder[i], holder[choice] = holder[choice], holder[i]
            i = choice
            left_index = 2 * i + 1

    def build_min_heap(self, holder, recursive=True):
        """
        Responsible for building the min-heap bottom up. It starts with the lowest
        non-leaf nodes and calls heapify on them. This function is useful for
        initialising a heap with an unordered array.

        Time complexity: O(n).

        @param holder: The array backing the min-heap.
        @type: C{list}
        @param recursive: If True, the min-heap is build using recursion otherwise iteration.
        @type: C{int}
        """
        for i in xrange(len(holder) / 2, - 1, -1):
            if recursive:
                self.min_heapify_recursive(holder, i)
            else:
                self.min_heapify_iterative(holder, i)

    def heap_sort(self):
        """
        The heap-sort algorithm.

        Time complexity: O(n*log(n)).
        """
        self.build_min_heap(self.array)
        output = []
        for i in xrange(len(self.array) -1, 0, -1):
            self.array[0], self.array[i] = self.array[i], self.array[0]
            output.append(self.array.pop())
            self.min_heapify_recursive(self.array, 0)
        output.append(self.array.pop())
        self.array = output

    def heap_extract_min(self):
        """
        Part of the Priority Queue, extracts the element on the top of
        the heap and then re-heapifies. Note: this function removes -and
        returns the element on top of the heap.

        Time complexity: O(log(n)).

        @return: The min element of this heap.
        @rtype: C{int}
        """
        min_value = self.array[0]
        data = self.array.pop()
        if len(self.array) > 0:
            self.array[0] = data
            self.min_heapify_recursive(self.array, 0)
        return min_value

    def heap_increase_key(self, i, key):
        """
        Implements the increase key operation.

        Time complexity: O(log(n)).

        @param i: The index of the node whose key is to increased.
        @type: C{int}
        @param key: The new key of the node to be increase in the heap.
        @type: C{int}
        """
        if key < self.array[i]:
            print "New key is smaller than current key!"
            return
        self.array[i] = key
        while i > 0 and self.array[self.parent(i)] > self.array[i]:
            self.swap(i, self.parent(i))
            i = self.parent(i)

    def increment(self, key, value):
        """
        Increments key by the input value.

        Time complexity: O(n).

        @param key: The key which should be increased.
        @type: C{int}
        @param value: The value by which the key should be increased.
        @type: C{int}
        """
        for i in xrange(len(self.array)):
            if self.array[i] == key:
                self.array[i] += value
                self.propagate_up(i)
                break

    def heap_merge(self, heap):
        """
        Implements the merge heap operation.

        Time complexity: O(n).

        @param i: The heap which should be merged with this heap.
        @type: L{heap}
        @return: A heap which is the result of a merge between this heap and h.
        @type: L{heap}
        """
        nodes = self.array + heap.array
        result = MinHeap(nodes)
        return result

