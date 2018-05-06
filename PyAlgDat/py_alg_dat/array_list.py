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
Provides a data structure for a dynamic list.
"""

__author__ = "Brian Horn"
__copyright__ = "Copyright (c) 2015 Brian Horn"
__credits__ = "Brian Horn"
__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Brian Horn"
__email__ = "trycatchhorn@gmail.com"
__status__ = "Prototype"

class ArrayList(object):

    """
    The interface of a dynamic list.
    """

    def __init__(self, length=0, base_index=0):
        """
        Constructs a dynamic list with the number of elements specified by
        the optional length parameter. If a length is specified, entries
        up to the specified length will be initialized to None.

        @param length: The number of elements contained in this array list.
        @type length: C{int}
        @param baseIndex: The index where the list starts.
        @type baseIndex: C{int}
        """
        self.data = [None for _ in xrange(length)]
        self.base_index = base_index

    def __repr__(self):
        """
        Returns the canonical representation of this array list.

        @return: Canonical string representation of the array list.
        @rtype: C{str}
        """
        return repr(self.data)

    def __str__(self):
        """
        Returns a string representation of this array list.

        @return: String representation of the array list.
        @rtype: C{str}
        """
        return self.__class__.__name__ + ": " + str(self.data)

    def __len__(self):
        """
        Returns the number of elements in this array list.

        @return: Number of elements in the array list.
        @rtype: C{int}
        """
        return len(self.data)

    def __eq__(self, other):
        """
        Compares two array lists for equality. The comparison
        is done by comparing the data field of the two
        array lists.

        @param other: The other array list.
        @type other: L{ArrayList}
        @return: True if the array lists are equal, false otherwise.
        @rtype: C{bool}
        """
        return self.data == other.data

    def __ne__(self, other):
        """
        Compares two array lists for inequality. The comparison
        is done by comparing the data field of the two
        array lists.

        @param other: The other array list.
        @type other: L{ArrayList}
        @return: True if the array lists are not equal, false otherwise.
        @rtype: C{bool}
        """
        return not self == other

    def __copy__(self):
        """
        Returns a shallow copy of this array list.

        @return: A copy of the array list.
        @rtype: L{ArrayList}
        """
        result = ArrayList(len(self.data))
        for i, datum in enumerate(self.data):
            result.data[i] = datum
        result.base_index = self.base_index
        return result

    def __getitem__(self, index):
        """
        Returns the item at the specified index.

        @param index: The index in the array list where the value is set.
        @type: C{int}
        @return: The item at the specified index.
        @rtype: C{object}
        """
        if isinstance(index, slice):
            return [self.data[i] for i in xrange(*index.indices(len(self.data)))]
        return self.data[self.get_offset(index)]

    def __setitem__(self, index, value):
        """
        Sets the item at the specified index in this
        array list equal to the specified value.

        @param index: The index in the array list where the value is set.
        @type: C{int}
        @param value: The value being set.
        @type: C{object}
        """
        self.data[self.get_offset(index)] = value

    def get_offset(self, index):
        """
        Returns the offset being the difference between the
        specified index and the base index of this array list.

        @param index: The index from where the offset is calculated.
        @type: C{int}
        @raises: IndexError if offset is out of bounds.
        @type: C{IndexError}
        @return: The offset of the array list.
        @rtype: C{int}
        """
        offset = index - self.base_index
        if offset < 0 or offset >= len(self.data):
            raise IndexError
        return offset

    def get_data(self):
        """
        Returns the data in this array list as a list.

        @return: The data in the array list.
        @rtype: C{list}
        """
        return self.data

    def get_base_index(self):
        """
        Returns the base index in this array list. The
        base index is the index where the array list starts.

        @return: The base index in the array list.
        @rtype: C{int}
        """
        return self.base_index

    def get_index(self, element):
        """
        Returns the first index of element in this array list.

        @param element: The element in the array list.
        @type: C{objec}
        @return: The first index of element in the array list.
        @rtype: C{int}
        """
        return self.data.index(element)

    def set_base_index(self, base_index):
        """
        Sets the base index in this array list. The
        base index is the index where the array list starts.

        @param baseIndex: The base index in the array list.
        @type: C{int}
        """
        self.base_index = base_index

    def set_length(self, value):
        """
        Sets the length of this array list to the specified
        value. If the specified value is different from the
        current length of the array list, existing elements,
        up to the size of value, are copied into a new array
        list.

        @param value: The new length of the array list.
        @type: C{int}
        """
        if len(self.data) != value:
            new_data = [None for i in xrange(value)]
            min_length = min(len(self.data), value)
            for i in xrange(min_length):
                new_data[i] = self.data[i]
            self.data = new_data

    def __delitem__(self, index):
        """
        Removes the item at the specified index from this array list.

        @param index: The index specifying the item being removed.
        @type: C{int}
        """
        if isinstance(index, int):
            return self.data.__delitem__(index)
        return
