#!/usr/bin/env python

#
# The MIT License (MIT)
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
Provides a data structure for a queue.
"""

__author__ = "Brian Horn"
__copyright__ = "Copyright (c) 2015 Brian Horn"
__credits__ = "Brian Horn"
__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Brian Horn"
__email__ = "trycatchhorn@gmail.com"
__status__ = "Prototype"

class Queue(object):

    """
    Implements a simple queue.
    """

    def __init__(self, start=None):
        """
        Constructs a queue. If the start parameter
        contains elements the queue is created containing
        these elements, otherwise the queue is initially
        empty.

        @param start: The elements in the queue after initialization.
        @type: C{list}
        """
        self.holder = []
        if start == None:
            start = []
        for elem in start:
            self.enqueue(elem)

    def __rep__(self):
        """
        Returns the canonical representation of this queue.

        @return: The canonical string representation of the queue.
        @rtype: C{string}
        """
        return "%s" % self.holder

    def __str__(self):
        """
        Returns the string representation of this queue.

        @return: The string representation of the queue.
        @rtype: C{string}
        """
        return str(self.__class__.__name__) + ": " + str(self.holder)

    def __eq__(self, other):
        """
        Compares two queues for equality. The comparison
        is done by comparing the list field of the two queues.

        @param other: The other queue.
        @type other: C{Queue}
        @return: True if the queues are equal, False otherwise.
        @rtype: C{bool}
        """
        return self.holder == other.holder

    def __ne__(self, other):
        """
        Compares two queues for inequality. The comparison
        is done by comparing the list field of the two queues.

        @param other: The other queue.
        @type other: C{Queue}
        @return: True if the queues are not equal, False otherwise.
        @rtype: C{bool}
        """
        return not self == other

    def __len__(self):
        """
        Returns the size of the queue.

        @return: The queue size.
        @rtype: C{int}
        """
        return len(self.holder)

    def enqueue(self, value):
        """
        Addes a value to the end of the queue.

        @param value: The value to be added to the queue.
        @type: C{object}
        """
        self.holder.append(value)

    def dequeue(self):
        """
        Removes and returns the front element in this queue.

        @return: The front element in the queue.
        @rtype: C{object}
        """
        value = None
        try:
            value = self.holder[0]
            if len(self.holder) == 1:
                self.holder = []
            else:
                self.holder = self.holder[1:]
        except:
            raise IndexError
        return value

    def is_empty(self):
        """
        Checks whether or not the queue is empty.

        @return: True if the queue is empty, otherwise False.
        @rtype: C{bool}
        """
        return len(self.holder) == 0

    def clear(self):
        """
        Removes all elements from the queue.
        """
        self.holder = []
