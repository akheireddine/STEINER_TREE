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
Provides a data structure for a stack.
"""

__author__ = "Brian Horn"
__copyright__ = "Copyright (c) 2015 Brian Horn"
__credits__ = "Brian Horn"
__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Brian Horn"
__email__ = "trycatchhorn@gmail.com"
__status__ = "Prototype"

class Stack(object):

    """
    Implements a simple stack.
    """

    def __init__(self, start=None):
        """
        Constructs a stack. If the start parameter
        contains elements the stack is created containing
        these elements, otherwise the stack is initially
        empty.

        @param start: The elements in the stack after initialization.
        @type: C{list}
        """
        self.holder = []
        if start == None:
            start = []
        for elem in start:
            self.push(elem)
        self.holder.reverse()

    def __repr__(self):
        """
        Returns the canonical representation of this stack.

        @return: The canonical string representation of the stack.
        @rtype: C{string}
        """
        return '%s' % self.holder

    def __str__(self):
        """
        Returns a string representation of this stack.

        @return: The string representation of the stack.
        @rtype: C{string}
        """
        return str(self.__class__.__name__) + ": " + str(self.holder)

    def __eq__(self, other):
        """
        Compares two stacks for equality. The comparison
        is done by comparing the list field of the two stacks.

        @param other: The other stack.
        @type other: C{Stack}
        @return: True if the stacks are equal, False otherwise.
        @rtype: C{bool}
        """
        return self.holder == other.holder

    def __ne__(self, other):
        """
        Compares two stacks for inequality. The comparison
        is done by comparing the list field of the two stacks.

        @param other: The other stack.
        @type other: C{Stack}
        @return: True if the stacks are not equal, False otherwise.
        @rtype: C{bool}
        """
        return not self == other

    def __len__(self):
        """
        Returns the size of the stack.

        @return: The stack size.
        @rtype: C{int}
        """
        return len(self.holder)

    def push(self, item):
        """
        Pushs an element onto the stack.

        @param item: The item to be pushed onto the stack.
        @type: C{object}
        """
        self.holder = [item] + self.holder

    def pop(self):
        """
        Pops the top element of the stack. The element
        is removed from the stack.

        @return: The top element of the stack.
        @rtype: C{object}
        """
        try:
            elem = self.holder[0]
            self.holder = self.holder[1:]
        except:
            raise IndexError
        return elem

    def is_empty(self):
        """
        Checks whether or not the stack is empty.

        @return: True if the stack is empty, False otherwise.
        @rtype: C{bool}
            """
        return not self.holder

    def clear(self):
        """
        Removes all elements from the stack.
        """
        self.holder = []
