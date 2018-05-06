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
Provides a general data structure for container.
"""
from abc import abstractmethod
from py_alg_dat.string_visitor import StringVisitor

__author__ = "Brian Horn"
__copyright__ = "Copyright (c) 2015 Brian Horn"
__credits__ = "Brian Horn"
__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Brian Horn"
__email__ = "trycatchhorn@gmail.com"
__status__ = "Prototype"

class Container(object):

    """
    The interface of a container.
    """

    def __init__(self):
        """
        Constructs a container and initializes its
        count value to zero.
        """
        self.count = 0

    def __str__(self):
        """
        Returns a string representation of this container
        by using a visitor.

        @return: String representation of the container.
        @rtype: C{str}
        """
        visitor = StringVisitor()
        self.visit(visitor)
        return "%s {%s}" % (self.__class__.__name__, str(visitor))

    def __hash__(self):
        """
        Returns the hash value of this container.

        @return: Hash value of the container.
        @rtype: C{int}
        """
        result = hash(self.__class__)
        for obj in self:
            result = (result + hash(obj))
        return result

    @abstractmethod
    def __iter__(self):
        """
        Abstract method used to support the iterator
        protocol.
        """
        pass

    def get_count(self):
        """
        Returns the number of elements, represented by
        the count field, in this container.

        @return: Number of elements in the container.
        @rtype: C{int}
        """
        return self.count

    def is_empty(self):
        """
        Returns if the container has any elements.

        @return: True if the container is empty, false otherwise.
        @rtype: C{bool}
        """
        return self.count == 0

    def visit(self, visitor):
        """
        Makes the specified visitor visit all the elements
        in this container.

        @param visitor: The visitor applied to each element.
        @type: L{Visitor}
        """
        for obj in self:
            visitor.visit(obj)

    def element(self):
        """
        Generator that yields the objects in this container.
        """
        for obj in self:
            yield obj

