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
Provides a data structure used in graph algorithms.
"""

import sys

__author__ = "Brian Horn"
__copyright__ = "Copyright (c) 2015 Brian Horn"
__credits__ = "Brian Horn"
__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Brian Horn"
__email__ = "trycatchhorn@gmail.com"
__status__ = "Prototype"

class Entry(object):

    """
    The interface of an entry.
    """

    def __init__(self, *args):
        """
        Constructs an entry object.
        """
        if len(args) == 0:
            self.discovered = False
            self.distance = sys.maxint
            self.predecessor = None
            self.edge = None
        elif len(args) == 4:
            self.discovered = args[0]
            self.distance = args[1]
            self.predecessor = args[2]
            self.edge = args[3]
        else:
            raise ValueError

    def __repr__(self):
        """
        Returns the canonical representation of this entry.

        @return: Canonical string representation of the entry.
        @rtype: C{str}
        """
        return repr((self.discovered, self.distance, self.predecessor, self.edge))

    def __str__(self):
        """
        Returns a string representation of this entry.

        @return: String representation of the entry.
        @rtype: C{str}
        """
        class_name = str(self.__class__.__name__) + ": ("
        discover_str = str(self.discovered) + ", "
        predecessor_str = str(self.predecessor) + ", "
        edge_str = str(self.edge) + " )"
        entry_str = class_name + discover_str + predecessor_str + edge_str
        return entry_str

    def __hash__(self):
        """
        Returns the hash of this entry.

        @return: The hash of the entry.
        @rtype: C{int}
        """
        key = hash(self.discovered) + hash(self.distance) + hash(self.predecessor) + hash(self.edge)
        return key

    def __eq__(self, other):
        """
        Compares two entries for equality. The comparison
        is done by comparing each of the fields in the two
        entries against each other.

        @param other: The other entry.
        @type other: L{Entry}
        @return: True if the entries are equal, false otherwise.
        @rtype: C{bool}
        """
        if self.discovered != other.discovered:
            return False
        elif self.distance != other.distance:
            return False
        elif self.predecessor != other.predecessor:
            return False
        elif self.edge != other.edge:
            return False
        return True

    def __ne__(self, other):
        """
        Compares two entries for inequality. The comparison
        is done by comparing each of the fields in the two
        entries against each other.

        @param other: The other entry.
        @type other: L{Entry}
        @return: True if the entries are not equal, false otherwise.
        @rtype: C{bool}
        """
        return not self == other

    def __lt__(self, other):
        """
        Implements the 'less than' operator for this entry.

        @param other: The other entry.
        @type other: L{Entry}
        @return: True if this entry is 'less than' the other, false otherwise.
        @rtype: C{bool}
        """
        if self.discovered >= other.discovered:
            return False
        elif self.distance >= other.distance:
            return False
        elif self.predecessor >= other.predecessor:
            return False
        elif self.edge >= other.edge:
            return False
        return True

    def __le__(self, other):
        """
        Implements the 'less than -or equal' operator for this entry.

        @param other: The other entry.
        @type other: L{Entry}
        @return: True if this entry is 'less than -or equal' to the other, false otherwise.
        @rtype: C{bool}
        """
        if self.discovered > other.discovered:
            return False
        elif self.distance > other.distance:
            return False
        elif self.predecessor > other.predecessor:
            return False
        elif self.edge > other.edge:
            return False
        return True

    def __gt__(self, other):
        """
        Implements the 'greater than' operator for this entry.

        @param other: The other entry.
        @type other: L{Entry}
        @return: True if this entry is 'greater than' the other, false otherwise.
        @rtype: C{bool}
        """
        if self.discovered <= other.discovered:
            return False
        elif self.distance <= other.distance:
            return False
        elif self.predecessor <= other.predecessor:
            return False
        elif self.edge <= other.edge:
            return False
        return True

    def __ge__(self, other):
        """
        Implements the 'greater than -or equal' operator for this entry.

        @param other: The other entry.
        @type other: L{Entry}
        @return: True if this entry is 'greater than -or equal' to the other, false otherwise.
        @rtype: C{bool}
        """
        if self.discovered < other.discovered:
            return False
        elif self.distance < other.distance:
            return False
        elif self.predecessor < other.predecessor:
            return False
        elif self.edge < other.edge:
            return False
        return True

    def get_discovered(self):
        """
        Returns if this entry has been discovered.

        @return: If the entry has been discovered.
        @rtype: Cbool}
        """
        return self.discovered

    def get_distance(self):
        """
        Returns the distance of this entry.

        @return: The distance of the entry.
        @rtype: C{int}
        """
        return self.distance

    def get_predecessor(self):
        """
        Returns the predecessor of this entry.

        @return: The predecessor of the entry.
        @rtype: L{UnWeightedGraphVertex}
        """
        return self.predecessor

    def get_edge(self):
        """
        Returns the edge of this entry.

        @return: The edge of the entry.
        @rtype: L{GraphEdge}
        """
        return self.edge

