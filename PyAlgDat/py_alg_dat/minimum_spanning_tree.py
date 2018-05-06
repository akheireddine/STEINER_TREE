#!/usr/bin/env python

# The MIT Licese (MIT)
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
Provides a data structure used to hold the edges in a minimum spanning tree.
"""

__author__ = "Brian Horn"
__copyright__ = "Copyright (c) 2015 Brian Horn"
__credits__ = "Brian Horn"
__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Brian Horn"
__email__ = "trycatchhorn@gmail.com"
__status__ = "Prototype"

class MinimumSpanningTree(object):

    """
    The interface of a minimum spanning tree.
    """

    def __init__(self, graph):
        """
        Constructs a minimum spanning tree.
        Initially, the minimum spanning tree is empty.
        """
        self.graph = graph
        self.edges = []
        self.totalweight = 0

    def __str__(self):
        """
        Returns a string representation of this minimum spanning tree.

        @return: String representation of the minimum spanning tree.
        @rtype: C{str}
        """

        class_name_str = str(self.__class__.__name__) + ": ("
        attributes_str = "Edges: " + str(self.edges) + ",  Weight: " + str(self.totalweight) + ")"
        str_rep = class_name_str + attributes_str
        return str_rep

    def __eq__(self, other):
        """
        Compares two minimum spanning trees for equality. The comparison
        is done by comparing the edges -and the total weight fields in
        the two minimum spanning trees against each other.

        @param other: The other minimum spanning tree.
        @type other: L{MinimumSpanningTree}
        @return: True if the minimum spanning trees are equal, false otherwise.
        @rtype: C{bool}
        """
        if isinstance(other, MinimumSpanningTree):
            return self.edges == other.edges and self.totalweight == other.totalweight
        return NotImplemented

    def __ne__(self, other):
        """
        Compares two minimum spanning trees for inequality. The comparison
        is done by comparing the edges -and the total weight fields in
        the two minimum spanning trees against each other.

        @param other: The other minimum spanning tree.
        @type other: L{MinimumSpanningTree}
        @return: True if the minimum spanning trees are not equal, false otherwise.
        @rtype: C{bool}
        """
        return not self == other

    def get_edges(self):
        """
        Returns the edges contained in this minimum spanning tree.

        @return: The edges contained in the minimum spanning tree.
        @rtype: C{list}
        """
        return self.edges

    def get_total_weight(self):
        """
        Returns the total weight of this minimum spanning tree.

        @return: The total weight of the minimum spanning tree.
        @rtype: C{int}
        """
        return self.totalweight

    def add_edge(self, edge):
        """
        Adds the specified edge to this minimum spanning tree.

        @param edge: The edge to be added to the minimum spanning tree.
        @type: L{MinimumSpanningTree}
        """
        self.edges.append(edge)
        self.totalweight += edge.get_weight()

