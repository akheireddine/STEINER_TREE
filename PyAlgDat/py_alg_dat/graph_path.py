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
Provides a data structure used to hold the vertices in a path. A path, in a graph,
is given by the vertices being visited when going along a sequence of vertices
starting from a source vertex and ending at a destination vertex. A valid path
connects a sequence of vertices, using edges, without discontinuities.
"""

__author__ = "Brian Horn"
__copyright__ = "Copyright (c) 2015 Brian Horn"
__credits__ = "Brian Horn"
__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Brian Horn"
__email__ = "trycatchhorn@gmail.com"
__status__ = "Prototype"

class GraphPath(object):

    """
    Implements a path in a graph.
    """

    def __init__(self, graph):
        """
        Constructs a graph path.

        @param graph: The graph of which contains this path.
        @type: C{object}
        """
        self.graph = graph
        self.vertices = []
        self.edges = []
        self.path_length = 0

    def __str__(self):
        """
        Returns a string representation of this graph path.

        @return: String representation of the graph path.
        @rtype: C{str}
        """
        str_rep = str(self.__class__.__name__) + ":\n"
        str_rep += "\tVertices: " + str(self.get_vertices()) + "\n"
        str_rep += "\tEdges: " + str(self.get_edges()) + "\n"
        str_rep += "\tPath length: " + str(self.path_length) + "\n"
        return str_rep

    def __eq__(self, other):
        """
        Compares two graph paths for equality. The comparison
        is done by comparing the vertices, edges, and path_lehgth
        fields of the two graph paths.

        @param other: The other graph path.
        @type other: L{GraphPath}
        @return: True if the graph paths are equal, false otherwise.
        @rtype: C{bool}
        """
        if isinstance(other, GraphPath):
            if self.vertices != other.vertices:
                return False
            elif self.edges != other.edges:
                return False
            elif self.path_length != other.path_length:
                return False
            return True
        return NotImplemented

    def __ne__(self, other):
        """
        Compares two graph paths for inequality. The comparison
        is done by comparing the vertices, edges, and path_lehgth
        fields of the two graph paths.

        @param other: The other graph path.
        @type other: L{GraphPath}
        @return: True if the graph paths are not equal, false otherwise.
        @rtype: C{bool}
        """
        return not self == other

    def get_vertices(self):
        """
        Returns the vertices contained in this graph path.

        @return: The vertices in the graph path.
        @rtype: C{list}
        """
        return self.vertices

    def get_edges(self):
        """
        Returns the edges contained in this graph path.

        @return: The edges in the graph path.
        @rtype: C{list}
        """
        return self.edges

    def get_path_length(self):
        """
        Returns the length of this graph path.

        @return: The length of the graph path.
        @rtype: C{int}
        """
        return self.path_length

    def get_number_of_edges(self):
        """
        Returns the number of edges contained in this graph path.

        @return: The number of edges in the graph path.
        @rtype: C{int}
        """
        return len(self.edges)

    def get_number_of_vertices(self):
        """
        Returns the number of vertices contained in this graph path.

        @return: The number of vertices in the graph path.
        @rtype: C{int}
        """
        return len(self.vertices)

    def is_empty(self):
        """
        Returns if this graph path is empty. The graph path is
        considered empty if it does not container any vertices
        and any edges.

        @return: True, if the graph path is empty, false otherwise.
        @rtype: C{bool}
        """
        return len(self.vertices) == 0 and len(self.edges) == 0

    def add_vertex(self, vertex):
        """
        Adds the specified vertex to this graph path.

        @param vertex: The vertex to be added to the graph path.
        @type: L{GraphVertex}
        """
        if self.graph.has_vertex(vertex):
            self.vertices.insert(0, vertex)

    def add_edge(self, edge):
        """
        Adds the specified edge to this graph path.

        @param edge: The edge to be added to the graph path.
        @type: L{GraphEdge}
        """
        if self.graph.has_edge(edge):
            self.edges.insert(0, edge)
            self.path_length += edge.get_weight()





