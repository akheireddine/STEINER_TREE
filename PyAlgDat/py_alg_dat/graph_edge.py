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
Provides a data structure for an edge in a graph.
"""

__author__ = "Brian Horn"
__copyright__ = "Copyright (c) 2015 Brian Horn"
__credits__ = "Brian Horn"
__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Brian Horn"
__email__ = "trycatchhorn@gmail.com"
__status__ = "Prototype"

class GraphEdge(object):

    """
    The interface of a graph edge data structure.
    """

    def __init__(self, graph, head_vertex, tail_vertex):
        """
        Constructs a graph edge connecting the two specified
        vertices found in the specified graph.

        @param graph: The graph including the edge.
        @type: L{Graph}
        @param v0: The first vertex of the edge.
        @type: L{GraphVertex}
        @param v1: The second vertex of the edge.
        @type: L{GraphVertex}
        """
        self.graph = graph
        self.head_vertex = head_vertex
        self.tail_vertex = tail_vertex

    def __repr__(self):
        """
        Returns the canonical representation of this graph edge.

        @return: Canonical string representation of the graph edge.
        @rtype: C{str}
        """
        return repr((self.head_vertex, self.tail_vertex))

    def __str__(self):
        """
        Returns a string representation of this graph edge.

        @return: String representation of the graph edge.
        @rtype: C{str}
        """
        class_name_str = str(self.__class__.__name__) + ": ("
        attributes_str = str(self.head_vertex) + ", " + str(self.tail_vertex) + ")"
        str_rep = class_name_str + attributes_str
        return str_rep

    def __eq__(self, other):
        """
        Compares two graph edges for equality. The comparison
        is done by comparing the vertices constituting the
        respective edge.

        @param other: The other graph edge.
        @type other: L{GraphEdge}
        @return: True if the edges are equal, false otherwise.
        @rtype: C{bool}
        """
        if isinstance(other, GraphEdge):
            return self.head_vertex == other.head_vertex and self.tail_vertex == other.tail_vertex
        return NotImplemented

    def __ne__(self, other):
        """
        Compares two graph edges for inequality. The comparison
        is done by comparing the vertices constituting the
        respective edge.

        @param other: The other graph edge.
        @type other: L{GraphEdge}
        @return: True if the edges are not equal, false otherwise.
        @rtype: C{bool}
        """
        return not self == other

    def __lt__(self, other):
        """
        Implements the 'less than' operator for this graph edge.

        @param other: The other graph edge.
        @type other: L{GraphEdge}
        @return: True if this graph edge is 'less than' the other, false otherwise.
        @rtype: C{bool}
        """
        return self.head_vertex < other.head_vertex and self.tail_vertex < other.tail_vertex

    def __le__(self, other):
        """
        Implements the 'less than -or equal' operator for this graph edge.

        @param other: The other graph edge.
        @type other: L{GraphEdge}
        @return: True if this graph edge is 'less than -or equal' to the other, false otherwise.
        @rtype: C{bool}
        """
        return self.head_vertex <= other.head_vertex and self.tail_vertex <= other.tail_vertex

    def __gt__(self, other):
        """
        Implements the 'greater than' operator for this graph edge.

        @param other: The other graph edge.
        @type other: L{GraphEdge}
        @return: True if this graph edge is 'greater than' the other, false otherwise.
        @rtype: C{bool}
        """
        return self.head_vertex > other.head_vertex and self.tail_vertex > other.tail_vertex

    def __ge__(self, other):
        """
        Implements the 'greater than -or equal' operator for this graph edge.

        @param other: The other graph edge.
        @type other: L{GraphEdge}
        @return: True if this graph edge is 'greater than -or equal' to the other, false otherwise.
        @rtype: C{bool}
        """
        return self.head_vertex >= other.head_vertex and self.tail_vertex >= other.tail_vertex

    def __hash__(self):
        """
        Returns the hash value of this graph edge.

        @return: Hash value of the graph edge.
        @rtype: C{int}
        """
        return 31 * hash(self.head_vertex) + hash(self.tail_vertex)

    def get_head_vertex(self):
        """
        Returns the first vertex in this graph edge.

        @return: The first vertex in the graph edge.
        @rtype: L{GraphVertex}
        """
        return self.graph.vertices[self.head_vertex.vertex_number]

    def get_tail_vertex(self):
        """
        Returns the second vertex in this graph edge.

        @return: The second vertex in the graph edge.
        @rtype: L{GraphVertex}
        """
        return self.graph.vertices[self.tail_vertex.vertex_number]

    def get_mate(self, vertex):
        """
        Returns the mate vertex connected to the specified
        vertex by this graph edge.

        @param v: The vertex which mate to find.
        @type: C{GraphVertex}
        @raises: ValueError if the specified vertex is not in the graph.
        @type: C{ValueError}
        @return: The mate of the specified vertex of the graph edge.
        @rtype: L{GraphVertex}
        """
        if vertex.vertex_number == self.head_vertex.vertex_number:
            return self.graph.vertices[self.tail_vertex.vertex_number]
        elif vertex.vertex_number == self.tail_vertex.vertex_number:
            return self.graph.vertices[self.head_vertex.vertex_number]
        else:
            raise ValueError

class DirectedGraphEdge(GraphEdge):

    """
    The interface of a directed graph edge data structure.
    """

    def __init__(self, graph, head_vertex, tail_vertex):
        """
        Constructs a directed graph edge connecting the
        two specified vertices found in the specified
        graph.

        @param graph: The graph including the directed edge.
        @type: L{Graph}
        @param v0: The first vertex of the edge.
        @type: L{GraphVertex}
        @param v1: The second vertex of the edge.
        @type: L{GraphVertex}
        """
        super(DirectedGraphEdge, self).__init__(graph, head_vertex, tail_vertex)
        self.directed = True

    def __eq__(self, other):
        """
        Compares two directed graph edges for equality. The comparison
        is done by comparing the vertices constituting the respective
        edges.

        @param other: The other graph edge.
        @type other: L{GraphEdge}
        @return: True if the edges are equal, false otherwise.
        @rtype: C{bool}
        """
        if isinstance(other, DirectedGraphEdge):
            return self.head_vertex == other.head_vertex and self.tail_vertex == other.tail_vertex
        return NotImplemented

    def is_directed(self):
        """
        Returns if this edge is directed.

        @return: True, if the edge is directed, false otherwise.
        @rtype: C{bool}
        """
        return self.directed

class DirectedWeightedGraphEdge(DirectedGraphEdge):

    """
    The interface of a directed weighted graph edge data structure.
    """

    def __init__(self, graph, head_vertex, tail_vertex, weight):
        """
        Constructs a directed weighted graph edge connecting the
        two specified vertices found in the specified graph.

        @param graph: The graph including the directed edge.
        @type: L{DirectedWeightedGraph}
        @param v0: The first vertex of the edge.
        @type: L{GraphVertex}
        @param v1: The second vertex of the edge.
        @type: L{GraphVertex}
        """
        super(DirectedWeightedGraphEdge, self).__init__(graph, head_vertex, tail_vertex)
        self.weighted = True
        self.weight = weight

    def __repr__(self):
        """
        Returns the canonical representation of this directed
        weighted graph edge.

        @return: Canonical string representation of the graph edge.
        @rtype: C{str}
        """
        return repr((self.head_vertex, self.tail_vertex, self.weight))

    def __str__(self):
        """
        Returns a string representation of this directed
        weighted graph edge.

        @return: String representation of the graph edge.
        @rtype: C{str}
        """
        class_name_str = str(self.__class__.__name__) + ": ("
        head_str = str(self.head_vertex) + ", "
        tail_str = str(self.tail_vertex) + ", "
        weight_str = str(self.weight) + ")"
        attributes_str = head_str + tail_str + weight_str
        str_rep = class_name_str + attributes_str
        return str_rep

    def __eq__(self, other):
        """
        Compares two directed weighted graph edges for equality. The
        comparison is done by comparing the vertices constituting
        the respective edges and the weight of the respective edges.

        @param other: The other graph edge.
        @type other: L{DirectedWeightedGraphEdge}
        @return: True if the edges are equal, false otherwise.
        @rtype: C{bool}
        """
        if isinstance(other, DirectedWeightedGraphEdge):
            if self.head_vertex != other.head_vertex:
                return False
            elif self.tail_vertex != other.tail_vertex:
                return False
            elif self.weight != other.weight:
                return False
            return True
        return NotImplemented

    def __ne__(self, other):
        """
        Compares two directed weighted graph edges for inequality. The
        comparison is done by comparing the vertices constituting
        the respective edges and the weight of the respective edges.

        @param other: The other graph edge.
        @type other: L{DirectedWeightedGraphEdge}
        @return: True if the edges are equal, false otherwise.
        @rtype: C{bool}
        """
        return not self == other

    def __lt__(self, other):
        """
        Implements the 'less than' operator for this directed
        weighted graph edge.

        @param other: The other graph edge.
        @type other: L{DirectedWeightedGraphEdge}
        @return: True if this graph edge is 'less than' the other, false otherwise.
        @rtype: C{bool}
        """
        if self.head_vertex >= other.head_vertex:
            return False
        elif self.tail_vertex >= other.tail_vertex:
            return False
        elif self.weight >= other.weight:
            return False
        return True

    def __le__(self, other):
        """
        Implements the 'less than -or equal' operator for
        this directed weighted graph edge.

        @param other: The other graph edge.
        @type other: L{DirectedWeightedGraphEdge}
        @return: True if this graph edge is 'less than -or equal' to the other, false otherwise.
        @rtype: C{bool}
        """
        if self.head_vertex > other.head_vertex:
            return False
        elif self.tail_vertex > other.tail_vertex:
            return False
        elif self.weight > other.weight:
            return False
        return True

    def __gt__(self, other):
        """
        Implements the 'greater than' operator for this
        directed weighted graph edge.

        @param other: The other graph edge.
        @type other: L{DirectedWeightedGraphEdge}
        @return: True if this graph edge is 'greater than' the other, false otherwise.
        @rtype: C{bool}
        """
        if self.head_vertex <= other.head_vertex:
            return False
        elif self.tail_vertex <= other.tail_vertex:
            return False
        elif self.weight <= other.weight:
            return False
        return True

    def __ge__(self, other):
        """
        Implements the 'greater than -or equal' operator for
        this directed weighted graph edge.

        @param other: The other graph edge.
        @type other: L{DirectedWeightedGraphEdge}
        @return: True if this graph edge is 'greater than -or equal' to the other, false otherwise.
        @rtype: C{bool}
        """
        if self.head_vertex < other.head_vertex:
            return False
        elif self.tail_vertex < other.tail_vertex:
            return False
        elif self.weight < other.weight:
            return False
        return True

    def __hash__(self):
        """
        Returns the hash value of this directed weighted
        graph edge.

        @return: Hash value of the graph edge.
        @rtype: C{int}
        """
        return 31 * hash(self.head_vertex) + hash(self.tail_vertex) + hash(self.weight)

    def is_weighted(self):
        """
        Returns if this edge is weighted.
        Note: This method does always return True.

        @return: True, since the edge is weighted.
        @rtype: C{bool}
        """
        return self.weighted

    def get_weight(self):
        """
        Returns the weight of the directed weighted
        graph edge.

        @return: The weight of the graph edge.
        @rtype: C{int}
        """
        return self.weight

class DirectedUnWeightedGraphEdge(DirectedGraphEdge):

    """
    The interface of a directed unweighted graph edge data structure.
    """

    def __init__(self, graph, head_vertex, tail_vertex):
        """
        Constructs a directed unweighted graph edge connecting the
        two specified vertices found in the specified graph.

        @param graph: The graph including the directed edge.
        @type: L{DirectedUnWeightedGraph}
        @param v0: The first vertex of the edge.
        @type: L{GraphVertex}
        @param v1: The second vertex of the edge.
        @type: L{GraphVertex}
        """
        super(DirectedUnWeightedGraphEdge, self).__init__(graph, head_vertex, tail_vertex)
        self.weighted = False

    def is_weighted(self):
        """
        Returns if this edge is weighted.
        Note: This method does always return False.

        @return: False, since the edge is unweighted.
        @rtype: C{bool}
        """
        return self.weighted

class UnDirectedGraphEdge(GraphEdge):

    """
    The interface of an undirected graph edge data structure.
    """

    def __init__(self, graph, head_vertex, tail_vertex):
        """
        Constructs an undirected graph edge connecting the
        two specified vertices found in the specified
        graph.

        @param graph: The graph including the undirected edge.
        @type: L{Graph}
        @param v0: The first vertex of the edge.
        @type: L{GraphVertex}
        @param v1: The second vertex of the edge.
        @type: L{GraphVertex}
        """
        super(UnDirectedGraphEdge, self).__init__(graph, head_vertex, tail_vertex)
        self.directed = False

    def is_directed(self):
        """
        Returns if this edge is directed.

        @return: True, if the edge is directed, false otherwise.
        @rtype: C{bool}
        """
        return self.directed

class UnDirectedWeightedGraphEdge(UnDirectedGraphEdge):

    """
    The interface of an undirected weighted graph edge data structure.
    """

    def __init__(self, graph, head_vertex, tail_vertex, weight):
        """
        Constructs an undirected weighted graph
        edge connecting the two specified
        vertices found in the specified graph.

        @param graph: The graph including the edge.
        @type: L{Graph}
        @param v0: The first vertex of the edge.
        @type: L{GraphVertex}
        @param v1: The second vertex of the edge.
        @type: L{GraphVertex}
        @param weight: The weight of the edge.
        """
        super(UnDirectedWeightedGraphEdge, self).__init__(graph, head_vertex, tail_vertex)
        self.weight = weight

    def __repr__(self):
        """
        Returns the canonical representation of this graph edge.

        @return: Canonical string representation of the graph edge.
        @rtype: C{str}
        """
        return repr((self.head_vertex, self.tail_vertex, self.weight))

    def __str__(self):
        """
        Returns a string representation of this graph edge.

        @return: String representation of the graph edge.
        @rtype: C{str}
        """
        class_name_str = str(self.__class__.__name__) + ": ("
        head_str = str(self.head_vertex) + ", "
        tail_str = str(self.tail_vertex) + ", "
        weight_str = str(self.weight) + ")"
        attributes_str = head_str + tail_str + weight_str
        str_rep = class_name_str + attributes_str
        return str_rep

    def __eq__(self, other):
        """
        Compares two undirected weighted graph edges for
        equality. The comparison is done by comparing the
        vertices constituting the respective edges and the
        weight of the respective edges.

        @param other: The other graph edge.
        @type other: L{UnDirectedWeightedGraphEdge}
        @return: True if the edges are equal, false otherwise.
        @rtype: C{bool}
        """
        if isinstance(other, UnDirectedWeightedGraphEdge):
            if self.head_vertex != other.head_vertex:
                return False
            elif self.tail_vertex != other.tail_vertex:
                return False
            elif self.weight != other.weight:
                return False
            return True
        return NotImplemented

    def __ne__(self, other):
        """
        Compares two undirected weighted graph edges for
        inequality. The comparison is done by comparing the
        vertices constituting the respective edges and the
        weight of the respective edges.

        @param other: The other graph edge.
        @type other: L{UnDirectedWeightedGraphEdge}
        @return: True if the edges are equal, false otherwise.
        @rtype: C{bool}
        """
        return not self == other

    def __lt__(self, other):
        """
        Implements the 'less than' operator for this undirected
        weighted graph edge.

        @param other: The other graph edge.
        @type other: L{UnDirectedWeightedGraphEdge}
        @return: True if this graph edge is 'less than' the other, false otherwise.
        @rtype: C{bool}
        """
        if self.head_vertex >= other.head_vertex:
            return False
        elif self.tail_vertex >= other.tail_vertex:
            return False
        elif self.weight >= other.weight:
            return False
        return True

    def __le__(self, other):
        """
        Implements the 'less than -or equal' operator for
        this undirected weighted graph edge.

        @param other: The other graph edge.
        @type other: L{UnDirectedWeightedGraphEdge}
        @return: True if this graph edge is 'less than -or equal' to the other, false otherwise.
        @rtype: C{bool}
        """
        if self.head_vertex > other.head_vertex:
            return False
        elif self.tail_vertex > other.tail_vertex:
            return False
        elif self.weight > other.weight:
            return False
        return True

    def __gt__(self, other):
        """
        Implements the 'greater than' operator for this
        undirected weighted graph edge.

        @param other: The other graph edge.
        @type other: L{UnDirectedWeightedGraphEdge}
        @return: True if this graph edge is 'greater than' the other, false otherwise.
        @rtype: C{bool}
        """
        if self.head_vertex <= other.head_vertex:
            return False
        elif self.tail_vertex <= other.tail_vertex:
            return False
        elif self.weight <= other.weight:
            return False
        return True

    def __ge__(self, other):
        """
        Implements the 'greater than -or equal' operator for
        this undirected weighted graph edge.

        @param other: The other graph edge.
        @type other: L{UnDirectedWeightedGraphEdge}
        @return: True if this graph edge is 'greater than -or equal' to the other, false otherwise.
        @rtype: C{bool}
        """
        if self.head_vertex < other.head_vertex:
            return False
        elif self.tail_vertex < other.tail_vertex:
            return False
        elif self.weight < other.weight:
            return False
        return True

    def __hash__(self):
        """
        Returns the hash value of this undirected weighted
        graph edge.

        @return: Hash value of the graph edge.
        @rtype: C{int}
        """
        return 31 * hash(self.head_vertex) + hash(self.tail_vertex) + hash(self.weight)

    def get_weight(self):
        """
        Returns the weight of the undirected weighted
        graph edge.

        @return: The weight of the graph edge.
        @rtype: C{int}
        """
        return self.weight
    ############################################################ AK ######################################
    def set_weight(self,value):
        self.weight = value

class UnDirectedUnWeightedGraphEdge(UnDirectedGraphEdge):

    """
    The interface of an undirected unweighted graph edge data structure.
    """

    def __init__(self, graph, head_vertex, tail_vertex):
        """
        Constructs an undirected unweighted graph edge connecting the
        two specified vertices found in the specified graph.

        @param graph: The graph including the undirected edge.
        @type: L{UnDirectedUnWeightedGraph}
        @param v0: The first vertex of the edge.
        @type: L{GraphVertex}
        @param v1: The second vertex of the edge.
        @type: L{GraphVertex}
        """
        super(UnDirectedUnWeightedGraphEdge, self).__init__(graph, head_vertex, tail_vertex)
        self.weighted = False

    def is_weighted(self):
        """
        Returns if this edge is weighted.
        Note: This method does always return False.

        @return: False, since the edge is unweighted.
        @rtype: C{bool}
        """
        return self.weighted

class EdgeClassification(object):

    """
    Provides a simple edge classification.
    """

    TREE_EDGE = "Tree"
    BACK_EDGE = "Back"
    FORWARD_EDGE = "Forward"
    CROSS_EDGE = "Cross"


