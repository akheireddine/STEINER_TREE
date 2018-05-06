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
Provides a class used to model a vertex in a graph.
"""

__author__ = "Brian Horn"
__copyright__ = "Copyright (c) 2015 Brian Horn"
__credits__ = "Brian Horn"
__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Brian Horn"
__email__ = "trycatchhorn@gmail.com"
__status__ = "Prototype"

class GraphVertex(object):

    """
    Implements a graph vertex.
    """

    def __init__(self, graph, vertex_name):
        """
        Constructs a graph vertex.

        @param graph: The graph of this graph vertex.
        @type: C{object}
        @param vertex_name: The name of this graph vertex.
        @type: C{str}
        """
        self.graph = graph
        self.vertex_name = vertex_name
        self.vertex_number = -1
        for i in xrange(len(graph.vertices)):
            if graph.vertices[i] is None:
                self.vertex_number = i
                return

    def __repr__(self):
        """
        Returns the canonical representation of this graph vertex.

        @return: Canonical string representation of the graph vertex.
        @rtype: C{str}
        """
        return repr((self.vertex_name, self.vertex_number))

    def __str__(self):
        """
        Returns a string representation of this graph vertex.

        @return: String representation of the graph vertex.
        @rtype: C{str}
        """
        class_name_str = str(self.__class__.__name__) + ": ("
        attributes_str = str(self.vertex_name) + ")"#", " + str(self.vertex_number) + ")"
        str_rep = class_name_str + attributes_str
        return str_rep

    def __hash__(self):
        """
        Returns the hash value of this graph vertex.
        The hash value is composed of the name of
        the vertex and the vertex number.

        @return: Hash value of the graph vertex.
        @rtype: C{int}
        """
        return 31 * hash(self.vertex_name) + hash(self.vertex_number)

    def __eq__(self, other):
        """
        Compares two graph vertices for equality. The comparison
        is done by comparing the name field of the two
        graph vertices.

        @param other: The other graph vertex.
        @type other: L{GraphVertex}
        @return: True if the vertices are equal, false otherwise.
        @rtype: C{bool}
        """
        return self.vertex_name == other.vertex_name and self.vertex_number == other.vertex_number

    def __ne__(self, other):
        """
        Compares two graph vertices for inequality. The comparison
        is done by comparing the name field of the two
        graph vertices.

        @param other: The other graph vertex.
        @type other: L{GraphVertex}
        @return: True if the vertices are not equal, false otherwise.
        @rtype: C{bool}
        """
        return not self == other

    def __lt__(self, other):
        """
        Implements the 'less than' operator for this graph vertex.

        @param other: The other graph vertex.
        @type other: L{GraphVertex}
        @return: True if this vertex is 'less than' to the other, false otherwise.
        @rtype: C{bool}
        """
        return self.vertex_name < other.vertex_name and self.vertex_number < other.vertex_number

    def __le__(self, other):
        """
        Implements the 'less than -or equal' operator for this graph vertex.

        @param other: The other graph vertex.
        @type other: L{GraphVertex}
        @return: True if this vertex is 'less than -or equal' to the other, false otherwise.
        @rtype: C{bool}
        """
        return self.vertex_name <= other.vertex_name and self.vertex_number <= other.vertex_number

    def __gt__(self, other):
        """
        Implements the 'greater than' operator for this graph vertex.

        @param other: The other graph vertex.
        @type other: L{GraphVertex}
        @return: True if this vertex is 'greater than' the other, false otherwise.
        @rtype: C{bool}
        """
        return self.vertex_name > other.vertex_name and self.vertex_number > other.vertex_number

    def __ge__(self, other):
        """
        Implements the 'greater than -or equal' operator for this graph vertex.

        @param other: The other graph vertex.
        @type other: L{GraphVertex}
        @return: True if this vertex is 'greater than -or equal' to the other, false otherwise.
        @rtype: C{bool}
        """
        return self.vertex_name >= other.vertex_name and self.vertex_number >= other.vertex_number

    def get_vertex_number(self):
        """
        Returns the number of this vertex.

        @return: The number of the vertex.
        @rtype: C{int}
        """
        return self.vertex_number

    def get_vertex_name(self):
        """
        Returns the name of this vertex.

        @return: The name of the vertex.
        @rtype: C{str}
        """
        return self.vertex_name

    def get_emanating_edges(self):
        """
        Returns a list enumerating the emanating edges of this vertex.

        @return: The emanating edges of the vertex.
        @rtype: C{list}
        """
        return self.graph.get_emanating_edges(self.vertex_number)

    def get_incident_edges(self):
        """
        Returns a list enumerating the incident edges of this vertex.

        @return: The incident edges of the vertex.
        @rtype: C{list}
        """
        return self.graph.get_incident_edges(self.vertex_number)

    def predecessor_generator(self):
        """
        Returns a generator enumerating the predecessor
        vertices of this vertex.

        @return: The predecessor vertices of this vertex.
        @rtype: C{list}
        """
        edges = self.graph.get_incident_edges(self.vertex_number)
        for edge in edges:
            yield edge.get_mate(self)

    def get_predecessors(self):
        """
        Returns a list enumerating the predecessor
        vertices of this vertex.

        @return: The predecessor vertices of the vertex.
        @rtype: C{list}
        """
        return list(self.predecessor_generator())

    def successor_generator(self):
        """
        Returns a generator enumerating the successor
        vertices of this vertex.

        @return: The successor vertices of this vertex.
        @rtype: C{list}
        """
        edges = self.graph.get_emanating_edges(self.vertex_number)
        for edge in edges:
            yield edge.get_mate(self)

    def get_successors(self):
        """
        Returns a list enumerating the successor
        vertices of this vertex.

        @return: The successor vertices of the vertex.
        @rtype: C{list}
        """
        return list(self.successor_generator())

class UnWeightedGraphVertex(GraphVertex):

    """
    Implements an unweighted graph vertex.
    """

    def __init__(self, graph, vertex_name):
        """
        Constructs an unweighted graph vertex.

        @param graph: The graph of this unweighted graph vertex.
        @type: C{object}
        @param vertex_name: The name of this unweighted graph vertex.
        @type: C{str}
        """
        super(UnWeightedGraphVertex, self).__init__(graph, vertex_name)

    def __repr__(self):
        """
        Returns the canonical representation of this unweighted graph vertex.

        @return: Canonical string representation of the unweighted graph vertex.
        @rtype: C{str}
        """
        return repr((self.vertex_name, self.vertex_number))

    def __str__(self):
        """
        Returns a string representation of this unweighted graph vertex.

        @return: String representation of the unweighted graph vertex.
        @rtype: C{str}
        """
        class_name_str = str(self.__class__.__name__) + ": ("
        attributes_str = str(self.vertex_name) + ")" #+ ", " + str(self.vertex_number) + ")"
        str_rep = class_name_str + attributes_str
        return str_rep

    def __hash__(self):
        """
        Returns the hash value of this unweighted graph vertex.
        The hash value is composed of the name of the vertex
        and the vertex number.

        @return: Hash value of the unweighted graph vertex.
        @rtype: C{int}
        """
        return 31 * hash(self.vertex_name) + hash(self.vertex_number)

    def __eq__(self, other):
        """
        Compares two unweighted graph vertices for equality. The comparison
        is done by comparing the name field of the two unweighted graph
        vertices.

        @param other: The other unweighted graph vertex.
        @type other: L{UnWeightedGraphVertex}
        @return: True if the vertices are equal, false otherwise.
        @rtype: C{bool}
        """
        if isinstance(other, UnWeightedGraphVertex):
            if self.vertex_name != other.vertex_name:
                return False
            elif self.vertex_number != other.vertex_number:
                return False
            return True
        return NotImplemented

    def __ne__(self, other):
        """
        Compares two unweighted graph vertices for inequality. The comparison
        is done by comparing the name field of the two unweighted graph
        vertices.

        @param other: The other unweighted graph vertex.
        @type other: L{UnWeightedGraphVertex}
        @return: True if the vertices are not equal, false otherwise.
        @rtype: C{bool}
        """
        return not self == other

    def __lt__(self, other):
        """
        Implements the 'less than' operator for this unweighted graph vertex.

        @param other: The other unweighted graph vertex.
        @type other: L{UnWeightedGraphVertex}
        @return: True if this vertex is 'less than' the other, false otherwise.
        @rtype: C{bool}
        """
        return self.vertex_name < other.vertex_name and self.vertex_number < other.vertex_number

    def __le__(self, other):
        """
        Implements the 'less than -or equal' operator for this unweighted graph vertex.

        @param other: The other unweighted graph vertex.
        @type other: L{UnWeightedGraphVertex}
        @return: True if this vertex is 'less than -or equal' to the other, false otherwise.
        @rtype: C{bool}
        """
        return self.vertex_name <= other.vertex_name and self.vertex_number <= other.vertex_number

    def __gt__(self, other):
        """
        Implements the 'greater than' operator for this unweighted graph vertex.

        @param other: The other unweighted graph vertex.
        @type other: L{UnWeightedGraphVertex}
        @return: True if this vertex is 'greater than' the other, false otherwise.
        @rtype: C{bool}
        """
        return self.vertex_name > other.vertex_name and self.vertex_number > other.vertex_number

    def __ge__(self, other):
        """
        Implements the 'greater than -or equal' operator for this unweighted graph vertex.

        @param other: The other unweighted graph vertex.
        @type other: L{UnWeightedGraphVertex}
        @return: True if this vertex is 'greater than -or equal' to the other, false otherwise.
        @rtype: C{bool}
        """
        return self.vertex_name >= other.vertex_name and self.vertex_number >= other.vertex_number
