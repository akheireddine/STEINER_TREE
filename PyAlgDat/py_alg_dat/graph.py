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
Provides an abstract interface of a graph.

The interface includes the most common methods needed to realize
the ADT for a graph.
"""

__author__ = "Brian Horn"
__copyright__ = "Copyright (c) 2015 Brian Horn"
__credits__ = "Brian Horn"
__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Brian Horn"
__email__ = "trycatchhorn@gmail.com"
__status__ = "Prototype"

import copy
from abc import abstractmethod
from py_alg_dat.array_list import ArrayList
from py_alg_dat.container import Container
from py_alg_dat.doubly_linked_list import DoublyLinkedList
from py_alg_dat.dfs_edge_classification import DFSEdgeClassification
from py_alg_dat.graph_edge import DirectedGraphEdge
from py_alg_dat.graph_edge import DirectedUnWeightedGraphEdge
from py_alg_dat.graph_edge import DirectedWeightedGraphEdge
from py_alg_dat.graph_edge import UnDirectedGraphEdge
from py_alg_dat.graph_edge import UnDirectedUnWeightedGraphEdge
from py_alg_dat.graph_edge import UnDirectedWeightedGraphEdge
from py_alg_dat.graph_visitor import GraphVisitor
from py_alg_dat.queue import Queue
from py_alg_dat.stack import Stack
from py_alg_dat.visitor import Visitor
from py_alg_dat.vertex_visitor import VertexVisitor

class Graph(Container):

    """
    The interface of a graph data structure.
    """

    def __init__(self, size):
        """
        Constructs a graph with the number of vertices specified by
        the size parameter.

        @param size: The number of vertices contained in this graph.
        @type: C{int}
        """
        super(Graph, self).__init__()
        self.size = size
        self.vertices = ArrayList(size)
        self.adjacency_list = ArrayList(size)
        for i in xrange(size):
            # NOTE: the SinglyLinkedList and the DoublyLikedList classes
            # have the same interface, therefore it should be possible to
            # toggle between these and still pass all unit tests.

            #from py_alg_dat.singly_linked_list import SinglyLinkedList
            #self.adjacency_list[i] = SinglyLinkedList()
            self.adjacency_list[i] = DoublyLinkedList()

    def __str__(self):
        """
        Returns a string representation of this graph using the
        visitor pattern to visit each vertex in the graph.

        @return: The string representation of the graph.
        @rtype: C{str}
        """
        visitor = GraphVisitor()
        self.accept(visitor)
        str_rep = str(self.__class__.__name__) + ": \n" + str(visitor)
        return str_rep

    def __eq__(self, other):
        """
        Compares two graphs for equality. The comparison
        is done by comparing the attributes: size, vertices,
        and adjacency_list of the respective graphs.

        @param other: The other graph.
        @type other: L{Graph}
        @return: True if the graphs are equal, false otherwise.
        @rtype: C{bool}
        """
        if isinstance(other, Graph):
            return self.is_equal(other)

    def __ne__(self, other):
        """
        Compares two graphs for inequality. The comparison
        is done by comparing the attributes: size, vertices,
        and adjacency_list of the respective graphs.

        @param other: The other graph.
        @type other: L{Graph}
        @return: True if the graphs are inequal, false otherwise.
        @rtype: C{bool}
        """
        return not self == other

    def is_equal(self, other):
        """
        Compares two graphs for equality. The comparison
        is done by comparing the attributes: size, vertices,
        and adjacency_list of the respective graphs.

        NOTE: This method is called by all __eq__ methods in
        the graph inheritance hierarchy to avoid the redundancy
        introduced if this functionality was placed directly
        in each of the __eq__ methods in each subclass.

        @param other: The other graph.
        @type other: L{Graph}
        @return: True if the graphs are equal, false otherwise.
        @rtype: C{bool}
        """
        if self.size != other.size:
            return False
        elif self.vertices != other.vertices:
            return False
        elif self.adjacency_list != other.adjacency_list:
            return False
        else:
            return True

    def do_copy(self, other):
        """
        Performs a shallow copy of this graph. The copy is
        done by copying the individual fields of this graph
        into a new graph object, where the graph type is
        specified by the other parameter.

        NOTE: This method is called by all __copy__ methods in
        the graph inheritance hierarchy to avoid the redundancy
        introduced if this functionalty was placed directly
        in each of the __copy__ methods in each subclass.

        @param other: Specifies the type of graph.
        @type other: L{Graph}
        @return: A shallow copy of this graph.
        @rtype: L{Graph}
        """
        result = None
        if type(other) == Graph:
            result = Graph(self.size)
        elif type(other) == UnDirectedGraph:
            result = UnDirectedGraph(self.size)
        elif type(other) == UnDirectedUnWeightedGraph:
            result = UnDirectedUnWeightedGraph(self.size)
        elif type(other) == UnDirectedWeightedGraph:
            result = UnDirectedWeightedGraph(self.size)
        elif type(other) == DirectedGraph:
            result = DirectedGraph(self.size)
        elif type(other) == DirectedWeightedGraph:
            result = DirectedWeightedGraph(self.size)
        elif type(other) == DirectedUnWeightedGraph:
            result = DirectedUnWeightedGraph(self.size)
        # Check that we have a instance of graph or one of its subtypes
        if isinstance(result, Graph):
            result.size = self.size
            result.vertices = copy.copy(ArrayList(self.size))
            result.adjacency_list = copy.copy(ArrayList(self.size))
            for i in xrange(self.size):
                result.vertices[i] = copy.copy(self.vertices[i])
                result.adjacency_list[i] = copy.copy(self.adjacency_list[i])
        return result

    def __len__(self):
        """
        Returns the number of vertices contained in this graph.

        @return: The number of vertices in the graph.
        @rtype: C{int}
        """
        return self.get_number_of_vertices()

    def __getitem__(self, index):
        """
        Returns the vertex at the specified index.

        @param index: The index of the vertex.
        @type: C{object}
        @return: The vertex at the specified index.
        @rtype: C{int}
        """
        return self.get_vertex_at_index(index)

    def __iter__(self):
        """
        Returns an iterator that enumerates the vertices of this graph.

        @return: Iterator enumerating the vertices of the graph.
        @rtype: C{object}
        """
        return iter(self.vertices)

    def accept(self, visitor):
        """
        Makes the given visitor visit all the vertices in this graph.

        @param visitor: The visitor used to visit each vertex in the graph.
        @type: L{Visitor}
        """
        assert isinstance(visitor, Visitor)
        for i in xrange(len(self.vertices)):
            visitor.visit(self.vertices[i])

    def add_vertex(self, vertex):
        """
        Adds a vertex to this graph.

        @param vertex: The vertex to be added to the graph.
        @type: L{UnWeightedGraphVertex}
        """
        vertex_number = self.get_number_of_vertices()
        vertex.vertex_number = vertex_number
        self.vertices[vertex_number] = vertex

    def remove_vertex(self, vertex):
        """
        Removes a vertex from this graph.

        @param vertex: The vertex to be removed from the graph.
        @type: L{UnWeightedGraphVertex}
        """
        for edge in self.get_edges():
            if edge.get_head_vertex() == vertex or edge.get_tail_vertex() == vertex:
                self.remove_edge(edge.get_head_vertex(), edge.get_tail_vertex())
        # Remove the specified vertex from the list of vertices.
        del self.vertices[vertex.get_vertex_number()]
        # Remove the specified vertex from the adjacency list.
        del self.adjacency_list[vertex.get_vertex_number()]
        # Now the vertices in the graph might not be numbered correctly
        # according to their position. This is fixed by iterating
        # through the list of vertices and correcting the vertex numbers.
        for i in xrange(vertex.get_vertex_number(), len(self.vertices)):
            self.vertices[i].vertex_number = i
        self.size = len(self.vertices)

    def remove_edge(self, vertex_u, vertex_v):
        """
        Removes the edge connecting the vertices u and v
        from this graph.

        @param u: The vertex from where the edge starts.
        @param v: The vertex from where the edge ends.
        """
        try:
            edge_uv = self.get_edge(vertex_u, vertex_v)
            adj_u = self.adjacency_list[vertex_u.get_vertex_number()]
            adj_u_copy = copy.copy(adj_u)
            adj_u_copy.remove(edge_uv)
            if len(adj_u_copy) == len(adj_u) - 1:
                self.adjacency_list[vertex_u.get_vertex_number()] = adj_u_copy

            if not self.is_directed():
                edge_vu = self.get_edge(vertex_v, vertex_u)
                adj_v = self.adjacency_list[vertex_v.get_vertex_number()]
                adj_v_copy = copy.copy(adj_v)
                adj_v_copy.remove(edge_vu)
                if len(adj_v_copy) == len(adj_v) - 1:
                    self.adjacency_list[vertex_v.get_vertex_number()] = adj_v_copy
        except KeyError:
            return

    @abstractmethod
    def  __copy__(self):
        """
        Returns a shallow copy of this graph.

        @return: A copy of the graph.
        @rtype: C{object}
        """
        pass

    @abstractmethod
    def add_edge(self, vertex_u, vertex_v, weight=0):
        """
        Adds an edge to this graph.

        @param u: The tail vertex of the edge.
        @type: C{object}
        @param v: The head vertex of the edge.
        @type: C{object}
        """
        pass

    @abstractmethod
    def is_directed(self):
        """
        Returns if this graph is directed.

        @return: True if the graph is directed, otherwise False.
        @rtype: C{bool}
        """
        pass

    @abstractmethod
    def is_weighted(self):
        """
        Returns if this graph is weighted.

        @return: True if the graph is weighted, otherwise False.
        @rtype: C{bool}
        """
        pass

    @abstractmethod
    def is_cyclic(self):
        """
        Returns if this graph contains one or more cycles.

        @return: True, if the graph has one or cycles, otherwise False.
        @rtype: C{bool}
        """
        pass

    def get_number_of_vertices(self):
        """
        Returns the number of vertices contained in this graph.

        @return: The number of vertices in the graph.
        @rtype: C{int}
        """
        return len([x for x in self.vertices if x is not None])

    def get_number_of_edges(self):
        """
        Returns the number of edges contained in this graph.

        @return: The number of edges in the graph.
        @rtype: C{int}
        """
        return len([x for x in self.get_edges() if x is not None])

    def get_vertices(self):
        """
        Returns a list of the vertices in this graph.

        @return: List of the vertices in the graph.
        @rtype: C{list}
        """
        result = list(self.vertices_generator())
        return result

    def vertices_generator(self):
        """
        Returns a generator that enumerates the vertices of this graph.

        @return: Generator enumerating the vertices of the graph.
        @rtype: C{object}
        """
        for vertex in self.vertices:
            yield vertex

    def get_edges(self):
        """
        Returns a list of edges in this graph.

        @return: List of edges in the graph.
        @rtype: C{list}
        """
        result = list(self.edges_generator())
        return result

    def edges_generator(self):
        """
        Returns a generator that enumerates the edges of this graph.

        @return: Generator enumerating the edges of the graph.
        @rtype: C{object}
        """
        for i in xrange(len(self.adjacency_list)):
            for j in xrange(len(self.adjacency_list[i])):
                if self.adjacency_list[i][j] != None:
                    yield self.adjacency_list[i][j].data

    def get_vertex_at_index(self, index):
        """
        Returns the vertex at the specified index.

        @param index: The index of the vertex.
        @type: C{object}
        @return: The vertex at the specified index.
        @rtype: C{int}
        """
        if index < 0 or index >= self.get_number_of_vertices():
            raise IndexError
        return self.vertices[index]

    def has_vertex(self, vertex):
        """
        Checks if a vertex is present in this graph.

        @param vertex: The vertex to search for.
        @type: C{object}
        @return: True if the vertex is present, false otherwise.
        @rtype: C{bool}
        """
        result = False
        for i in self.get_vertices():
            if i == vertex:
                result = True
        return result

    def has_edge(self, edge):
        """
        Checks if the specified edge is present in this graph.

        @param edge: The edge to search for.
        @type: C{object}
        @return: True if the edge is present, false otherwise.
        @rtype: C{bool}
        """
        result = False
        if edge is not None:
            vertex_u = edge.head_vertex
            vertex_v = edge.tail_vertex
            try:
                res = self.get_edge(vertex_u, vertex_v)
                if res != None:
                    result = True
            except IndexError:
                result = False
        return result

    def get_edge(self, vertex_u, vertex_v):
        """
        Returns the edge connecting the specified vertices in this graph.

        @param u: The tail vertex.
        @type: C{object}
        @param v: The head vertex.
        @type: C{object}
        @return: The edge connecting the vertices v and w.
        @rtype: C{object}
        """
        if vertex_u.get_vertex_number() < 0:
            raise IndexError
        elif vertex_u.get_vertex_number() >= self.get_number_of_vertices():
            raise IndexError
        elif vertex_v.get_vertex_number() < 0:
            raise IndexError
        elif vertex_v.get_vertex_number() >= self.get_number_of_vertices():
            raise IndexError
        ptr = self.adjacency_list[vertex_u.get_vertex_number()].head
        while ptr is not None:
            edge = ptr.data
            tail_vertex_number = edge.get_tail_vertex().get_vertex_number()
            tail_vertex_name = edge.get_tail_vertex().get_vertex_name()
            vertex_v_number = vertex_v.get_vertex_number()
            vertex_v_name = vertex_v.get_vertex_name()
            if tail_vertex_number == vertex_v_number and tail_vertex_name == vertex_v_name:
                return edge
            ptr = ptr.next
        return

    def is_edge(self, vertex_u, vertex_v):
        """
        Returns if there exists an edge connecting the specified vertices in this graph.

        @param u: The tail vertex.
        @type: C{object}
        @param v: The head vertex.
        @type: C{object}
        @return: True, if there exists an edge connecting the specified vertices, otherwise false.
        @rtype: C{boolean}
        """
        edge = self.get_edge(vertex_u, vertex_v)
        if edge is not None:
            return True
        return False

    def emanating_edge_generator(self, index):
        """
        Returns a generator enumerating the emanating edges to the vertex with the specified
        vertex index.

        @param index: Index specifying from which vertex the emanating edges should be returned.
        @type: C{int}
        @return: Generator enumerating the emanating edges of the specified vertex index.
        @rtype: C{object}
        """
        i = 0
        while i >= 0 and i < self.get_number_of_vertices():
            ptr = self.adjacency_list[i].head
            while ptr is not None:
                if ptr.data is not None:
                    if index == ptr.data.get_head_vertex().get_vertex_number():
                        yield ptr.data
                ptr = ptr.next
            i += 1

    def get_emanating_edges(self, index):
        """
        Returns the emanating edges to the vertex with the specified
        vertex index.

        @param index: Index specifying from which vertex the emanating edges should be returned.
        @type: C{int}
        @return: List enumerating the emanating edges of the specified vertex index.
        @rtype: C{list}
        """
        return list(self.emanating_edge_generator(index))

    def get_incident_edges(self, index):
        """
        Returns the incident edges to the vertex with the specified
        vertex index.

        @return: The incident edges to the specified vertex.
        @rtype: C{list}
        """
        result = list(self.incident_edge_generator(index))
        return result

    def incident_edge_generator(self, index):
        """
        Returns a generator enumerating the incident edges to the vertex with the specified
        vertex index.

        @param index: Index specifying from which vertex the incident edges should be returned.
        @type: C{int}
        @return: Generator enumerating the incident edges to the specified vertex.
        @rtype: C{object}
        """
        i = 0
        while i >= 0 and i < self.get_number_of_vertices():
            ptr = self.adjacency_list[i].head
            while ptr is not None:
                if self.is_directed():
                    if index == ptr.data.get_tail_vertex().get_vertex_number():
                        yield ptr.data
                else:
                    if index == ptr.data.get_head_vertex().get_vertex_number():
                        yield ptr.data
                ptr = ptr.next
            i += 1

    def get_out_degree(self, vertex):
        """
        Returns the number of edges which emanating the specified vertex.

        @param vertex: The vertex from which the number of emanating edges should be returned.
        @type: L{object}
        @return: The number of edges emanating to the specified vertex.
        @rtype: C{int}
        """
        return len(self.get_emanating_edges(vertex.get_vertex_number()))

    def get_in_degree(self, vertex):
        """
        Returns the number of edges which incident the specified vertex.

        @param vertex: The vertex from which the number of incident edges should be returned.
        @type: L{object}
        @return: The number of edges incident to the specified vertex.
        @rtype: C{int}
        """
        return len(self.get_incident_edges(vertex.get_vertex_number()))

    def breadth_first_traversal(self, visitor, start):
        """
        Performs a Breadth-First-Search of this graph.
        Each vertex in the graph is visited in a
        breadth first manner starting from the given vertex.

        @param visitor: The visitor being applied to each vertex.
        @type: L{Visitor}
        @param start: The vertex from where the search begins.
        @type: C{int}
        """
        assert isinstance(visitor, Visitor)
        number_of_vertices = self.get_number_of_vertices()
        enqueued = ArrayList(number_of_vertices)
        for vertex in xrange(number_of_vertices):
            enqueued[vertex] = False
        queue = Queue()
        queue.enqueue(self[start])
        enqueued[start] = True
        while not queue.is_empty() and not visitor.is_done():
            vertex = queue.dequeue()
            visitor.visit(vertex)
            for successor in vertex.get_successors():
                if not enqueued[successor.get_vertex_number()]:
                    queue.enqueue(successor)
                    enqueued[successor.get_vertex_number()] = True

    def depth_first_traversal(self, visitor, start):
        """
        Performs a Depth-First-Search of this graph.
        Each vertex in the graph is visited in a
        depth first manner starting from the given vertex.

        @param visitor: The visitor being applied to each vertex.
        @type: L{Visitor}
        @param start: The vertex from where the search begins.
        @type: C{int}
        """
        assert isinstance(visitor, Visitor)
        number_of_vertices = self.get_number_of_vertices()
        visited = ArrayList(number_of_vertices)
        for vertex in xrange(number_of_vertices):
            visited[vertex] = False

        visitor.visit(self[start])
        visited[self[start].get_vertex_number()] = True
        stack = Stack()
        stack.push(self[start])
        while not stack.is_empty() and not visitor.is_done():
            vertex = stack.pop()
            for successor in vertex.get_successors():
                if not visited[successor.get_vertex_number()]:
                    visitor.visit(successor)
                    visited[successor.get_vertex_number()] = True
                    stack.push(successor)

    def classify_edges(self):
        """
        Performs a classification of the edges contained
        in the graph.
        """
        if self.is_directed():
            return DFSEdgeClassification(self).dfs_recursive_directed()
        else:
            return DFSEdgeClassification(self).dfs_recursive_undirected()

class DirectedGraph(Graph):

    """
    The interface of a directed graph data structure.
    """

    def __init__(self, size):
        """
        Constructs a directed graph with the number of vertices
        specified by the size parameter.

        @param size: The number of vertices contained in this directed graph.
        @type: C{int}
        """
        super(DirectedGraph, self).__init__(size)

    def __eq__(self, other):
        """
        Compares two directed graphs for equality. The comparison
        is done by comparing the attributes: size, vertices,
        and adjacency_list of the respective directed graphs.

        @param other: The other directed graph.
        @type other: L{DirectedGraph}
        @return: True if the directed graphs are equal, false otherwise.
        @rtype: C{bool}
        """
        if isinstance(other, DirectedGraph):
            return self.is_equal(other)

    def __ne__(self, other):
        """
        Compares two directed graphs for inequality. The comparison
        is done by comparing the attributes: size, vertices,
        and adjacency_list of the respective directed graphs.

        @param other: The other directed graph.
        @type other: L{DirectedGraph}
        @return: True if the directed graphs are inequal, false otherwise.
        @rtype: C{bool}
        """
        return not self == other

    def __copy__(self):
        """
        Performs a shallow copy of this directed graph. The copy is
        done by copying the individual fields of this directed graph
        into a new directed graph object.

        @return: A shallow copy of this directed graph.
        @rtype: L{DirectedGraph}
        """
        return self.do_copy(DirectedGraph(self.size))

    @abstractmethod
    def is_weighted(self):
        """
        Returns if this directed graph is weighted.

        @return: True if the graph is weighted, false otherwise.
        @rtype: C{bool}
        """
        pass

    def is_directed(self):
        """
        Returns if this graph is directed.
        Note: This method does always return True.

        @return: True, since the graph is directed.
        @rtype: C{bool}
        """
        return True

    def is_strongly_connected(self):
        """
        Returns if this directed graph is strongly
        connected.

        @return: True, if the directed graph is strongly connected.
        @rtype: C{bool}
        """
        for vertex in self.get_vertices():
            visitor = VertexVisitor()
            visited = []
            self.depth_first_traversal(visitor, vertex.get_vertex_number())
            visited = visitor.get_visited()
            if len(visited) != self.get_number_of_vertices():
                return False
        return True

    def topological_order_traversal(self, visitor):
        """
        Performs a topological order traversal of the
        vertices in this directed graph. For each visited
        vertex the specified visitor is applied.

        Informally, a topological sort of a directed graph
        is a list of the vertices in which all successors
        of any given vertex appear in the sequence after
        that vertex. A topological sort, if it exists, of
        a directed graph is not unique.

        @param visitor: The visitor being applied to each vertex being visited.
        @type: L{Visitor}
        """
        number_of_vertices = self.get_number_of_vertices()
        in_degree = ArrayList(number_of_vertices)
        for vertex in xrange(number_of_vertices):
            in_degree[vertex] = 0
        for edge in self.get_edges():
            in_degree[edge.tail_vertex.get_vertex_number()] += 1
        queue = Queue()
        for vertex in xrange(number_of_vertices):
            if in_degree[vertex] == 0:
                queue.enqueue(self[vertex])
        while not queue.is_empty():
            vertex = queue.dequeue()
            visitor.visit(vertex)
            for successor in vertex.get_successors():
                in_degree[successor.get_vertex_number()] -= 1
                if in_degree[successor.get_vertex_number()] == 0:
                    queue.enqueue(successor)

    def is_cyclic(self):
        """
        Returns if this directed graph contains one or more cycles.

        @return: True, if the graph has one or more cycles, otherwise False.
        @rtype: C{bool}
        """
        visitor = VertexVisitor()
        self.topological_order_traversal(visitor)
        return len(visitor.get_visited()) != self.get_number_of_vertices()

    def add_edge(self, u, v, weight=0):
        """
        Adds an directed edge to this graph.

        @param u: The first vertex connected to second vertex by the directed edge.
        @type: L{DirecteGraphEdge}
        @param v: The second vertex connected to first vertex by the directed edge.
        @type: L{DirecteGraphEdge}
        """
        head_vertex_index = u.get_vertex_number()
        self.adjacency_list[head_vertex_index].append(DirectedGraphEdge(self, u, v))

class DirectedUnWeightedGraph(DirectedGraph):

    """
    The interface of a directed unweighted graph data structure.
    """

    def __init__(self, size):
        """
        Constructs a directed unweighted graph with the number
        of vertices specified by the size parameter.

        @param size: The number of vertices contained in this directed unweighted graph.
        @type: C{int}
        """
        super(DirectedUnWeightedGraph, self).__init__(size)

    def __eq__(self, other):
        """
        Compares two directed unweighted graphs for equality.
        The comparison is done by comparing the attributes:
        size, vertices, and adjacency_list of the respective
        directed unweighted graphs.

        @param other: The other directed unweighted graph.
        @type other: L{DirectedUnWeightedGraph}
        @return: True if the directed unweighted graphs are equal, false otherwise.
        @rtype: C{bool}
        """
        if isinstance(other, DirectedUnWeightedGraph):
            return self.is_equal(other)

    def __ne__(self, other):
        """
        Compares two directed unweighted graphs for inequality.
        The comparison is done by comparing the attributes:
        size, vertices, and adjacency_list of the respective
        directed unweighted graphs.

        @param other: The other directed unweighted graph.
        @type other: L{DirectedUnWeightedGraph}
        @return: True if the directed unweighted graphs are inequal, false otherwise.
        @rtype: C{bool}
        """
        return not self == other

    def __copy__(self):
        """
        Performs a shallow copy of this directed unweighted graph.
        The copy is made by copying the individual fields of this
        directed unweighted graph into a new directed
        unweighted graph object.

        @return: A shallow copy of this directed unweighted graph.
        @rtype: L{DirectedUnWeightedGraph}
        """
        return self.do_copy(DirectedUnWeightedGraph(self.size))

    def is_weighted(self):
        """
        Returns if this graph is weighted
        Note: This method does always return False.

        @return: False, since the graph is unweighted.
        @rtype: C{bool}
        """
        return False

    def add_edge(self, u, v, weight=0):
        """
        Adds an directed unweighted edge to this graph.

        @param u: The first vertex connected to second vertex by the directed unweighted edge.
        @type: L{DirecteUnWeightedGraphEdge}
        @param v: The second vertex connected to first vertex by the directed unweighted edge.
        @type: L{DirecteUnWeightedGraphEdge}
        """
        head_vertex_index = u.get_vertex_number()
        self.adjacency_list[head_vertex_index].append(DirectedUnWeightedGraphEdge(self, u, v))

class DirectedWeightedGraph(DirectedGraph):

    """
    The interface of a directed weighted graph data structure.
    """

    def __init__(self, size):
        """
        Constructs a directed weighted graph with the number
        of vertices specified by the size parameter.

        @param size: The number of vertices contained in this directed weighted graph.
        @type: C{int}
        """
        super(DirectedWeightedGraph, self).__init__(size)

    def __eq__(self, other):
        """
        Compares two directed weighted graphs for equality.
        The comparison is done by comparing the attributes:
        size, vertices, and adjacency_list of the respective
        directed weighted graphs.

        @param other: The other directed weighted graph.
        @type other: L{DirectedWeightedGraph}
        @return: True if the directed weighted graphs are equal, false otherwise.
        @rtype: C{bool}
        """
        if isinstance(other, DirectedWeightedGraph):
            return self.is_equal(other)

    def __ne__(self, other):
        """
        Compares two directed weighted graphs for inequality.
        The comparison is done by comparing the attributes:
        size, vertices, and adjacency_list of the respective
        directed weighted graphs.

        @param other: The other directed weighted graph.
        @type other: L{DirectedWeightedGraph}
        @return: True if the directed weighted graphs are inequal, false otherwise.
        @rtype: C{bool}
        """
        return not self == other

    def __copy__(self):
        """
        Performs a shallow copy of this directed weighted graph.
        The copy is made by copying the individual fields of this
        directed weighted graph into a new directed weighted graph
        object.

        @return: A shallow copy of this directed weighted graph.
        @rtype: L{DirectedWeightedGraph}
        """
        return self.do_copy(DirectedWeightedGraph(self.size))

    def is_weighted(self):
        """
        Returns if this graph is weighted
        Note: This method does always return True.

        @return: True, since the graph is weighted.
        @rtype: C{bool}
        """
        return True

    def add_edge(self, u, v, weight=0):
        """
        Adds an directed weighted edge to this graph.

        @param u: The first vertex connected to second vertex by the directed weighted edge.
        @type: L{DirecteWeightedGraphEdge}
        @param v: The second vertex connected to first vertex by the directed weighted edge.
        @type: L{DirecteWeightedGraphEdge}
        @param weight: The weight of the undirected edge.
        @type: C{int}
        """
        head_vertex_index = u.get_vertex_number()
        self.adjacency_list[head_vertex_index].append(DirectedWeightedGraphEdge(self, u, v, weight))

class UnDirectedGraph(Graph):

    """
    The interface of an undirected graph data structure.
    """

    def __init__(self, size):
        """
        Constructs an undirected graph with the number of vertices
        specified by the size parameter.

        @param size: The number of vertices contained in this undirected graph.
        @type: C{int}
        """
        super(UnDirectedGraph, self).__init__(size)

    def __eq__(self, other):
        """
        Compares two undirected graphs for equality. The
        comparison is done by comparing the attributes:
        size, vertices, and adjacency_list of the respective
        undirected graphs.

        @param other: The other undirected graph.
        @type other: L{UnDirectedGraph}
        @return: True if the undirected graphs are equal, false otherwise.
        @rtype: C{bool}
        """
        if isinstance(other, UnDirectedGraph):
            return self.is_equal(other)

    def __ne__(self, other):
        """
        Compares two undirected graphs for inequality. The
        comparison is done by comparing the attributes:
        size, vertices, and adjacency_list of the respective
        undirected graphs.

        @param other: The other undirected graph.
        @type other: L{UnDirectedGraph}
        @return: True if the undirected graphs are inequal, false otherwise.
        @rtype: C{bool}
        """
        return not self == other

    def __copy__(self):
        """
        Performs a shallow copy of this undirected graph. The copy
        is made by copying the individual fields of this undirected
        graph into a new undirected graph object.

        @return: A shallow copy of this undirected graph.
        @rtype: L{UnDirectedGraph}
        """
        return self.do_copy(UnDirectedGraph(self.size))

    @abstractmethod
    def is_weighted(self):
        """
        Returns if this undirected graph is weighted.

        @return: True if the graph is weighted, false otherwise.
        @rtype: C{bool}
        """
        pass

    def is_directed(self):
        """
        Returns if this graph is directed.
        Note: This method does always return False.

        @return: False, since the graph is undirected.
        @rtype: C{bool}
        """
        return False

    def is_connected(self):
        """
        Returns if this graph is connected. That is, if every
        pair of vertices in the graph is connected.

        @return: True if the graph is connected, false otherwise.
        @rtype: C{bool}
        """
        visitor = VertexVisitor()
        self.depth_first_traversal(visitor, 0)
        visited = []
        visited = visitor.get_visited()
        return len(visited) == self.get_number_of_vertices()

    def is_cyclic(self):
        """
        Returns if this undirected graph contains any cycles.

        An undirected graph G = (V, E) will have at most
        |V - 1| edges for it to be acyclic. In this situation
        the graph is a tree - more precisly the minimum spanning
        tree, which does not contain cycles. Therefore, counting
        the number of edges |E|, by performing a depth-first
        traversal from an abitrary vertex and comparing the result
        with |V - 1|, will determine if the undirected graph contains
        cycles. If the number of edges |E| is greather than
        |V - 1| the graph has at least one cycle. Otherwise,
        the undirected graph has no cycles and is acyclic.

        @return: True, if the undirected graph has one or cycles, otherwise False.
        @rtype: C{bool}
        """
        result = self.classify_edges()
        return result.has_back_edges()

    def add_edge(self, u, v, weight=0):
        """
        Adds an undirected edge to this graph.

        @param u: The first vertex connected to the second vertex by the undirected edge.
        @type: L{GraphVertex}
        @param v: The second vertex connected to the first vertex by the undirected edge.
        @type: L{GraphVertex}
        """
        head_vertex_index = u.get_vertex_number()
        tail_vertex_index = v.get_vertex_number()
        self.adjacency_list[head_vertex_index].append(UnDirectedGraphEdge(self, u, v))
        self.adjacency_list[tail_vertex_index].append(UnDirectedGraphEdge(self, v, u))

class UnDirectedUnWeightedGraph(UnDirectedGraph):

    """
    The interface of an undirected unweighted graph data structure.
    """

    def __init__(self, size):
        """
        Constructs an undirected unweighted graph with the number
        of vertices specified by the size parameter.

        @param size: The number of vertices contained in this directed unweighted graph.
        @type: C{int}
        """
        super(UnDirectedUnWeightedGraph, self).__init__(size)

    def __eq__(self, other):
        """
        Compares two undirected unweighted graphs for equality.
        The comparison is done by comparing the attributes:
        size, vertices, and adjacency_list of the respective
        undirected unweighted graphs.

        @param other: The other undirected unweighted graph.
        @type other: L{UnDirectedUnWeightedGraph}
        @return: True if the undirected unweighted graphs are equal, false otherwise.
        @rtype: C{bool}
        """
        if isinstance(other, UnDirectedUnWeightedGraph):
            return self.is_equal(other)
        return NotImplemented

    def __ne__(self, other):
        """
        Compares two undirected unweighted graphs for inequality.
        The comparison is done by comparing the attributes:
        size, vertices, and adjacency_list of the respective
        undirected unweighted graphs.

        @param other: The other undirected unweighted graph.
        @type other: L{UnDirectedUnWeightedGraph}
        @return: True if the undirected unweighted graphs are inequal, false otherwise.
        @rtype: C{bool}
        """
        return not self == other

    def __copy__(self):
        """
        Performs a shallow copy of this undirected unweighted graph.
        The copy is made by copying the individual fields of this
        undirected unweighted graph into a new undirected unweighted
        graph object.

        @return: A shallow copy of this undirected unweighted graph.
        @rtype: L{UnDirectedUnWeightedGraph}
        """
        return self.do_copy(UnDirectedUnWeightedGraph(self.size))

    def is_weighted(self):
        """
        Returns if this graph is weighted
        Note: This method does always return False.

        @return: False, since the graph is unweighted.
        @rtype: C{bool}
        """
        return False

    def add_edge(self, u, v, weight=0):
        """
        Adds an undirected unweighted edge to this graph.

        @param u: The first vertex connected to second vertex by the undirected unweighted edge.
        @type: L{UnDirectedUnWeightedGraphEdge}
        @param v: The second vertex connected to first vertex by the undirected unweighted edge.
        @type: L{UnDirectedUnWeightedGraphEdge}
        """
        head_vertex_index = u.get_vertex_number()
        tail_vertex_index = v.get_vertex_number()
        self.adjacency_list[head_vertex_index].append(UnDirectedUnWeightedGraphEdge(self, u, v))
        self.adjacency_list[tail_vertex_index].append(UnDirectedUnWeightedGraphEdge(self, v, u))

class UnDirectedWeightedGraph(UnDirectedGraph):

    """
    The interface of an undirected weighted graph data structure.
    """

    def __init__(self, size):
        """
        Constructs an undirected weighted graph with the number
        of vertices specified by the size parameter.

        @param size: The number of vertices contained in this undirected weighted graph.
        @type: C{int}
        """
        super(UnDirectedWeightedGraph, self).__init__(size)

    def __eq__(self, other):
        """
        Compares two undirected weighted graphs for equality.
        The comparison is done by comparing the attributes:
        size, vertices, and adjacency_list of the respective
        undirected weighted graphs.

        @param other: The other undirected weighted graph.
        @type other: L{UnDirectedWeightedGraph}
        @return: True if the undirected weighted graphs are equal, false otherwise.
        @rtype: C{bool}
        """
        if isinstance(other, UnDirectedWeightedGraph):
            return self.is_equal(other)

    def __ne__(self, other):
        """
        Compares two undirected weighted graphs for inequality.
        The comparison is done by comparing the attributes:
        size, vertices, and adjacency_list of the respective
        undirected weighted graphs.

        @param other: The other undirected weighted graph.
        @type other: L{UnDirectedWeightedGraph}
        @return: True if the undirected weighted graphs are inequal, false otherwise.
        @rtype: C{bool}
        """
        return not self == other

    def __copy__(self):
        """
        Performs a shallow copy of this undirected weighted graph.
        The copy is made by copying the individual fields of this
        undirected weighted graph into a new undirected weighted
        graph object.

        @return: A shallow copy of this undirected weighted graph.
        @rtype: L{UnDirectedWeightedGraph}
        """
        return self.do_copy(UnDirectedWeightedGraph(self.size))

    def is_weighted(self):
        """
        Returns if this graph is weighted
        Note: This method does always return True.

        @return: True, since the graph is weighted.
        @rtype: C{bool}
        """
        return True

    def add_edge(self, u, v, weight=0):
        """
        Adds an undirected weighted edge to this graph.

        @param u: The first vertex connected to second vertex by the undirected weighted edge.
        @type: L{UnDirectedWeightedGraphEdge}
        @param v: The second vertex connected to first vertex by the undirected weighted edge.
        @type: L{UnDirectedWeightedGraphEdge}
        @param weight: The weight of the undirected edge.
        @type: C{int}
        """
        head_vertex_index = u.get_vertex_number()
        tail_vertex_index = v.get_vertex_number()
        edge_uv = UnDirectedWeightedGraphEdge(self, u, v, weight)
        edge_vu = UnDirectedWeightedGraphEdge(self, v, u, weight)
        self.adjacency_list[head_vertex_index].append(edge_uv)
        self.adjacency_list[tail_vertex_index].append(edge_vu)


