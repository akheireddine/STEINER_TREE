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
Provides functionality to perform edge classification
of directed -and undirected graphs. The edges are
traversed in a depth-first search and classified into
four edge types: tree edge, back edge, forward edge,
or cross edge. During execution of the depth-first
search, the classifiction of edge (u, v), the edge
from vertex u to vertex v, depends on whether v has
been visited before during the depth-first search and
if so, the relationship between u and v.

NOTE:
Edge classification in directed graphs is different
from edge classification in undirected graphs. In a
directed graph an edge is classified into one of the
the four types mentioned above. An undirected graph
cannot contain forward edges and cross edges, since
in those cases, the edge (u, v) would already have
been traversed during depth-first search before
vertex u is reached and depth-first search tries to
visit vertex v.
"""

__author__ = "Brian Horn"
__copyright__ = "Copyright (c) 2015 Brian Horn"
__credits__ = "Brian Horn"
__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Brian Horn"
__email__ = "trycatchhorn@gmail.com"
__status__ = "Prototype"

from py_alg_dat.graph_edge import EdgeClassification

class DFSEdgeClassification(object):

    """
    The interface for edge classificaton of graphs.
    """

    def __init__(self, graph):
        """
        Constructs an object used to hold the result of
        performing edge classification on a graph.

        @param graph: The graph where edge classification is performed.
        @type graph: L{Graph}
        """
        self.graph = graph
        self.classification = DFSResult()

    def clear(self):
        """
        Clears the classification by re-initializing each attribute
        contained in this edge classification.
        """
        self.classification.clear()

    def get_classification(self):
        """
        Returns this edge classification.

        @return: The edge classification.
        @rtype: L{DFSResult}
        """
        return self.classification

    def get_parent(self):
        """
        Returns the dictionary holding the parent
        vertices of this edge classification.

        @return: The parent dictionary of the classification.
        @rtype: C{dictionary}
        """
        return self.classification.get_parent()

    def get_discovery_time(self):
        """
        Returns the dictionary holding the discovery time
        of this edge classification.

        @return: The discovery time dictionary of the classification.
        @rtype: C{dictionary}
        """
        return self.classification.get_discovery_time()

    def get_finishing_time(self):
        """
        Returns the dictionary holding the finishing time
        of this edge classification.

        @return: The finishing time dictionary of the classification.
        @rtype: C{dictionary}
        """
        return self.classification.get_finishing_time()

    def get_edges(self):
        """
        Returns the dictionary holding edges of this
        edge classification.

        @return: The edge dictionary of the classification.
        @rtype: C{dictionary}
        """
        return self.classification.get_edges()

    def get_order(self):
        """
        Returns the list of vertices visited during the
        dfs edge classification. The list is arranged in
        a manner of when a specific vertex was being
        visited.

        @return: The order list of the classification.
        @rtype: C{list}
        """
        return self.classification.get_order()

    def get_parent_of_vertex(self, vertex):
        """
        Returns the parent vertex of the specified
        vertex in this edge classification.

        @param vertex: The vertex which parent should be found.
        @type vertex: L{GraphVertex}
        @return: The parent vertex of the specified vertex.
        @rtype: L{GraphVertex}
        """
        return self.classification.get_parent_of_vertex(vertex)

    def get_discovery_time_of_vertex(self, vertex):
        """
        Returns the discovery time of the specified
        vertex in this edge classification.

        @param vertex: The vertex which discovery time should be found.
        @type vertex: L{GraphVertex}
        @return: The discovery time of the specified vertex.
        @rtype: L{GraphVertex}
        """
        return self.classification.get_discovery_time_of_vertex(vertex)

    def get_finishing_time_of_vertex(self, vertex):
        """
        Returns the finishing time of the specified
        vertex in this edge classification.

        @param vertex: The vertex which finishing time should be found.
        @type vertex: L{GraphVertex}
        @return: The finishing time of the specified vertex.
        @rtype: L{GraphVertex}
        """
        return self.classification.get_finishing_time_of_vertex(vertex)

    def get_tree_edges(self):
        """
        Returns the tree edges contained in this edge
        classification. The tree edges are returned in
        a set.

        @return: The tree edges in the classification.
        @rtype: C{set}
        """
        return self.classification.get_tree_edges()

    def get_back_edges(self):
        """
        Returns the back edges contained in this edge
        classification. The back edges are returned in
        a set.

        @return: The back edges in the classification.
        @rtype: C{set}
        """
        return self.classification.get_back_edges()

    def get_forward_edges(self):
        """
        Returns the forward edges contained in this edge
        classification. The forward edges are returned in
        a set.

        @return: The forward edges in the classification.
        @rtype: C{set}
        """
        return self.classification.get_forward_edges()

    def get_cross_edges(self):
        """
        Returns the cross edges contained in this edge
        classification. The cross edges are returned in
        a set.

        @return: The cross edges in the classification.
        @rtype: C{set}
        """
        return self.classification.get_cross_edges()

    def get_number_of_tree_edges(self):
        """
        Returns the number of tree edges contained in this
        edge classification.

        @return: The number of tree edges in the classification.
        @rtype: C{int}
        """
        return self.classification.get_number_of_tree_edges()

    def get_number_of_back_edges(self):
        """
        Returns the number of back edges contained in this
        edge classification.

        @return: The number of back edges in the classification.
        @rtype: C{int}
        """
        return self.classification.get_number_of_back_edges()

    def get_number_of_forward_edges(self):
        """
        Returns the number of forward edges contained in this
        edge classification.

        @return: The number of forward edges in the classification.
        @rtype: C{int}
        """
        return self.classification.get_number_of_forward_edges()

    def get_number_of_cross_edges(self):
        """
        Returns the number of cross edges contained in this
        edge classification.

        @return: The number of cross edges in the classification.
        @rtype: C{int}
        """
        return self.classification.get_number_of_cross_edges()

    def has_tree_edges(self):
        """
        Returns if this edge classification contains any
        tree edges.

        @return: True if the edge classification has any tree edges, false otherwise.
        @rtype: C{bool}
        """
        return self.classification.has_tree_edges()

    def has_back_edges(self):
        """
        Returns if this edge classification contains any
        back edges.

        @return: True if the edge classification has any back edges, false otherwise.
        @rtype: C{bool}
        """
        return self.classification.has_back_edges()

    def has_forward_edges(self):
        """
        Returns if this edge classification contains any
        forward edges.

        @return: True if the edge classification has any forward edges, false otherwise.
        @rtype: C{bool}
        """
        return self.classification.has_forward_edges()

    def has_cross_edges(self):
        """
        Returns if this edge classification contains any
        cross edges.

        @return: True if the edge classification has any cross edges, false otherwise.
        @rtype: C{bool}
        """
        return self.classification.has_cross_edges()

    def dfs_recursive_directed(self):
        """
        Performs a recursive depth-first traversal of the
        directed graph in this edge classification. During
        the traversal all edges are visited -and classified
        as being either a tree edge, back edge, forward edge,
        or cross edge.
        """
        for vertex in self.graph.get_vertices():
            if vertex not in self.classification.parent:
                self.dfs_visit_recursive_directed(vertex, self.classification)
        return self.classification

    def dfs_visit_recursive_directed(self, vertex_u, results, parent=None):
        """
        Classifies the edges in the directed graph by recursively
        traversing the edges.

        1) If v is visited for the first time as edge (u, v) is
        traversed, then the edge is a tree edge.

        2) Else, v has already been visited:

        a) If v is an ancestor of u, then edge (u, v) is a
        back edge.

        b) Else, if v is a descendant of u, then edge (u, v) is
        a forward edge.

        c) Else, if v is neither an ancestor or descendant of u,
        then edge (u, v) is a cross edge.

        @param u: The vertex from where the dfs traversal begins.
        @type u: L{GraphVertex}
        @param result: The result of the dfs classification.
        @type result: L{DFSResult}
        @param parent: The parent vertex - defaults to None.
        @type parent: L{GraphVertex}
        """
        results.parent[vertex_u] = parent
        results.time += 1
        results.discovery_time[vertex_u] = results.time

        if parent:
            edge_parent_u = self.graph.get_edge(parent, vertex_u)
            results.edges[edge_parent_u] = EdgeClassification.TREE_EDGE

        for vertex_v in vertex_u.get_successors():
            edge_u_v = self.graph.get_edge(vertex_u, vertex_v)
            if vertex_v not in results.parent:
                self.dfs_visit_recursive_directed(vertex_v, results, vertex_u)
            elif vertex_v not in results.finishing_time:
                results.edges[edge_u_v] = EdgeClassification.BACK_EDGE
            elif results.discovery_time[vertex_u] < results.discovery_time[vertex_v]:
                results.edges[edge_u_v] = EdgeClassification.FORWARD_EDGE
            else:
                results.edges[edge_u_v] = EdgeClassification.CROSS_EDGE
        results.time += 1
        results.finishing_time[vertex_u] = results.time
        results.order.append(vertex_u)

    def dfs_recursive_undirected(self):
        """
        Performs a recursive depth-first traversal of the
        undirected graph in this edge classification. During
        the traversal all edges are visited -and classified
        as being either a tree edge or back edge

        NOTE: An undirected graph cannot contain forward edges
        or cross edges.
        """
        for vertex in self.graph.get_vertices():
            if vertex not in self.classification.parent:
                self.dfs_visit_recursive_undirected(vertex, self.classification)
        return self.classification

    def dfs_visit_recursive_undirected(self, vertex_u, results, parent=None):
        """
        Classifies the edges in the undirected graph by recursively
        traversing the edges.

        1) If v is visited for the first time as edge (u, v) is
        traversed, then the edge is a tree edge.

        2) Else, v has already been visited:

        a) If v is an ancestor of u, then edge (u, v) is a
        back edge.

        @param u: The vertex from where the dfs traversal begins.
        @type u: L{GraphVertex}
        @param result: The result of the dfs classification.
        @type result: L{DFSResult}
        @param parent: The parent vertex - defaults to None.
        @type parent: L{GraphVertex}
        """
        results.parent[vertex_u] = parent
        results.time += 1
        results.discovery_time[vertex_u] = results.time

        # Only classify edge (parent, u) if the graph is undirected,
        # since edge (u, parent) has the same classification in an
        # undirected graph.
        if parent:
            edge_parent_u = self.graph.get_edge(parent, vertex_u)
            edge_u_parent = self.graph.get_edge(vertex_u, parent)

        if parent and not edge_parent_u in results.edges and not edge_u_parent in results.edges:
            results.edges[edge_parent_u] = EdgeClassification.TREE_EDGE
        for vertex_v in vertex_u.get_successors():

            # Only classify edge (u, v) if the graph is undirected,
            # since edge (v, u) has the same classification in an
            # undirected graph.
            if vertex_u is not None and vertex_v is not None:
                edge_u_v = self.graph.get_edge(vertex_u, vertex_v)
                edge_v_u = self.graph.get_edge(vertex_v, vertex_u)

            if not edge_u_v in results.edges and not edge_v_u in results.edges:
                if vertex_v not in results.parent:
                    self.dfs_visit_recursive_undirected(vertex_v, results, vertex_u)
                elif vertex_v not in results.finishing_time:
                    results.edges[edge_u_v] = EdgeClassification.BACK_EDGE
        results.time += 1
        results.finishing_time[vertex_u] = results.time
        results.order.append(vertex_u)

class DFSResult(object):

    """
    Data structure holding the result of performing an edge
    classification on a graph.
    """

    def __init__(self):
        """
        Constructs an object used to hold the result of
        performing edge classification on a graph.
        """
        self.parent = {}
        self.discovery_time = {}
        self.finishing_time = {}
        self.edges = {}
        self.order = []
        self.time = 0

    def clear(self):
        """
        Clears the classification by re-initializing each attribute
        contained in this edge classification.
        """
        self.parent = {}
        self.discovery_time = {}
        self.finishing_time = {}
        self.edges = {}
        self.order = []
        self.time = 0

    def get_parent(self):
        """
        Returns the dictionary holding the parent
        vertices of this edge classification.

        @return: The parent dictionary of the classification.
        @rtype: C{dictionary}
        """
        return self.parent

    def get_discovery_time(self):
        """
        Returns the dictionary holding the discovery time
        of this edge classification.

        @return: The discovery time dictionary of the classification.
        @rtype: C{dictionary}
        """
        return self.discovery_time

    def get_finishing_time(self):
        """
        Returns the dictionary holding the finishing time
        of this edge classification.

        @return: The finishing time dictionary of the classification.
        @rtype: C{dictionary}
        """
        return self.finishing_time

    def get_edges(self):
        """
        Returns the dictionary holding edges of this
        edge classification.

        @return: The edge dictionary of the classification.
        @rtype: C{dictionary}
        """
        return self.edges

    def get_order(self):
        """
        Returns the list of vertices visited during the
        dfs edge classification. The list is arranged in
        a manner of when a specific vertex was being
        visited.

        @return: The order list of the classification.
        @rtype: C{list}
        """
        return self.order

    def get_parent_of_vertex(self, vertex):
        """
        Returns the parent vertex of the specified
        vertex in this edge classification.

        @param vertex: The vertex which parent should be found.
        @type vertex: L{GraphVertex}
        @return: The parent vertex of the specified vertex.
        @rtype: L{GraphVertex}
        """
        return self.parent[vertex]

    def get_discovery_time_of_vertex(self, vertex):
        """
        Returns the discovery time of the specified
        vertex in this edge classification.

        @param vertex: The vertex which discovery time should be found.
        @type vertex: L{GraphVertex}
        @return: The discovery time of the specified vertex.
        @rtype: L{GraphVertex}
        """
        return self.discovery_time[vertex]

    def get_finishing_time_of_vertex(self, vertex):
        """
        Returns the finishing time of the specified
        vertex in this edge classification.

        @param vertex: The vertex which finishing time should be found.
        @type vertex: L{GraphVertex}
        @return: The finishing time of the specified vertex.
        @rtype: L{GraphVertex}
        """
        return self.finishing_time[vertex]

    def get_tree_edges(self):
        """
        Returns the tree edges contained in this edge
        classification. The tree edges are returned in
        a set.

        @return: The tree edges in the classification.
        @rtype: C{set}
        """
        result = set()
        for key, value in self.edges.items():
            if value == EdgeClassification.TREE_EDGE:
                result.add(key)
        return result

    def get_back_edges(self):
        """
        Returns the back edges contained in this edge
        classification. The back edges are returned in
        a set.

        @return: The back edges in the classification.
        @rtype: C{set}
        """
        result = set()
        for key, value in self.edges.items():
            if value == EdgeClassification.BACK_EDGE:
                result.add(key)
        return result

    def get_forward_edges(self):
        """
        Returns the forward edges contained in this edge
        classification. The forward edges are returned in
        a set.

        @return: The forward edges in the classification.
        @rtype: C{set}
        """
        result = set()
        for key, value in self.edges.items():
            if value == EdgeClassification.FORWARD_EDGE:
                result.add(key)
        return result

    def get_cross_edges(self):
        """
        Returns the cross edges contained in this edge
        classification. The cross edges are returned in
        a set.

        @return: The cross edges in the classification.
        @rtype: C{set}
        """
        result = set()
        for key, value in self.edges.items():
            if value == EdgeClassification.CROSS_EDGE:
                result.add(key)
        return result

    def get_number_of_tree_edges(self):
        """
        Returns the number of tree edges contained in this
        edge classification.

        @return: The number of tree edges in the classification.
        @rtype: C{int}
        """
        return len(self.get_tree_edges())

    def get_number_of_back_edges(self):
        """
        Returns the number of back edges contained in this
        edge classification.

        @return: The number of back edges in the classification.
        @rtype: C{int}
        """
        return len(self.get_back_edges())

    def get_number_of_forward_edges(self):
        """
        Returns the number of forward edges contained in this
        edge classification.

        @return: The number of forward edges in the classification.
        @rtype: C{int}
        """
        return len(self.get_forward_edges())

    def get_number_of_cross_edges(self):
        """
        Returns the number of cross edges contained in this
        edge classification.

        @return: The number of cross edges in the classification.
        @rtype: C{int}
        """
        return len(self.get_cross_edges())

    def has_tree_edges(self):
        """
        Returns if this edge classification contains any
        tree edges.

        @return: True if the edge classification has any tree edges, false otherwise.
        @rtype: C{bool}
        """
        return self.get_number_of_tree_edges() > 0

    def has_back_edges(self):
        """
        Returns if this edge classification contains any
        back edges.

        @return: True if the edge classification has any back edges, false otherwise.
        @rtype: C{bool}
        """
        return self.get_number_of_back_edges() > 0

    def has_forward_edges(self):
        """
        Returns if this edge classification contains any
        forward edges.

        @return: True if the edge classification has any forward edges, false otherwise.
        @rtype: C{bool}
        """
        return self.get_number_of_forward_edges() > 0

    def has_cross_edges(self):
        """
        Returns if this edge classification contains any
        cross edges.

        @return: True if the edge classification has any cross edges, false otherwise.
        @rtype: C{bool}
        """
        return self.get_number_of_cross_edges() > 0


