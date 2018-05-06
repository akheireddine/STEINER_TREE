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
Provides various graph algorithms.
"""

__author__ = "Brian Horn"
__copyright__ = "Copyright (c) 2015 Brian Horn"
__credits__ = "Brian Horn"
__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Brian Horn"
__email__ = "trycatchhorn@gmail.com"
__status__ = "Prototype"

from py_alg_dat.array_list import ArrayList
from py_alg_dat.association import Association
from py_alg_dat.entry import Entry
from py_alg_dat.graph_edge import UnDirectedWeightedGraphEdge
from py_alg_dat.graph_path import GraphPath
from py_alg_dat.min_heap import MinHeap
from py_alg_dat.minimum_spanning_tree import MinimumSpanningTree
from py_alg_dat.partition import Partition

class GraphAlgorithms(object):

    """
    The interface for graph algorithms.
    """

    @staticmethod
    def prims_algorithm(graph, start):
        """
        Implements Prim's algorithm used to find the minimum spanning
        tree for the specified connected weighted undirected graph. The
        implementation uses a minimum priority queue, implemented as a
        min-heap, in order to store the next edge to explore.

        Initially, a table for all vertices in the graph is constructed.
        Each entry contains a record of whether the vertex has been
        discovered, the distance to the vertex from the specified start
        vertex, the predecessor of the vertex, and the edge leading to
        the vertex. Before the traversal begins, the field specifying if
        the vertex has been discovered is set to false, the distance is
        set to infinity, which is similar to being unreachable, the
        predecessor is set to None, and the edge is set to None.
        Additionally, an association between the specified start vertex
        and its distance to itself, is added to the minimum priority queue.
        The distance from a vertex to itself is zero - therefore the
        association has the form: Association(0, start) in the beginning
        of the algorithm.

        The algorithm proceedes by extracting the vertex with the smallest
        distance from the start vertex, by extracting it from the min-heap.
        If the vertex has not previously be discovered it is marked as
        discovered and the emanating edges of the vertex is explored. If
        it is found that an edge leads to an undiscovered vertex with
        smaller distance, the table entry for this is vertex is updated
        accordingly.

        Time complexity: O(m log(n)), where m is the number of edges and
        n is the number of vertices.

        @param graph: The graph from where the minimum spanning tree is computed.
        @type: L{UnDirectedWeightedGraph}
        @param start: The vertex from where Prim's algorithm begins.
        @type: L{UnWeightedGraphVertex}
        @return: The minimum spanning tree for the graph.
        @rtype: L{MinimumSpanningTree}
        """
        number_of_vertices = graph.get_number_of_vertices()
        table = ArrayList(number_of_vertices)
        for i in xrange(number_of_vertices):
            table[i] = Entry()
        table[start.vertex_number].distance = 0
        queue = MinHeap()
        queue.insert(Association(0, start))
        while not queue.is_empty():
            vertex = queue.heap_extract_min().get_value()
            entry1 = vertex.vertex_number
            if not table[entry1].discovered:
                table[entry1].discovered = True
                for edge in vertex.get_emanating_edges():
                    vertex_mate = edge.get_mate(vertex)
                    entry2 = vertex_mate.vertex_number
                    if not table[entry2].discovered and table[entry2].distance > edge.get_weight():
                        table[entry2].distance = edge.get_weight()
                        table[entry2].predecessor = vertex
                        table[entry2].edge = edge
                        queue.insert(Association(edge.get_weight(), vertex_mate))

        mst = MinimumSpanningTree(graph)
        for i in xrange(number_of_vertices):
            if i != start.vertex_number:
                vertex_u = graph[i]
                vertex_v = table[i].predecessor
                weight = table[i].distance
                edge = UnDirectedWeightedGraphEdge(graph, vertex_u, vertex_v, weight)
                mst.add_edge(edge)
        return mst

    @staticmethod
    def kruskals_algorithm(graph):
        """
        Implements Kruskal's algorithm used to find the minimum spanning
        tree for the specified connected weighted undirected graph. The
        implementation uses a minimum priority queue, implemented as a
        min-heap, in order to store the next edge to explore. Initially,
        all edges in the graph are added to the min-heap as associations
        between the weight and the edge, with the weight acting as priority.
        Edges are extracted -and processed from the min-heap in order of
        their weight, where edges with minimum weight are processed first.
        The vertices connected by each edge is put into a disjoint set,
        implemented using Union-Find with rank and path compression.

        Time complexity: O(m log(n)), where m is the number of edges and
        n is the number of vertices.

        @param graph: The graph from where the minimum spanning tree is computed.
        @type: L{UnDirectedWeightedGraph}
        @return: The minimum spanning tree for the graph.
        @rtype: L{MinimumSpanningTree}
        """
        queue = MinHeap()
        for edge in graph.get_edges():
            weight = edge.get_weight()
            queue.insert(Association(weight, edge))

        partition = Partition()
        for vertex in graph.get_vertices():
            partition.make_set(vertex.vertex_number)
        mst = MinimumSpanningTree(graph)
        while not queue.is_empty():
            association = queue.heap_extract_min()
            edge = association.get_value()
            head_vertex_index = edge.head_vertex.vertex_number
            tail_vertex_index = edge.tail_vertex.vertex_number
            set1 = partition.find(head_vertex_index)
            set2 = partition.find(tail_vertex_index)

            if set1 != set2:
                partition.union(set1, set2)
                vertex_u = edge.head_vertex
                vertex_v = edge.tail_vertex
                weight = edge.get_weight()
                edge_v_u = UnDirectedWeightedGraphEdge(graph, vertex_u, vertex_v, weight)
                mst.add_edge(edge_v_u)
        return mst

    @staticmethod
    def dijkstras_algorithm(graph, source):
        """
        Implements Dijkstra's algorithm for finding the single shortest path
        for the specified source vertex to all other vertices in the directed
        weighted graph. The implementation uses a minimum priority queue,
        implemented as a min-heap, in order to store the next vertex to explore.
        Vertices are added to the min-heap with their currrent-know total
        distance - their distance acting as their "priority". The priority of
        a vertex can later be updated with a new total distance if a shorter path
        to the vertex is found.

        The result is returned in a table in the format:
        (row_0, column_0) = (vertex_number_0, Entry_0)
        (row_1, column_1) = (vertex_number_1, Entry_1)
        .
        ..
        ...
        (row_n-1, column_n-1) = (vertex_number_n-1, Entry_n-1)

        The Entry object contains the attributes:
        (discovered, distance, predecessor, edge)

        discovered: boolean flag indicating if the vertex with the spcified index has been visited.
        distance: the accumulated edge weight distance from the source vertex.
        predecessor: predecesser vertex to the vertex with index specified by vertex_number.
        edge: emanating edge of the vertex with the index specified by vertex_number.

        Time complexity: O(m log(n)), where m is the number of edges and
        n is the number of vertices.

        @param graph: The graph from where the shortest path is computed.
        @type: L{DirectedWeightedGraph}
        @param source: The vertex from where Prim's algorithm begins.
        @type: L{UnWeightedGraphVertex}
        @return: Table of entries giving the shortest path from source to all other vertices.
        @rtype: L{ArrayList}
        """
        number_of_vertices = graph.get_number_of_vertices()
        table = ArrayList(number_of_vertices)
        for i in xrange(number_of_vertices):
            table[i] = Entry()
        table[source.vertex_number].distance = 0
        queue = MinHeap()
        queue.insert(Association(0, source))
        while not queue.is_empty():
            association = queue.heap_extract_min()
            vertex_one = association.get_value()
            if not table[vertex_one.vertex_number].discovered:
                table[vertex_one.vertex_number].discovered = True
                for edge in vertex_one.get_emanating_edges():
                    vertex_two = edge.get_mate(vertex_one)
                    path_distance = table[vertex_one.vertex_number].distance + edge.get_weight()
                    if table[vertex_two.vertex_number].distance > path_distance:
                        table[vertex_two.vertex_number].distance = path_distance
                        table[vertex_two.vertex_number].predecessor = vertex_one
                        table[vertex_two.vertex_number].edge = edge
                        queue.insert(Association(path_distance, vertex_two))
        return table

    @staticmethod
    def shortest_path(graph, source, destination):
        """
        Implements the shortest path algorithm for the specified directed
        weighted graph by using Dijkstra's algorithm for finding the single
        shortest path for the specified source vertex to the specified
        destination vertex.

        The result is returned in a GraphPath object, which includes
        the list of vertices visited when going from the source vertex
        to the destination vertex along the shortest path. Additionally,
        the GraphPath contains the edges traversed along this path
        together with the total length of the path.

        Time complexity: O(m log(n)), where m is the number of edges and
        n is the number of vertices.

        @param graph: The graph from where the shortest path is computed.
        @type: L{DirectedWeightedGraph}
        @param source: The source vertex from where path is computed.
        @type: L{UnWeightedGraphVertex}
        @param destination: The destination vertex.
        @type: L{UnWeightedGraphVertex}
        @return: The path between source -and destination vertex.
        @rtype: L{GraphPath}
        """
        path = GraphPath(graph)
        if graph.has_vertex(source) and graph.has_vertex(destination):
            table = GraphAlgorithms.dijkstras_algorithm(graph, source)
            start = table[source.vertex_number]
            end = table[destination.vertex_number]
            path.add_vertex(graph[table.get_index(end)])
            while start != end:
                path.add_vertex(end.predecessor)
                path.add_edge(end.edge)
                end = table[end.predecessor.get_vertex_number()]
        return path

    @staticmethod
    def bellman_ford_algorithm(graph, source):
        """
        Implements the Bellman-Ford algorithm for finding single-source
        shortest paths in a directed graph. Like Dijkstra's algorithm,
        the Bellman-Ford algorithm computes the shortest path from a
        source vertex to all other vertices in the graph. However, unlike
        Dijkstra's algorithm, the Bellman-Ford algorithm works correctly in
        graphs containing negative weighted edges, as long as the graph does
        not contain a negative cycle. If the graph contains a negative cycle,
        the cost of a shortest path may not be well defined.

        The result is returned in a tuple, where the first element specifies
        whether or not the graph contains a cycle. The second element
        contains a dictionary, where the keys are the vertices -and the values
        are the weights in the shortest path from the source vertex to all
        other vertices in the graph.

        @param graph: The graph from where the shortest path is computed.
        @type: L{DirectedWeightedGraph}
        @param source: The vertex from where Bellman-Ford's algorithm begins.
        @type: L{UnWeightedGraphVertex}
        @return: Dictionary giving the shortest path from source to all other vertices.
        @rtype: C{Dictionary}
        """
        distances = {}
        has_cycle = False

        # Initialize every vertex distance to infinity
        # except the source vertex.
        for vertex in graph.get_vertices():
            if vertex == source:
                distances[vertex] = 0
            else:
                distances[vertex] = float('inf')

        for _ in xrange(1, graph.get_number_of_vertices()):
            for edge in graph.get_edges():
                vertex_u = edge.get_head_vertex()
                vertex_v = edge.get_tail_vertex()
                weight = edge.get_weight()
                if distances[vertex_v] > distances[vertex_u] + weight:
                    distances[vertex_v] = distances[vertex_u] + weight

        for edge in graph.get_edges():
            vertex_u = edge.get_head_vertex()
            vertex_v = edge.get_tail_vertex()
            weight = edge.get_weight()
            if distances[vertex_v] > distances[vertex_u] + weight:
                has_cycle = True

        return has_cycle, distances


