#!/usr/bin/env py.test

"""
Test directed unweighted graph.
"""

import unittest
import copy
from py_alg_dat import dfs_edge_classification
from py_alg_dat import graph
from py_alg_dat import graph_edge
from py_alg_dat import graph_vertex

class TestDirectedUnWeightedGraph(unittest.TestCase):

    """
    Test directed unweighted graph.
    """

    def setUp(self):
        # Create directed unweighted graph Cormen page 596.
        self.graph1 = graph.DirectedUnWeightedGraph(5)

        self.v1_g1 = graph_vertex.UnWeightedGraphVertex(self.graph1, 'S')
        self.v2_g1 = graph_vertex.UnWeightedGraphVertex(self.graph1, 'T')
        self.v3_g1 = graph_vertex.UnWeightedGraphVertex(self.graph1, 'X')
        self.v4_g1 = graph_vertex.UnWeightedGraphVertex(self.graph1, 'Y')
        self.v5_g1 = graph_vertex.UnWeightedGraphVertex(self.graph1, 'Z')

        self.graph1.add_vertex(self.v1_g1)
        self.graph1.add_vertex(self.v2_g1)
        self.graph1.add_vertex(self.v3_g1)
        self.graph1.add_vertex(self.v4_g1)
        self.graph1.add_vertex(self.v5_g1)

        self.e12 = graph_edge.DirectedUnWeightedGraphEdge(self.graph1, self.v1_g1, self.v2_g1) # S -> T
        self.e14 = graph_edge.DirectedUnWeightedGraphEdge(self.graph1, self.v1_g1, self.v4_g1) # S -> Y
        self.e23 = graph_edge.DirectedUnWeightedGraphEdge(self.graph1, self.v2_g1, self.v3_g1) # T -> X
        self.e24 = graph_edge.DirectedUnWeightedGraphEdge(self.graph1, self.v2_g1, self.v4_g1) # T -> Y
        self.e35 = graph_edge.DirectedUnWeightedGraphEdge(self.graph1, self.v3_g1, self.v5_g1) # X -> Z
        self.e42 = graph_edge.DirectedUnWeightedGraphEdge(self.graph1, self.v4_g1, self.v2_g1) # Y -> T
        self.e43 = graph_edge.DirectedUnWeightedGraphEdge(self.graph1, self.v4_g1, self.v3_g1) # Y -> X
        self.e45 = graph_edge.DirectedUnWeightedGraphEdge(self.graph1, self.v4_g1, self.v5_g1) # Y -> Z
        self.e53 = graph_edge.DirectedUnWeightedGraphEdge(self.graph1, self.v5_g1, self.v3_g1) # Z -> X
        self.e51 = graph_edge.DirectedUnWeightedGraphEdge(self.graph1, self.v5_g1, self.v1_g1) # Z -> S

        self.graph1.add_edge(self.v1_g1, self.v2_g1)
        self.graph1.add_edge(self.v1_g1, self.v4_g1)
        self.graph1.add_edge(self.v2_g1, self.v3_g1)
        self.graph1.add_edge(self.v2_g1, self.v4_g1)
        self.graph1.add_edge(self.v3_g1, self.v5_g1)
        self.graph1.add_edge(self.v4_g1, self.v2_g1)
        self.graph1.add_edge(self.v4_g1, self.v3_g1)
        self.graph1.add_edge(self.v4_g1, self.v5_g1)
        self.graph1.add_edge(self.v5_g1, self.v3_g1)
        self.graph1.add_edge(self.v5_g1, self.v1_g1)

        # Create directed unweighted acyclic graph Bruno R. Preiss - Java - page 563.
        self.graph2 = graph.DirectedUnWeightedGraph(9)

        self.v0_g2 = graph_vertex.UnWeightedGraphVertex(self.graph2, 'a')
        self.v1_g2 = graph_vertex.UnWeightedGraphVertex(self.graph2, 'b')
        self.v2_g2 = graph_vertex.UnWeightedGraphVertex(self.graph2, 'c')
        self.v3_g2 = graph_vertex.UnWeightedGraphVertex(self.graph2, 'd')
        self.v4_g2 = graph_vertex.UnWeightedGraphVertex(self.graph2, 'e')
        self.v5_g2 = graph_vertex.UnWeightedGraphVertex(self.graph2, 'f')
        self.v6_g2 = graph_vertex.UnWeightedGraphVertex(self.graph2, 'g')
        self.v7_g2 = graph_vertex.UnWeightedGraphVertex(self.graph2, 'h')
        self.v8_g2 = graph_vertex.UnWeightedGraphVertex(self.graph2, 'i')

        self.graph2.add_vertex(self.v0_g2)
        self.graph2.add_vertex(self.v1_g2)
        self.graph2.add_vertex(self.v2_g2)
        self.graph2.add_vertex(self.v3_g2)
        self.graph2.add_vertex(self.v4_g2)
        self.graph2.add_vertex(self.v5_g2)
        self.graph2.add_vertex(self.v6_g2)
        self.graph2.add_vertex(self.v7_g2)
        self.graph2.add_vertex(self.v8_g2)

        self.e01_g2 = graph_edge.DirectedUnWeightedGraphEdge(self.graph2, self.v0_g2, self.v1_g2)
        self.e02_g2 = graph_edge.DirectedUnWeightedGraphEdge(self.graph2, self.v0_g2, self.v2_g2)
        self.e04_g2 = graph_edge.DirectedUnWeightedGraphEdge(self.graph2, self.v0_g2, self.v4_g2)
        self.e13_g2 = graph_edge.DirectedUnWeightedGraphEdge(self.graph2, self.v1_g2, self.v3_g2)
        self.e14_g2 = graph_edge.DirectedUnWeightedGraphEdge(self.graph2, self.v1_g2, self.v4_g2)
        self.e27_g2 = graph_edge.DirectedUnWeightedGraphEdge(self.graph2, self.v2_g2, self.v7_g2)
        self.e25_g2 = graph_edge.DirectedUnWeightedGraphEdge(self.graph2, self.v2_g2, self.v5_g2)
        self.e36_g2 = graph_edge.DirectedUnWeightedGraphEdge(self.graph2, self.v3_g2, self.v6_g2)
        self.e46_g2 = graph_edge.DirectedUnWeightedGraphEdge(self.graph2, self.v4_g2, self.v6_g2)
        self.e48_g2 = graph_edge.DirectedUnWeightedGraphEdge(self.graph2, self.v4_g2, self.v8_g2)
        self.e47_g2 = graph_edge.DirectedUnWeightedGraphEdge(self.graph2, self.v4_g2, self.v7_g2)
        self.e57_g2 = graph_edge.DirectedUnWeightedGraphEdge(self.graph2, self.v5_g2, self.v7_g2)
        self.e68_g2 = graph_edge.DirectedUnWeightedGraphEdge(self.graph2, self.v6_g2, self.v8_g2)
        self.e78_g2 = graph_edge.DirectedUnWeightedGraphEdge(self.graph2, self.v7_g2, self.v8_g2)

        self.graph2.add_edge(self.v0_g2, self.v1_g2) # a -> b
        self.graph2.add_edge(self.v0_g2, self.v2_g2) # a -> c
        self.graph2.add_edge(self.v0_g2, self.v4_g2) # a -> e
        self.graph2.add_edge(self.v1_g2, self.v3_g2) # b -> d
        self.graph2.add_edge(self.v1_g2, self.v4_g2) # b -> e
        self.graph2.add_edge(self.v2_g2, self.v7_g2) # c -> h
        self.graph2.add_edge(self.v2_g2, self.v5_g2) # c -> f
        self.graph2.add_edge(self.v3_g2, self.v6_g2) # d -> g
        self.graph2.add_edge(self.v4_g2, self.v6_g2) # e -> g
        self.graph2.add_edge(self.v4_g2, self.v8_g2) # e -> i
        self.graph2.add_edge(self.v4_g2, self.v7_g2) # e -> h
        self.graph2.add_edge(self.v5_g2, self.v7_g2) # f -> h
        self.graph2.add_edge(self.v6_g2, self.v8_g2) # g -> i
        self.graph2.add_edge(self.v7_g2, self.v8_g2) # h -> i

    def test_directed_un_weighted_graph_copy(self):
        """
        Test operator "copy".
        """
        a_graph = graph.DirectedUnWeightedGraph(5)
        vertex1 = graph_vertex.UnWeightedGraphVertex(a_graph, 'A')
        vertex2 = graph_vertex.UnWeightedGraphVertex(a_graph, 'B')
        vertex3 = graph_vertex.UnWeightedGraphVertex(a_graph, 'C')
        vertex4 = graph_vertex.UnWeightedGraphVertex(a_graph, 'D')
        vertex5 = graph_vertex.UnWeightedGraphVertex(a_graph, 'E')

        a_graph.add_vertex(vertex1)
        a_graph.add_vertex(vertex2)
        a_graph.add_vertex(vertex3)
        a_graph.add_vertex(vertex4)
        a_graph.add_vertex(vertex5)

        a_graph.add_edge(vertex1, vertex2)
        a_graph.add_edge(vertex1, vertex3)
        a_graph.add_edge(vertex1, vertex4)
        a_graph.add_edge(vertex1, vertex5)
        a_graph.add_edge(vertex2, vertex3)
        a_graph.add_edge(vertex2, vertex4)
        a_graph.add_edge(vertex2, vertex5)
        a_graph.add_edge(vertex3, vertex4)
        a_graph.add_edge(vertex3, vertex5)
        a_graph.add_edge(vertex4, vertex5)
        ref = copy.copy(a_graph)
        self.assertEqual(a_graph, ref)

    def test_directed_un_weighted_graph_len(self):
        """
        Test operator "len".
        """
        self.assertEqual(5, len(self.graph1))

    def test_directed_un_weighted_graph_get_item(self):
        """
        Test operator "get_item".
        """
        self.assertEqual(self.graph1.get_vertex_at_index(3), self.graph1[3])

    def test_directed_un_weighted_graph_get_number_of_vertices(self):
        """
        Test method "get_number_of_vertices".
        """
        self.assertEqual(5, self.graph1.get_number_of_vertices())

    def test_directed_un_weighted_graph_get_number_of_edges(self):
        """
        Test method "get_number_of_edges".
        """
        self.assertEqual(10, self.graph1.get_number_of_edges())

    def test_directed_un_weighted_graph_get_vertices(self):
        """
        Test method "get_number_of_vertices".
        """
        tmp1 = []
        tmp1.append(self.v1_g1)
        tmp1.append(self.v2_g1)
        tmp1.append(self.v3_g1)
        tmp1.append(self.v4_g1)
        tmp1.append(self.v5_g1)

        tmp2 = []
        for i in self.graph1.get_vertices():
            tmp2.append(i)

        s_list1 = sorted(tmp1, key=lambda vertex: vertex.vertex_number)
        s_list2 = sorted(tmp2, key=lambda vertex: vertex.vertex_number)
        self.assertEqual(s_list1, s_list2)

    def test_directed_un_weighted_graph_get_edges(self):
        """
        Test method "get_edges".
        """
        tmp1 = []
        tmp1.append(self.e12)
        tmp1.append(self.e14)
        tmp1.append(self.e23)
        tmp1.append(self.e24)
        tmp1.append(self.e35)
        tmp1.append(self.e42)
        tmp1.append(self.e43)
        tmp1.append(self.e45)
        tmp1.append(self.e53)
        tmp1.append(self.e51)

        tmp2 = []
        for i in self.graph1.get_edges():
            tmp2.append(i)

        s_list1 = sorted(tmp1, key=lambda edge: (edge.head_vertex, edge.tail_vertex))
        s_list2 = sorted(tmp2, key=lambda edge: (edge.head_vertex, edge.tail_vertex))
        self.assertEqual(s_list1, s_list2)

    def test_directed_un_weighted_graph_get_edge(self):
        """
        Test method "get_edge".
        """
        self.assertEqual(self.e12, self.graph1.get_edge(self.v1_g1, self.v2_g1))

    def test_directed_un_weighted_graph_is_edge(self):
        """
        Test method "is_edge".
        """
        try:
            self.assertTrue(self.graph1.is_edge(self.v1_g1, self.v2_g1))
        except KeyError:
            print "Exception caught: %s" % str(KeyError)

    def test_directed_un_weighted_graph_is_directed(self):
        """
        Test method "is_directed".
        """
        self.assertTrue(self.graph1.is_directed())

    def test_directed_un_weighted_graph_remove_vertex_v0(self):
        """
        Test method "remove_vertex".
        """
        # Create a graph from where a vertex should be removed.
        a_graph = graph.DirectedUnWeightedGraph(5)
        vertex0 = graph_vertex.UnWeightedGraphVertex(a_graph, 'A')
        vertex1 = graph_vertex.UnWeightedGraphVertex(a_graph, 'B')
        vertex2 = graph_vertex.UnWeightedGraphVertex(a_graph, 'C')
        vertex3 = graph_vertex.UnWeightedGraphVertex(a_graph, 'D')
        vertex4 = graph_vertex.UnWeightedGraphVertex(a_graph, 'E')

        # Add vertices to the graph.
        a_graph.add_vertex(vertex0)
        a_graph.add_vertex(vertex1)
        a_graph.add_vertex(vertex2)
        a_graph.add_vertex(vertex3)
        a_graph.add_vertex(vertex4)

        # Add edges to the graph.
        a_graph.add_edge(vertex0, vertex1)
        a_graph.add_edge(vertex0, vertex2)
        a_graph.add_edge(vertex0, vertex3)
        a_graph.add_edge(vertex0, vertex4)
        a_graph.add_edge(vertex1, vertex2)
        a_graph.add_edge(vertex1, vertex3)
        a_graph.add_edge(vertex1, vertex4)
        a_graph.add_edge(vertex2, vertex3)
        a_graph.add_edge(vertex2, vertex4)
        a_graph.add_edge(vertex3, vertex4)

        # Create a reference graph used to compare the result after a vertex has been removed.
        g_ref = graph.DirectedUnWeightedGraph(4)

        # Create reference vertices.
        v1_ref = graph_vertex.UnWeightedGraphVertex(g_ref, 'B')
        v2_ref = graph_vertex.UnWeightedGraphVertex(g_ref, 'C')
        v3_ref = graph_vertex.UnWeightedGraphVertex(g_ref, 'D')
        v4_ref = graph_vertex.UnWeightedGraphVertex(g_ref, 'E')

        # Add vertices to the reference graph.
        g_ref.add_vertex(v1_ref)
        g_ref.add_vertex(v2_ref)
        g_ref.add_vertex(v3_ref)
        g_ref.add_vertex(v4_ref)

        # Add edges to the reference graph.
        g_ref.add_edge(v1_ref, v2_ref)
        g_ref.add_edge(v1_ref, v3_ref)
        g_ref.add_edge(v1_ref, v4_ref)
        g_ref.add_edge(v2_ref, v3_ref)
        g_ref.add_edge(v2_ref, v4_ref)
        g_ref.add_edge(v3_ref, v4_ref)

        # Remove vertex form graph.
        a_graph.remove_vertex(vertex0)
        self.assertEqual(g_ref, a_graph)

    def test_directed_un_weighted_graph_remove_vertex_v1(self):
        """
        Test method "remove_vertex".
        """
        # Create a graph from where a vertex should be removed.
        a_graph = graph.DirectedUnWeightedGraph(5)
        vertex0 = graph_vertex.UnWeightedGraphVertex(a_graph, 'A')
        vertex1 = graph_vertex.UnWeightedGraphVertex(a_graph, 'B')
        vertex2 = graph_vertex.UnWeightedGraphVertex(a_graph, 'C')
        vertex3 = graph_vertex.UnWeightedGraphVertex(a_graph, 'D')
        vertex4 = graph_vertex.UnWeightedGraphVertex(a_graph, 'E')

        # Add vertices to the graph.
        a_graph.add_vertex(vertex0)
        a_graph.add_vertex(vertex1)
        a_graph.add_vertex(vertex2)
        a_graph.add_vertex(vertex3)
        a_graph.add_vertex(vertex4)

        # Add edges to the graph.
        a_graph.add_edge(vertex0, vertex1)
        a_graph.add_edge(vertex0, vertex2)
        a_graph.add_edge(vertex0, vertex3)
        a_graph.add_edge(vertex0, vertex4)
        a_graph.add_edge(vertex1, vertex2)
        a_graph.add_edge(vertex1, vertex3)
        a_graph.add_edge(vertex1, vertex4)
        a_graph.add_edge(vertex2, vertex3)
        a_graph.add_edge(vertex2, vertex4)
        a_graph.add_edge(vertex3, vertex4)

        # Create a reference graph used to compare the result after a vertex has been removed.
        g_ref = graph.DirectedUnWeightedGraph(4)

        # Create reference vertices.
        v0_ref = graph_vertex.UnWeightedGraphVertex(g_ref, 'A')
        v2_ref = graph_vertex.UnWeightedGraphVertex(g_ref, 'C')
        v3_ref = graph_vertex.UnWeightedGraphVertex(g_ref, 'D')
        v4_ref = graph_vertex.UnWeightedGraphVertex(g_ref, 'E')

        # Add vertices to the reference graph.
        g_ref.add_vertex(v0_ref)
        g_ref.add_vertex(v2_ref)
        g_ref.add_vertex(v3_ref)
        g_ref.add_vertex(v4_ref)

        # Add edges to the reference graph.
        g_ref.add_edge(v0_ref, v2_ref)
        g_ref.add_edge(v0_ref, v3_ref)
        g_ref.add_edge(v0_ref, v4_ref)
        g_ref.add_edge(v2_ref, v3_ref)
        g_ref.add_edge(v2_ref, v4_ref)
        g_ref.add_edge(v3_ref, v4_ref)

        # Remove vertex form graph.
        a_graph.remove_vertex(vertex1)
        self.assertEqual(g_ref, a_graph)

    def test_directed_un_weighted_graph_remove_vertex_v2(self):
        """
        Test method "remove_vertex".
        """
        # Create a graph from where a vertex should be removed.
        a_graph = graph.DirectedUnWeightedGraph(5)
        vertex0 = graph_vertex.UnWeightedGraphVertex(a_graph, 'A')
        vertex1 = graph_vertex.UnWeightedGraphVertex(a_graph, 'B')
        vertex2 = graph_vertex.UnWeightedGraphVertex(a_graph, 'C')
        vertex3 = graph_vertex.UnWeightedGraphVertex(a_graph, 'D')
        vertex4 = graph_vertex.UnWeightedGraphVertex(a_graph, 'E')

        # Add vertices to the graph.
        a_graph.add_vertex(vertex0)
        a_graph.add_vertex(vertex1)
        a_graph.add_vertex(vertex2)
        a_graph.add_vertex(vertex3)
        a_graph.add_vertex(vertex4)

        # Add edges to the graph.
        a_graph.add_edge(vertex0, vertex1)
        a_graph.add_edge(vertex0, vertex2)
        a_graph.add_edge(vertex0, vertex3)
        a_graph.add_edge(vertex0, vertex4)
        a_graph.add_edge(vertex1, vertex2)
        a_graph.add_edge(vertex1, vertex3)
        a_graph.add_edge(vertex1, vertex4)
        a_graph.add_edge(vertex2, vertex3)
        a_graph.add_edge(vertex2, vertex4)
        a_graph.add_edge(vertex3, vertex4)

        # Create a reference graph used to compare the result after a vertex has been removed.
        g_ref = graph.DirectedUnWeightedGraph(4)

        # Create reference vertices.
        v0_ref = graph_vertex.UnWeightedGraphVertex(g_ref, 'A')
        v1_ref = graph_vertex.UnWeightedGraphVertex(g_ref, 'B')
        v3_ref = graph_vertex.UnWeightedGraphVertex(g_ref, 'D')
        v4_ref = graph_vertex.UnWeightedGraphVertex(g_ref, 'E')

        # Add vertices to the reference graph.
        g_ref.add_vertex(v0_ref)
        g_ref.add_vertex(v1_ref)
        g_ref.add_vertex(v3_ref)
        g_ref.add_vertex(v4_ref)

        # Add edges to the reference graph.
        g_ref.add_edge(v0_ref, v1_ref)
        g_ref.add_edge(v0_ref, v3_ref)
        g_ref.add_edge(v0_ref, v4_ref)
        g_ref.add_edge(v1_ref, v3_ref)
        g_ref.add_edge(v1_ref, v4_ref)
        g_ref.add_edge(v3_ref, v4_ref)

        # Remove vertex form graph.
        a_graph.remove_vertex(vertex2)
        self.assertEqual(g_ref, a_graph)

    def test_directed_un_weighted_graph_remove_vertex_v3(self):
        """
        Test method "remove_vertex".
        """
        # Create a graph from where a vertex should be removed.
        a_graph = graph.DirectedUnWeightedGraph(5)
        vertex0 = graph_vertex.UnWeightedGraphVertex(a_graph, 'A')
        vertex1 = graph_vertex.UnWeightedGraphVertex(a_graph, 'B')
        vertex2 = graph_vertex.UnWeightedGraphVertex(a_graph, 'C')
        vertex3 = graph_vertex.UnWeightedGraphVertex(a_graph, 'D')
        vertex4 = graph_vertex.UnWeightedGraphVertex(a_graph, 'E')

        # Add vertices to the graph.
        a_graph.add_vertex(vertex0)
        a_graph.add_vertex(vertex1)
        a_graph.add_vertex(vertex2)
        a_graph.add_vertex(vertex3)
        a_graph.add_vertex(vertex4)

        # Add edges to the graph.
        a_graph.add_edge(vertex0, vertex1)
        a_graph.add_edge(vertex0, vertex2)
        a_graph.add_edge(vertex0, vertex3)
        a_graph.add_edge(vertex0, vertex4)
        a_graph.add_edge(vertex1, vertex2)
        a_graph.add_edge(vertex1, vertex3)
        a_graph.add_edge(vertex1, vertex4)
        a_graph.add_edge(vertex2, vertex3)
        a_graph.add_edge(vertex2, vertex4)
        a_graph.add_edge(vertex3, vertex4)

        # Create a reference graph used to compare the result after a vertex has been removed.
        g_ref = graph.DirectedUnWeightedGraph(4)

        # Create reference vertices.
        v0_ref = graph_vertex.UnWeightedGraphVertex(g_ref, 'A')
        v1_ref = graph_vertex.UnWeightedGraphVertex(g_ref, 'B')
        v2_ref = graph_vertex.UnWeightedGraphVertex(g_ref, 'C')
        v4_ref = graph_vertex.UnWeightedGraphVertex(g_ref, 'E')

        # Add vertices to the reference graph.
        g_ref.add_vertex(v0_ref)
        g_ref.add_vertex(v1_ref)
        g_ref.add_vertex(v2_ref)
        g_ref.add_vertex(v4_ref)

        # Add edges to the reference graph.
        g_ref.add_edge(v0_ref, v1_ref)
        g_ref.add_edge(v0_ref, v2_ref)
        g_ref.add_edge(v0_ref, v4_ref)
        g_ref.add_edge(v1_ref, v2_ref)
        g_ref.add_edge(v1_ref, v4_ref)
        g_ref.add_edge(v2_ref, v4_ref)

        # Remove vertex form graph.
        a_graph.remove_vertex(vertex3)
        self.assertEqual(g_ref, a_graph)

    def test_directed_un_weighted_graph_remove_vertex_v4(self):
        """
        Test method "remove_vertex".
        """
        # Create a graph from where a vertex should be removed.
        a_graph = graph.DirectedUnWeightedGraph(5)
        vertex0 = graph_vertex.UnWeightedGraphVertex(a_graph, 'A')
        vertex1 = graph_vertex.UnWeightedGraphVertex(a_graph, 'B')
        vertex2 = graph_vertex.UnWeightedGraphVertex(a_graph, 'C')
        vertex3 = graph_vertex.UnWeightedGraphVertex(a_graph, 'D')
        vertex4 = graph_vertex.UnWeightedGraphVertex(a_graph, 'E')

        # Add vertices to the graph.
        a_graph.add_vertex(vertex0)
        a_graph.add_vertex(vertex1)
        a_graph.add_vertex(vertex2)
        a_graph.add_vertex(vertex3)
        a_graph.add_vertex(vertex4)

        # Add edges to the graph.
        a_graph.add_edge(vertex0, vertex1)
        a_graph.add_edge(vertex0, vertex2)
        a_graph.add_edge(vertex0, vertex3)
        a_graph.add_edge(vertex0, vertex4)
        a_graph.add_edge(vertex1, vertex2)
        a_graph.add_edge(vertex1, vertex3)
        a_graph.add_edge(vertex1, vertex4)
        a_graph.add_edge(vertex2, vertex3)
        a_graph.add_edge(vertex2, vertex4)
        a_graph.add_edge(vertex3, vertex4)

        # Create a reference graph used to compare the result after a vertex has been removed.
        g_ref = graph.DirectedUnWeightedGraph(4)

        # Create reference vertices.
        v0_ref = graph_vertex.UnWeightedGraphVertex(g_ref, 'A')
        v1_ref = graph_vertex.UnWeightedGraphVertex(g_ref, 'B')
        v2_ref = graph_vertex.UnWeightedGraphVertex(g_ref, 'C')
        v3_ref = graph_vertex.UnWeightedGraphVertex(g_ref, 'D')

        # Add vertices to the reference graph.
        g_ref.add_vertex(v0_ref)
        g_ref.add_vertex(v1_ref)
        g_ref.add_vertex(v2_ref)
        g_ref.add_vertex(v3_ref)

        # Add edges to the reference graph.
        g_ref.add_edge(v0_ref, v1_ref)
        g_ref.add_edge(v0_ref, v2_ref)
        g_ref.add_edge(v0_ref, v3_ref)
        g_ref.add_edge(v1_ref, v2_ref)
        g_ref.add_edge(v1_ref, v3_ref)
        g_ref.add_edge(v2_ref, v3_ref)

        # Remove vertex form graph.
        a_graph.remove_vertex(vertex4)
        self.assertEqual(g_ref, a_graph)

    def test_directed_un_weighted_graph_is_strongly_connected(self):
        """
        Test method "is_strongly_connected".
        """
        self.assertTrue(self.graph1.is_strongly_connected())

    def test_directed_un_weighted_graph_is_strongly_connected_not(self):
        """
        Test method "is_strongly_connected" - inverted.
        """
        # The graph 'g2' is not strongly connected,
        # since no vertex can be reached from vertex 'a'
        self.assertFalse(self.graph2.is_strongly_connected())

    def test_directed_un_weighted_graph_is_cyclic(self):
        """
        Test method "is_cyclic".
        """
        # Create the cyclic graph shown below:
        #
        # A---------->----------B
        # |                     |
        # |                     |
        # |                     |
        # ^                     V
        # |                     |
        # |                     |
        # |                     |
        # D----------<----------C

        a_graph = graph.DirectedUnWeightedGraph(4)
        v_a = graph_vertex.UnWeightedGraphVertex(a_graph, 'A')
        v_b = graph_vertex.UnWeightedGraphVertex(a_graph, 'B')
        v_c = graph_vertex.UnWeightedGraphVertex(a_graph, 'C')
        v_d = graph_vertex.UnWeightedGraphVertex(a_graph, 'D')

        a_graph.add_vertex(v_a)
        a_graph.add_vertex(v_b)
        a_graph.add_vertex(v_c)
        a_graph.add_vertex(v_d)

        a_graph.add_edge(v_a, v_b)
        a_graph.add_edge(v_b, v_c)
        a_graph.add_edge(v_c, v_d)
        a_graph.add_edge(v_d, v_a)
        self.assertTrue(a_graph.is_cyclic())

    def test_directed_un_weighted_graph_is_cyclic_not(self):
        """
        Test method "is_cyclic" - inverted.
        """
        # Create the acyclic graph shown below:
        #
        # A----------<----------B
        # |                     |
        # |                     |
        # |                     |
        # ^                     ^
        # |                     |
        # |                     |
        # |                     |
        # D----------<----------C

        a_graph = graph.DirectedUnWeightedGraph(4)

        v_a = graph_vertex.UnWeightedGraphVertex(a_graph, 'A')
        v_b = graph_vertex.UnWeightedGraphVertex(a_graph, 'B')
        v_c = graph_vertex.UnWeightedGraphVertex(a_graph, 'C')
        v_d = graph_vertex.UnWeightedGraphVertex(a_graph, 'D')

        a_graph.add_vertex(v_a)
        a_graph.add_vertex(v_b)
        a_graph.add_vertex(v_c)
        a_graph.add_vertex(v_d)

        a_graph.add_edge(v_b, v_a)
        a_graph.add_edge(v_d, v_a)
        a_graph.add_edge(v_c, v_b)
        a_graph.add_edge(v_c, v_d)
        self.assertFalse(a_graph.is_cyclic())

    def test_directed_un_weighted_graph_get_vertex_at_index(self):
        """
        Test method "get_vertex_at_index".
        """
        self.assertEqual(self.v4_g1, self.graph1.get_vertex_at_index(3))

    def test_directed_un_weighted_graph_get_emanating_edges(self):
        """
        Test method "get_emanating_edges".
        """
        ref = []
        res = []
        ref.append(self.e12)
        ref.append(self.e14)
        res = self.graph1.get_emanating_edges(self.v1_g1.get_vertex_number())
        self.assertEqual(ref, res)

    def test_directed_un_weighted_graph_get_incident_edges_v1(self):
        """
        Test method "get_incident_edges".
        """
        ref = []
        res = []
        ref.append(self.e51)
        res = self.graph1.get_incident_edges(self.v1_g1.get_vertex_number())
        self.assertEqual(ref, res)

    def test_directed_un_weighted_graph_classify_edges_cyclic(self):
        """
        Test edge classification - directed unweighted cyclic graph.
        """
        # Create a directed unweighted cyclic graph

        a_graph = graph.DirectedUnWeightedGraph(4)
        vertex1 = graph_vertex.UnWeightedGraphVertex(a_graph, 'A')
        vertex2 = graph_vertex.UnWeightedGraphVertex(a_graph, 'B')
        vertex3 = graph_vertex.UnWeightedGraphVertex(a_graph, 'C')
        vertex4 = graph_vertex.UnWeightedGraphVertex(a_graph, 'D')

        a_graph.add_vertex(vertex1)
        a_graph.add_vertex(vertex2)
        a_graph.add_vertex(vertex3)
        a_graph.add_vertex(vertex4)

        a_graph.add_edge(vertex1, vertex2)
        a_graph.add_edge(vertex2, vertex3)
        a_graph.add_edge(vertex3, vertex1)

        res = a_graph.classify_edges().get_edges()
        ref = dfs_edge_classification.DFSEdgeClassification(a_graph).get_edges()
        e12 = graph_edge.DirectedUnWeightedGraphEdge(a_graph, vertex1, vertex2)
        e23 = graph_edge.DirectedUnWeightedGraphEdge(a_graph, vertex2, vertex3)
        e31 = graph_edge.DirectedUnWeightedGraphEdge(a_graph, vertex3, vertex1)
        ref[e12] = graph_edge.EdgeClassification.TREE_EDGE
        ref[e23] = graph_edge.EdgeClassification.TREE_EDGE
        ref[e31] = graph_edge.EdgeClassification.BACK_EDGE
        self.assertEqual(res, ref)

    def test_directed_un_weighted_graph_classify_edges_acyclic(self):
        """
        Test edge classification - directed unweighted acyclic graph.
        """
        # Create a directed unweighted acyclic graph

        a_graph = graph.DirectedUnWeightedGraph(4)
        vertex1 = graph_vertex.UnWeightedGraphVertex(a_graph, 'A')
        vertex2 = graph_vertex.UnWeightedGraphVertex(a_graph, 'B')
        vertex3 = graph_vertex.UnWeightedGraphVertex(a_graph, 'C')
        vertex4 = graph_vertex.UnWeightedGraphVertex(a_graph, 'D')

        a_graph.add_vertex(vertex1)
        a_graph.add_vertex(vertex2)
        a_graph.add_vertex(vertex3)
        a_graph.add_vertex(vertex4)

        a_graph.add_edge(vertex1, vertex2)
        a_graph.add_edge(vertex2, vertex3)
        a_graph.add_edge(vertex2, vertex4)

        res = a_graph.classify_edges().get_edges()
        ref = dfs_edge_classification.DFSEdgeClassification(a_graph).get_edges()
        e12 = graph_edge.DirectedUnWeightedGraphEdge(a_graph, vertex1, vertex2)
        e23 = graph_edge.DirectedUnWeightedGraphEdge(a_graph, vertex2, vertex3)
        e24 = graph_edge.DirectedUnWeightedGraphEdge(a_graph, vertex2, vertex4)
        ref[e12] = graph_edge.EdgeClassification.TREE_EDGE
        ref[e23] = graph_edge.EdgeClassification.TREE_EDGE
        ref[e24] = graph_edge.EdgeClassification.TREE_EDGE
        self.assertEqual(res, ref)

