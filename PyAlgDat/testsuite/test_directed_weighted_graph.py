#!/usr/bin/env py.test

"""
Test directed weighted graph.
"""

import unittest
import copy
from py_alg_dat import dfs_edge_classification
from py_alg_dat import graph
from py_alg_dat import graph_edge
from py_alg_dat import graph_vertex

class TestDirectedWeightedGraph(unittest.TestCase):

    """
    Test directed weighted graph.
    """

    def setUp(self):
        # Create directed weighted graph Cormen page 596.
        self.graph1 = graph.DirectedWeightedGraph(5)

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

        self.e12 = graph_edge.DirectedWeightedGraphEdge(self.graph1, self.v1_g1, self.v2_g1, 10)
        self.e14 = graph_edge.DirectedWeightedGraphEdge(self.graph1, self.v1_g1, self.v4_g1, 5)
        self.e23 = graph_edge.DirectedWeightedGraphEdge(self.graph1, self.v2_g1, self.v3_g1, 1)
        self.e24 = graph_edge.DirectedWeightedGraphEdge(self.graph1, self.v2_g1, self.v4_g1, 2)
        self.e35 = graph_edge.DirectedWeightedGraphEdge(self.graph1, self.v3_g1, self.v5_g1, 4)
        self.e42 = graph_edge.DirectedWeightedGraphEdge(self.graph1, self.v4_g1, self.v2_g1, 3)
        self.e43 = graph_edge.DirectedWeightedGraphEdge(self.graph1, self.v4_g1, self.v3_g1, 9)
        self.e45 = graph_edge.DirectedWeightedGraphEdge(self.graph1, self.v4_g1, self.v5_g1, 2)
        self.e53 = graph_edge.DirectedWeightedGraphEdge(self.graph1, self.v5_g1, self.v3_g1, 6)
        self.e51 = graph_edge.DirectedWeightedGraphEdge(self.graph1, self.v5_g1, self.v1_g1, 7)

        self.graph1.add_edge(self.v1_g1, self.v2_g1, 10) # (S -> T, 10)
        self.graph1.add_edge(self.v1_g1, self.v4_g1, 5)  # (S -> Y, 5)
        self.graph1.add_edge(self.v2_g1, self.v3_g1, 1)  # (T -> X, 1)
        self.graph1.add_edge(self.v2_g1, self.v4_g1, 2)  # (T -> Y, 2)
        self.graph1.add_edge(self.v3_g1, self.v5_g1, 4)  # (X -> Z, 4)
        self.graph1.add_edge(self.v4_g1, self.v2_g1, 3)  # (Y -> T, 3)
        self.graph1.add_edge(self.v4_g1, self.v3_g1, 9)  # (Y -> X, 9)
        self.graph1.add_edge(self.v4_g1, self.v5_g1, 2)  # (Y -> Z, 2)
        self.graph1.add_edge(self.v5_g1, self.v3_g1, 6)  # (Z -> X, 6)
        self.graph1.add_edge(self.v5_g1, self.v1_g1, 7)  # (Z -> S, 7)

        # Create directed unweighted acyclic graph Bruno R. Preiss - Java - page 563.
        self.graph2 = graph.DirectedWeightedGraph(9)

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

        self.e01_g2 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v0_g2, self.v1_g2, 10)
        self.e02_g2 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v0_g2, self.v2_g2, 11)
        self.e04_g2 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v0_g2, self.v4_g2, 12)
        self.e13_g2 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v1_g2, self.v3_g2, 13)
        self.e14_g2 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v1_g2, self.v4_g2, 14)
        self.e27_g2 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v2_g2, self.v7_g2, 15)
        self.e25_g2 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v2_g2, self.v5_g2, 16)
        self.e36_g2 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v3_g2, self.v6_g2, 17)
        self.e46_g2 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v4_g2, self.v6_g2, 18)
        self.e48_g2 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v4_g2, self.v8_g2, 19)
        self.e47_g2 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v4_g2, self.v7_g2, 20)
        self.e57_g2 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v5_g2, self.v7_g2, 21)
        self.e68_g2 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v6_g2, self.v8_g2, 22)
        self.e78_g2 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v7_g2, self.v8_g2, 23)

        self.graph2.add_edge(self.v0_g2, self.v1_g2, 10) # a -> b
        self.graph2.add_edge(self.v0_g2, self.v2_g2, 11) # a -> c
        self.graph2.add_edge(self.v0_g2, self.v4_g2, 12) # a -> e
        self.graph2.add_edge(self.v1_g2, self.v3_g2, 13) # b -> d
        self.graph2.add_edge(self.v1_g2, self.v4_g2, 14) # b -> e
        self.graph2.add_edge(self.v2_g2, self.v7_g2, 15) # c -> h
        self.graph2.add_edge(self.v2_g2, self.v5_g2, 16) # c -> f
        self.graph2.add_edge(self.v3_g2, self.v6_g2, 17) # d -> g
        self.graph2.add_edge(self.v4_g2, self.v6_g2, 18) # e -> g
        self.graph2.add_edge(self.v4_g2, self.v8_g2, 19) # e -> i
        self.graph2.add_edge(self.v4_g2, self.v7_g2, 20) # e -> h
        self.graph2.add_edge(self.v5_g2, self.v7_g2, 21) # f -> h
        self.graph2.add_edge(self.v6_g2, self.v8_g2, 22) # g -> i
        self.graph2.add_edge(self.v7_g2, self.v8_g2, 24) # h -> i

    def test_directed_weighted_graph_copy(self):
        """
        Test operator "copy".
        """
        a_graph = graph.DirectedWeightedGraph(5)
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

        a_graph.add_edge(vertex1, vertex2, 1)
        a_graph.add_edge(vertex1, vertex3, 2)
        a_graph.add_edge(vertex1, vertex4, 3)
        a_graph.add_edge(vertex1, vertex5, 4)
        a_graph.add_edge(vertex2, vertex3, 5)
        a_graph.add_edge(vertex2, vertex4, 6)
        a_graph.add_edge(vertex2, vertex5, 7)
        a_graph.add_edge(vertex3, vertex4, 8)
        a_graph.add_edge(vertex3, vertex5, 9)
        a_graph.add_edge(vertex4, vertex5, 10)
        ref = copy.copy(a_graph)
        self.assertEqual(a_graph, ref)

    def test_directed_weighted_graph_len(self):
        """
        Test operator "len".
        """
        self.assertEqual(5, len(self.graph1))

    def test_directed_weighted_graph_get_item(self):
        """
        Test operator "get_item".
        """
        self.assertEqual(self.graph1.get_vertex_at_index(3), self.graph1[3])

    def test_directed_weighted_graph_get_number_of_vertices(self):
        """
        Test method "get_number_of_vertices".
        """
        self.assertEqual(5, self.graph1.get_number_of_vertices())

    def test_directed_weighted_graph_get_number_of_edges(self):
        """
        Test method "get_number_of_edges".
        """
        self.assertEqual(10, self.graph1.get_number_of_edges())

    def test_directed_weighted_graph_get_vertices(self):
        """
        Test method "get_vertices".
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

    def test_directed_weighted_graph_get_edges(self):
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

        s_list1 = sorted(tmp1, key=lambda edge: (edge.head_vertex, edge.tail_vertex, edge.weight))
        s_list2 = sorted(tmp2, key=lambda edge: (edge.head_vertex, edge.tail_vertex, edge.weight))
        self.assertEqual(s_list1, s_list2)

    def test_directed_weighted_graph_get_edge(self):
        """
        Test method "get_edge".
        """
        self.assertEqual(self.e12, self.graph1.get_edge(self.v1_g1, self.v2_g1))

    def test_directed_weighted_graph_is_edge(self):
        """
        Test method "is_edge".
        """
        try:
            self.assertTrue(self.graph1.is_edge(self.v1_g1, self.v2_g1))
        except KeyError:
            print "Exception caught: %s" % str(KeyError)

    def test_directed_weighted_graph_is_directed(self):
        """
        Test method "is_directed".
        """
        self.assertTrue(self.graph1.is_directed())

    def test_directed_weighted_graph_remove_vertex_v0(self):
        """
        Test method "remove_vertex".
        """
        # Create a graph from where a vertex should be removed.
        a_graph = graph.DirectedWeightedGraph(5)
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
        a_graph.add_edge(vertex0, vertex1, 10)
        a_graph.add_edge(vertex0, vertex2, 20)
        a_graph.add_edge(vertex0, vertex3, 30)
        a_graph.add_edge(vertex0, vertex4, 40)
        a_graph.add_edge(vertex1, vertex2, 50)
        a_graph.add_edge(vertex1, vertex3, 60)
        a_graph.add_edge(vertex1, vertex4, 70)
        a_graph.add_edge(vertex2, vertex3, 80)
        a_graph.add_edge(vertex2, vertex4, 90)
        a_graph.add_edge(vertex3, vertex4, 100)

        # Create a reference graph used to compare the result after a vertex has been removed.
        g_ref = graph.DirectedWeightedGraph(4)

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
        g_ref.add_edge(v1_ref, v2_ref, 50)
        g_ref.add_edge(v1_ref, v3_ref, 60)
        g_ref.add_edge(v1_ref, v4_ref, 70)
        g_ref.add_edge(v2_ref, v3_ref, 80)
        g_ref.add_edge(v2_ref, v4_ref, 90)
        g_ref.add_edge(v3_ref, v4_ref, 100)

        # Remove vertex form graph.
        a_graph.remove_vertex(vertex0)
        self.assertEqual(g_ref, a_graph)

    def test_directed_weighted_graph_remove_vertex_v1(self):
        """
        Test method "remove_vertex".
        """
        # Create a graph from where a vertex should be removed.
        a_graph = graph.DirectedWeightedGraph(5)
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
        a_graph.add_edge(vertex0, vertex1, 10)
        a_graph.add_edge(vertex0, vertex2, 20)
        a_graph.add_edge(vertex0, vertex3, 30)
        a_graph.add_edge(vertex0, vertex4, 40)
        a_graph.add_edge(vertex1, vertex2, 50)
        a_graph.add_edge(vertex1, vertex3, 60)
        a_graph.add_edge(vertex1, vertex4, 70)
        a_graph.add_edge(vertex2, vertex3, 80)
        a_graph.add_edge(vertex2, vertex4, 90)
        a_graph.add_edge(vertex3, vertex4, 100)

        # Create a reference graph used to compare the result after a vertex has been removed.
        g_ref = graph.DirectedWeightedGraph(4)

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
        g_ref.add_edge(v0_ref, v2_ref, 20)
        g_ref.add_edge(v0_ref, v3_ref, 30)
        g_ref.add_edge(v0_ref, v4_ref, 40)
        g_ref.add_edge(v2_ref, v3_ref, 80)
        g_ref.add_edge(v2_ref, v4_ref, 90)
        g_ref.add_edge(v3_ref, v4_ref, 100)

        # Remove vertex form graph.
        a_graph.remove_vertex(vertex1)
        self.assertEqual(g_ref, a_graph)

    def test_directed_weighted_graph_remove_vertex_v2(self):
        """
        Test method "remove_vertex".
        """
        # Create a graph from where a vertex should be removed.
        a_graph = graph.DirectedWeightedGraph(5)
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
        a_graph.add_edge(vertex0, vertex1, 10)
        a_graph.add_edge(vertex0, vertex2, 20)
        a_graph.add_edge(vertex0, vertex3, 30)
        a_graph.add_edge(vertex0, vertex4, 40)
        a_graph.add_edge(vertex1, vertex2, 50)
        a_graph.add_edge(vertex1, vertex3, 60)
        a_graph.add_edge(vertex1, vertex4, 70)
        a_graph.add_edge(vertex2, vertex3, 80)
        a_graph.add_edge(vertex2, vertex4, 90)
        a_graph.add_edge(vertex3, vertex4, 100)

        # Create a reference graph used to compare the result after a vertex has been removed.
        g_ref = graph.DirectedWeightedGraph(4)

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
        g_ref.add_edge(v0_ref, v1_ref, 10)
        g_ref.add_edge(v0_ref, v3_ref, 30)
        g_ref.add_edge(v0_ref, v4_ref, 40)
        g_ref.add_edge(v1_ref, v3_ref, 60)
        g_ref.add_edge(v1_ref, v4_ref, 70)
        g_ref.add_edge(v3_ref, v4_ref, 100)

        # Remove vertex form graph.
        a_graph.remove_vertex(vertex2)
        self.assertEqual(g_ref, a_graph)

    def test_directed_weighted_graph_remove_vertex_v3(self):
        """
        Test method "remove_vertex".
        """
        # Create a graph from where a vertex should be removed.
        a_graph = graph.DirectedWeightedGraph(5)
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
        a_graph.add_edge(vertex0, vertex1, 10)
        a_graph.add_edge(vertex0, vertex2, 20)
        a_graph.add_edge(vertex0, vertex3, 30)
        a_graph.add_edge(vertex0, vertex4, 40)
        a_graph.add_edge(vertex1, vertex2, 50)
        a_graph.add_edge(vertex1, vertex3, 60)
        a_graph.add_edge(vertex1, vertex4, 70)
        a_graph.add_edge(vertex2, vertex3, 80)
        a_graph.add_edge(vertex2, vertex4, 90)
        a_graph.add_edge(vertex3, vertex4, 100)

        # Create a reference graph used to compare the result after a vertex has been removed.
        g_ref = graph.DirectedWeightedGraph(4)

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
        g_ref.add_edge(v0_ref, v1_ref, 10)
        g_ref.add_edge(v0_ref, v2_ref, 20)
        g_ref.add_edge(v0_ref, v4_ref, 40)
        g_ref.add_edge(v1_ref, v2_ref, 50)
        g_ref.add_edge(v1_ref, v4_ref, 70)
        g_ref.add_edge(v2_ref, v4_ref, 90)

        # Remove vertex form graph.
        a_graph.remove_vertex(vertex3)
        self.assertEqual(g_ref, a_graph)

    def test_directed_weighted_graph_remove_vertex_v4(self):
        """
        Test method "remove_vertex".
        """
        # Create a graph from where a vertex should be removed.
        a_graph = graph.DirectedWeightedGraph(5)
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
        a_graph.add_edge(vertex0, vertex1, 10)
        a_graph.add_edge(vertex0, vertex2, 20)
        a_graph.add_edge(vertex0, vertex3, 30)
        a_graph.add_edge(vertex0, vertex4, 40)
        a_graph.add_edge(vertex1, vertex2, 50)
        a_graph.add_edge(vertex1, vertex3, 60)
        a_graph.add_edge(vertex1, vertex4, 70)
        a_graph.add_edge(vertex2, vertex3, 80)
        a_graph.add_edge(vertex2, vertex4, 90)
        a_graph.add_edge(vertex3, vertex4, 100)

        # Create a reference graph used to compare the result after a vertex has been removed.
        g_ref = graph.DirectedWeightedGraph(4)

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
        g_ref.add_edge(v0_ref, v1_ref, 10)
        g_ref.add_edge(v0_ref, v2_ref, 20)
        g_ref.add_edge(v0_ref, v3_ref, 30)
        g_ref.add_edge(v1_ref, v2_ref, 50)
        g_ref.add_edge(v1_ref, v3_ref, 60)
        g_ref.add_edge(v2_ref, v3_ref, 80)

        # Remove vertex form graph.
        a_graph.remove_vertex(vertex4)
        self.assertEqual(g_ref, a_graph)

    def test_directed_weighted_graph_is_strongly_connected(self):
        """
        Test method "is_strongly_connected".
        """
        self.assertTrue(self.graph1.is_strongly_connected())

    def test_directed_weighted_graph_is_strongly_connected_not(self):
        """
        Test method "is_strongly_connected" - inverted.
        """
        # The graph 'g2' is not strongly connected,
        # since no vertex can be reached from vertex 'a'
        self.assertFalse(self.graph2.is_strongly_connected())

    def test_directed_weighted_graph_is_cyclic(self):
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

        a_graph = graph.DirectedWeightedGraph(4)
        vertex_a = graph_vertex.UnWeightedGraphVertex(a_graph, 'A')
        vertex_b = graph_vertex.UnWeightedGraphVertex(a_graph, 'B')
        vertex_c = graph_vertex.UnWeightedGraphVertex(a_graph, 'C')
        vertex_d = graph_vertex.UnWeightedGraphVertex(a_graph, 'D')

        a_graph.add_vertex(vertex_a)
        a_graph.add_vertex(vertex_b)
        a_graph.add_vertex(vertex_c)
        a_graph.add_vertex(vertex_d)

        a_graph.add_edge(vertex_a, vertex_b, 1)
        a_graph.add_edge(vertex_b, vertex_c, 2)
        a_graph.add_edge(vertex_c, vertex_d, 3)
        a_graph.add_edge(vertex_d, vertex_a, 4)
        self.assertTrue(a_graph.is_cyclic())

    def test_directed_weighted_graph_is_cyclic_not(self):
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

        a_graph = graph.DirectedWeightedGraph(4)

        vertex_a = graph_vertex.UnWeightedGraphVertex(a_graph, 'A')
        vertex_b = graph_vertex.UnWeightedGraphVertex(a_graph, 'B')
        vertex_c = graph_vertex.UnWeightedGraphVertex(a_graph, 'C')
        vertex_d = graph_vertex.UnWeightedGraphVertex(a_graph, 'D')

        a_graph.add_vertex(vertex_a)
        a_graph.add_vertex(vertex_b)
        a_graph.add_vertex(vertex_c)
        a_graph.add_vertex(vertex_d)

        a_graph.add_edge(vertex_b, vertex_a, 1)
        a_graph.add_edge(vertex_d, vertex_a, 2)
        a_graph.add_edge(vertex_c, vertex_b, 3)
        a_graph.add_edge(vertex_c, vertex_d, 4)
        self.assertFalse(a_graph.is_cyclic())

    def test_directed_weighted_graph_get_vertex_at_index(self):
        """
        Test method "get_vertex_at_index".
        """
        self.assertEqual(self.v4_g1, self.graph1.get_vertex_at_index(3))

    def test_directed_weighted_graph_get_emanating_edges(self):
        """
        Test method "get_emanating_edges".
        """
        ref = []
        res = []
        ref.append(self.e12)
        ref.append(self.e14)
        res = self.graph1.get_emanating_edges(self.v1_g1.get_vertex_number())
        self.assertEqual(ref, res)

    def test_directed_weighted_graph_get_incident_edges(self):
        """
        Test method "get_incident_edges".
        """
        ref = []
        res = []
        ref.append(self.e51)
        res = self.graph1.get_incident_edges(self.v1_g1.get_vertex_number())
        self.assertEqual(ref, res)

    def test_directed_weighted_graph_classify_edges_cyclic(self):
        """
        Test graph classification for directed weighted cyclic graph.
        """
        # Create a directed weighted cyclic graph

        a_graph = graph.DirectedWeightedGraph(4)
        vertex1 = graph_vertex.UnWeightedGraphVertex(a_graph, 'A')
        vertex2 = graph_vertex.UnWeightedGraphVertex(a_graph, 'B')
        vertex3 = graph_vertex.UnWeightedGraphVertex(a_graph, 'C')
        vertex4 = graph_vertex.UnWeightedGraphVertex(a_graph, 'D')

        a_graph.add_vertex(vertex1)
        a_graph.add_vertex(vertex2)
        a_graph.add_vertex(vertex3)
        a_graph.add_vertex(vertex4)

        a_graph.add_edge(vertex1, vertex2, 10)
        a_graph.add_edge(vertex2, vertex3, 20)
        a_graph.add_edge(vertex3, vertex1, 30)

        res = a_graph.classify_edges().get_edges()
        ref = dfs_edge_classification.DFSEdgeClassification(a_graph).get_edges()
        e12 = graph_edge.DirectedWeightedGraphEdge(a_graph, vertex1, vertex2, 10)
        e23 = graph_edge.DirectedWeightedGraphEdge(a_graph, vertex2, vertex3, 20)
        e31 = graph_edge.DirectedWeightedGraphEdge(a_graph, vertex3, vertex1, 30)
        ref[e12] = graph_edge.EdgeClassification.TREE_EDGE
        ref[e23] = graph_edge.EdgeClassification.TREE_EDGE
        ref[e31] = graph_edge.EdgeClassification.BACK_EDGE
        self.assertEqual(res, ref)

    def test_directed_weighted_graph_classify_edges_acyclic(self):
        """
        Test graph classification for directed weighted acyclic graph.
        """
        # Create a directed weighted acyclic graph

        a_graph = graph.DirectedWeightedGraph(4)
        vertex1 = graph_vertex.UnWeightedGraphVertex(a_graph, 'A')
        vertex2 = graph_vertex.UnWeightedGraphVertex(a_graph, 'B')
        vertex3 = graph_vertex.UnWeightedGraphVertex(a_graph, 'C')
        vertex4 = graph_vertex.UnWeightedGraphVertex(a_graph, 'D')

        a_graph.add_vertex(vertex1)
        a_graph.add_vertex(vertex2)
        a_graph.add_vertex(vertex3)
        a_graph.add_vertex(vertex4)

        a_graph.add_edge(vertex1, vertex2, 10)
        a_graph.add_edge(vertex2, vertex3, 20)
        a_graph.add_edge(vertex2, vertex4, 30)

        res = a_graph.classify_edges().get_edges()
        ref = dfs_edge_classification.DFSEdgeClassification(a_graph).get_edges()
        e12 = graph_edge.DirectedWeightedGraphEdge(a_graph, vertex1, vertex2, 10)
        e23 = graph_edge.DirectedWeightedGraphEdge(a_graph, vertex2, vertex3, 20)
        e24 = graph_edge.DirectedWeightedGraphEdge(a_graph, vertex2, vertex4, 30)
        ref[e12] = graph_edge.EdgeClassification.TREE_EDGE
        ref[e23] = graph_edge.EdgeClassification.TREE_EDGE
        ref[e24] = graph_edge.EdgeClassification.TREE_EDGE
        self.assertEqual(res, ref)

