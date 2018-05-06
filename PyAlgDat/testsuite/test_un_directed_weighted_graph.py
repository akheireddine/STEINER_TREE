#!/usr/bin/env py.test

"""
Test UnDirectedWeightedGraph class.
"""

import unittest
import copy
from py_alg_dat import dfs_edge_classification
from py_alg_dat import graph
from py_alg_dat import graph_edge
from py_alg_dat import graph_vertex
from py_alg_dat import vertex_visitor

class TestUnDirectedWeightedGraph(unittest.TestCase):

    """
    Test UnDirectedWeightedGraph class.
    """

    def setUp(self):
        self.g_1 = graph.UnDirectedWeightedGraph(7)

        self.v_1 = graph_vertex.UnWeightedGraphVertex(self.g_1, 'A')
        self.v_2 = graph_vertex.UnWeightedGraphVertex(self.g_1, 'B')
        self.v_3 = graph_vertex.UnWeightedGraphVertex(self.g_1, 'C')
        self.v_4 = graph_vertex.UnWeightedGraphVertex(self.g_1, 'D')
        self.v_5 = graph_vertex.UnWeightedGraphVertex(self.g_1, 'E')
        self.v_6 = graph_vertex.UnWeightedGraphVertex(self.g_1, 'F')
        self.v_7 = graph_vertex.UnWeightedGraphVertex(self.g_1, 'G')

        self.g_1.add_vertex(self.v_1)
        self.g_1.add_vertex(self.v_2)
        self.g_1.add_vertex(self.v_3)
        self.g_1.add_vertex(self.v_4)
        self.g_1.add_vertex(self.v_5)
        self.g_1.add_vertex(self.v_6)
        self.g_1.add_vertex(self.v_7)

        self.e12 = graph_edge.UnDirectedWeightedGraphEdge(self.g_1, self.v_1, self.v_2, 7)
        self.e21 = graph_edge.UnDirectedWeightedGraphEdge(self.g_1, self.v_2, self.v_1, 7)
        self.e14 = graph_edge.UnDirectedWeightedGraphEdge(self.g_1, self.v_1, self.v_4, 5)
        self.e41 = graph_edge.UnDirectedWeightedGraphEdge(self.g_1, self.v_4, self.v_1, 5)
        self.e23 = graph_edge.UnDirectedWeightedGraphEdge(self.g_1, self.v_2, self.v_3, 8)
        self.e32 = graph_edge.UnDirectedWeightedGraphEdge(self.g_1, self.v_3, self.v_2, 8)
        self.e24 = graph_edge.UnDirectedWeightedGraphEdge(self.g_1, self.v_2, self.v_4, 9)
        self.e42 = graph_edge.UnDirectedWeightedGraphEdge(self.g_1, self.v_4, self.v_2, 9)
        self.e25 = graph_edge.UnDirectedWeightedGraphEdge(self.g_1, self.v_2, self.v_5, 7)
        self.e52 = graph_edge.UnDirectedWeightedGraphEdge(self.g_1, self.v_5, self.v_2, 7)
        self.e35 = graph_edge.UnDirectedWeightedGraphEdge(self.g_1, self.v_3, self.v_5, 5)
        self.e53 = graph_edge.UnDirectedWeightedGraphEdge(self.g_1, self.v_5, self.v_3, 5)
        self.e45 = graph_edge.UnDirectedWeightedGraphEdge(self.g_1, self.v_4, self.v_5, 15)
        self.e54 = graph_edge.UnDirectedWeightedGraphEdge(self.g_1, self.v_5, self.v_4, 15)
        self.e46 = graph_edge.UnDirectedWeightedGraphEdge(self.g_1, self.v_4, self.v_6, 6)
        self.e64 = graph_edge.UnDirectedWeightedGraphEdge(self.g_1, self.v_6, self.v_4, 6)
        self.e56 = graph_edge.UnDirectedWeightedGraphEdge(self.g_1, self.v_5, self.v_6, 8)
        self.e65 = graph_edge.UnDirectedWeightedGraphEdge(self.g_1, self.v_6, self.v_5, 8)
        self.e57 = graph_edge.UnDirectedWeightedGraphEdge(self.g_1, self.v_5, self.v_7, 9)
        self.e75 = graph_edge.UnDirectedWeightedGraphEdge(self.g_1, self.v_7, self.v_5, 9)
        self.e67 = graph_edge.UnDirectedWeightedGraphEdge(self.g_1, self.v_6, self.v_7, 11)
        self.e76 = graph_edge.UnDirectedWeightedGraphEdge(self.g_1, self.v_7, self.v_6, 11)

        self.g_1.add_edge(self.v_1, self.v_2, 7)
        self.g_1.add_edge(self.v_1, self.v_4, 5)
        self.g_1.add_edge(self.v_2, self.v_3, 8)
        self.g_1.add_edge(self.v_2, self.v_4, 9)
        self.g_1.add_edge(self.v_2, self.v_5, 7)
        self.g_1.add_edge(self.v_3, self.v_5, 5)
        self.g_1.add_edge(self.v_4, self.v_5, 15)
        self.g_1.add_edge(self.v_4, self.v_6, 6)
        self.g_1.add_edge(self.v_5, self.v_6, 8)
        self.g_1.add_edge(self.v_5, self.v_7, 9)
        self.g_1.add_edge(self.v_6, self.v_7, 11)

    def test_un_directed_weighted_graph_is_weighted(self):
        """
        Test method "is_weighted".
        """
        self.assertTrue(self.g_1.is_weighted())

    def test_un_directed_weighted_graph_len(self):
        """
        Test operator "len".
        """
        self.assertEqual(7, len(self.g_1))

    def test_un_directed_weighted_graph_get_item(self):
        """
        Test operator "get_item".
        """
        self.assertEqual(self.g_1.get_vertex_at_index(3), self.g_1[3])

    def test_un_directed_weighted_graph_get_number_of_vertices(self):
        """
        Test method "get_number_of_vertices".
        """
        self.assertEqual(7, self.g_1.get_number_of_vertices())

    def test_un_directed_weighted_graph_get_number_of_edges(self):
        """
        Test method "get_number_of_edges".
        """
        self.assertEqual(22, self.g_1.get_number_of_edges())

    def test_un_directed_weighted_graph_get_vertices(self):
        """
        Test method "get_vertices".
        """
        list1 = []
        list1.append(self.v_1)
        list1.append(self.v_2)
        list1.append(self.v_3)
        list1.append(self.v_4)
        list1.append(self.v_5)
        list1.append(self.v_6)
        list1.append(self.v_7)

        list2 = []
        for i in self.g_1.get_vertices():
            list2.append(i)

        s_list1 = sorted(list1, key=lambda vertex: (vertex.vertex_name, vertex.vertex_number))
        s_list2 = sorted(list2, key=lambda vertex: (vertex.vertex_name, vertex.vertex_number))
        self.assertEqual(s_list1, s_list2)

    def test_un_directed_weighted_graph_get_edges(self):
        """
        Test method "get_edges".
        """
        list1 = []
        list1.append(self.e12)
        list1.append(self.e21)
        list1.append(self.e14)
        list1.append(self.e41)
        list1.append(self.e23)
        list1.append(self.e32)
        list1.append(self.e24)
        list1.append(self.e42)
        list1.append(self.e25)
        list1.append(self.e52)
        list1.append(self.e35)
        list1.append(self.e53)
        list1.append(self.e45)
        list1.append(self.e54)
        list1.append(self.e46)
        list1.append(self.e64)
        list1.append(self.e56)
        list1.append(self.e65)
        list1.append(self.e57)
        list1.append(self.e75)
        list1.append(self.e67)
        list1.append(self.e76)

        list2 = []
        for i in self.g_1.get_edges():
            list2.append(i)

        s_list1 = sorted(list1, key=lambda edge: (edge.head_vertex, edge.tail_vertex, edge.weight))
        s_list2 = sorted(list2, key=lambda edge: (edge.head_vertex, edge.tail_vertex, edge.weight))
        self.assertEqual(s_list1, s_list2)

    def test_un_directed_weighted_graph_copy(self):
        """
        Test operator "copy".
        """
        a_graph = graph.UnDirectedWeightedGraph(5)
        v_1 = graph_vertex.UnWeightedGraphVertex(a_graph, 'A')
        v_2 = graph_vertex.UnWeightedGraphVertex(a_graph, 'B')
        v_3 = graph_vertex.UnWeightedGraphVertex(a_graph, 'C')
        v_4 = graph_vertex.UnWeightedGraphVertex(a_graph, 'D')
        v_5 = graph_vertex.UnWeightedGraphVertex(a_graph, 'E')

        a_graph.add_vertex(v_1)
        a_graph.add_vertex(v_2)
        a_graph.add_vertex(v_3)
        a_graph.add_vertex(v_4)
        a_graph.add_vertex(v_5)

        a_graph.add_edge(v_1, v_2, 1)
        a_graph.add_edge(v_1, v_3, 2)
        a_graph.add_edge(v_1, v_4, 3)
        a_graph.add_edge(v_1, v_5, 4)
        a_graph.add_edge(v_2, v_3, 5)
        a_graph.add_edge(v_2, v_4, 6)
        a_graph.add_edge(v_2, v_5, 7)
        a_graph.add_edge(v_3, v_4, 8)
        a_graph.add_edge(v_3, v_5, 9)
        a_graph.add_edge(v_4, v_5, 10)
        ref = copy.copy(a_graph)
        self.assertEqual(a_graph, ref)

    def test_un_directed_weighted_graph_has_vertex(self):
        """
        Test method "has_vertex".
        """
        a_graph = graph.UnDirectedWeightedGraph(1)
        a_vertex = graph_vertex.UnWeightedGraphVertex(a_graph, 'A')
        a_graph.add_vertex(a_vertex)
        self.assertTrue(a_graph.has_vertex(a_vertex))

    def test_un_directed_weighted_graph_has_vertex_not(self):
        """
        Test method "has_vertex" - inverted.
        """
        a_vertex = graph_vertex.UnWeightedGraphVertex(self.g_1, 'X')
        self.assertFalse(self.g_1.has_vertex(a_vertex))

    def test_un_directed_weighted_graph_has_edge(self):
        """
        Test method "has_edge".
        """
        a_graph = graph.UnDirectedWeightedGraph(2)
        vertex_a = graph_vertex.UnWeightedGraphVertex(a_graph, 'A')
        vertex_b = graph_vertex.UnWeightedGraphVertex(a_graph, 'B')
        a_graph.add_vertex(vertex_a)
        a_graph.add_vertex(vertex_b)
        a_edge = graph_edge.UnDirectedWeightedGraphEdge(a_graph, vertex_a, vertex_b, 7)
        a_graph.add_edge(vertex_a, vertex_b, 7)
        self.assertTrue(a_graph.has_edge(a_edge))

    def test_un_directed_weighted_graph_has_edge_not(self):
        """
        Test method "has_edge" - inverted.
        """
        vertex_a = graph_vertex.UnWeightedGraphVertex(self.g_1, 'X')
        vertex_b = graph_vertex.UnWeightedGraphVertex(self.g_1, 'Y')
        a_edge = graph_edge.UnDirectedWeightedGraphEdge(self.g_1, vertex_a, vertex_b, 7)
        self.assertFalse(self.g_1.has_edge(a_edge))

    def test_un_directed_weighted_graph_get_edge(self):
        """
        Test method "get_edge".
        """
        self.assertEqual(self.e12, self.g_1.get_edge(self.v_1, self.v_2))

    def test_un_directed_weighted_graph_is_edge(self):
        """
        Test method "is_edge".
        """
        try:
            self.assertTrue(self.g_1.is_edge(self.v_1, self.v_2))
        except KeyError:
            print "Exception caught: %s" % str(KeyError)

    def test_un_directed_weighted_graph_is_directed(self):
        """
        Test method "is_directed".
        """
        self.assertFalse(self.g_1.is_directed())

    def test_un_directed_weighted_graph_remove_vertex_v0(self):
        """
        Test method "remove_vertex".
        """
        # https://reference.wolfram.com/mathematica/ref/VertexDelete.html
        # Create a graph from where a vertex should be removed.
        a_graph = graph.UnDirectedWeightedGraph(5)
        v_0 = graph_vertex.UnWeightedGraphVertex(a_graph, 'A')
        v_1 = graph_vertex.UnWeightedGraphVertex(a_graph, 'B')
        v_2 = graph_vertex.UnWeightedGraphVertex(a_graph, 'C')
        v_3 = graph_vertex.UnWeightedGraphVertex(a_graph, 'D')
        v_4 = graph_vertex.UnWeightedGraphVertex(a_graph, 'E')

        # Add vertices to the graph.
        a_graph.add_vertex(v_0)
        a_graph.add_vertex(v_1)
        a_graph.add_vertex(v_2)
        a_graph.add_vertex(v_3)
        a_graph.add_vertex(v_4)

        # Add edges to the graph.
        a_graph.add_edge(v_0, v_1, 10)
        a_graph.add_edge(v_0, v_2, 20)
        a_graph.add_edge(v_0, v_3, 30)
        a_graph.add_edge(v_0, v_4, 40)
        a_graph.add_edge(v_1, v_2, 50)
        a_graph.add_edge(v_1, v_3, 60)
        a_graph.add_edge(v_1, v_4, 70)
        a_graph.add_edge(v_2, v_3, 80)
        a_graph.add_edge(v_2, v_4, 90)
        a_graph.add_edge(v_3, v_4, 100)

        # Create a reference graph used to compare the result after a vertex has been removed.
        g_ref = graph.UnDirectedWeightedGraph(4)

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
        a_graph.remove_vertex(v_0)
        self.assertEqual(g_ref, a_graph)

    def test_un_directed_weighted_graph_remove_vertex_v1(self):
        """
        Test method "remove_vertex".
        """
        # Create a graph from where a vertex should be removed.
        a_graph = graph.UnDirectedWeightedGraph(5)
        v_0 = graph_vertex.UnWeightedGraphVertex(a_graph, 'A')
        v_1 = graph_vertex.UnWeightedGraphVertex(a_graph, 'B')
        v_2 = graph_vertex.UnWeightedGraphVertex(a_graph, 'C')
        v_3 = graph_vertex.UnWeightedGraphVertex(a_graph, 'D')
        v_4 = graph_vertex.UnWeightedGraphVertex(a_graph, 'E')

        # Add vertices to the graph.
        a_graph.add_vertex(v_0)
        a_graph.add_vertex(v_1)
        a_graph.add_vertex(v_2)
        a_graph.add_vertex(v_3)
        a_graph.add_vertex(v_4)

        # Add edges to the graph.
        a_graph.add_edge(v_0, v_1, 10)
        a_graph.add_edge(v_0, v_2, 20)
        a_graph.add_edge(v_0, v_3, 30)
        a_graph.add_edge(v_0, v_4, 40)
        a_graph.add_edge(v_1, v_2, 50)
        a_graph.add_edge(v_1, v_3, 60)
        a_graph.add_edge(v_1, v_4, 70)
        a_graph.add_edge(v_2, v_3, 80)
        a_graph.add_edge(v_2, v_4, 90)
        a_graph.add_edge(v_3, v_4, 100)

        # Create a reference graph used to compare the result after a vertex has been removed.
        g_ref = graph.UnDirectedWeightedGraph(4)

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
        a_graph.remove_vertex(v_1)
        self.assertEqual(g_ref, a_graph)

    def test_un_directed_weighted_graph_remove_vertex_v2(self):
        """
        Test method "remove_vertex".
        """
        # Create a graph from where a vertex should be removed.
        a_graph = graph.UnDirectedWeightedGraph(5)
        v_0 = graph_vertex.UnWeightedGraphVertex(a_graph, 'A')
        v_1 = graph_vertex.UnWeightedGraphVertex(a_graph, 'B')
        v_2 = graph_vertex.UnWeightedGraphVertex(a_graph, 'C')
        v_3 = graph_vertex.UnWeightedGraphVertex(a_graph, 'D')
        v_4 = graph_vertex.UnWeightedGraphVertex(a_graph, 'E')

        # Add vertices to the graph.
        a_graph.add_vertex(v_0)
        a_graph.add_vertex(v_1)
        a_graph.add_vertex(v_2)
        a_graph.add_vertex(v_3)
        a_graph.add_vertex(v_4)

        # Add edges to the graph.
        a_graph.add_edge(v_0, v_1, 10)
        a_graph.add_edge(v_0, v_2, 20)
        a_graph.add_edge(v_0, v_3, 30)
        a_graph.add_edge(v_0, v_4, 40)
        a_graph.add_edge(v_1, v_2, 50)
        a_graph.add_edge(v_1, v_3, 60)
        a_graph.add_edge(v_1, v_4, 70)
        a_graph.add_edge(v_2, v_3, 80)
        a_graph.add_edge(v_2, v_4, 90)
        a_graph.add_edge(v_3, v_4, 100)

        # Create a reference graph used to compare the result after a vertex has been removed.
        g_ref = graph.UnDirectedWeightedGraph(4)

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
        a_graph.remove_vertex(v_2)
        self.assertEqual(g_ref, a_graph)

    def test_un_directed_weighted_graph_remove_vertex_v3(self):
        """
        Test method "remove_vertex".
        """
        # Create a graph from where a vertex should be removed.
        a_graph = graph.UnDirectedWeightedGraph(5)
        v_0 = graph_vertex.UnWeightedGraphVertex(a_graph, 'A')
        v_1 = graph_vertex.UnWeightedGraphVertex(a_graph, 'B')
        v_2 = graph_vertex.UnWeightedGraphVertex(a_graph, 'C')
        v_3 = graph_vertex.UnWeightedGraphVertex(a_graph, 'D')
        v_4 = graph_vertex.UnWeightedGraphVertex(a_graph, 'E')

        # Add vertices to the graph.
        a_graph.add_vertex(v_0)
        a_graph.add_vertex(v_1)
        a_graph.add_vertex(v_2)
        a_graph.add_vertex(v_3)
        a_graph.add_vertex(v_4)

        # Add edges to the graph.
        a_graph.add_edge(v_0, v_1, 10)
        a_graph.add_edge(v_0, v_2, 20)
        a_graph.add_edge(v_0, v_3, 30)
        a_graph.add_edge(v_0, v_4, 40)
        a_graph.add_edge(v_1, v_2, 50)
        a_graph.add_edge(v_1, v_3, 60)
        a_graph.add_edge(v_1, v_4, 70)
        a_graph.add_edge(v_2, v_3, 80)
        a_graph.add_edge(v_2, v_4, 90)
        a_graph.add_edge(v_3, v_4, 100)

        # Create a reference graph used to compare the result after a vertex has been removed.
        g_ref = graph.UnDirectedWeightedGraph(4)

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
        a_graph.remove_vertex(v_3)
        self.assertEqual(g_ref, a_graph)

    def test_un_directed_weighted_graph_remove_vertex_v4(self):
        """
        Test method "remove_vertex".
        """
        # Create a graph from where a vertex should be removed.
        a_graph = graph.UnDirectedWeightedGraph(5)
        v_0 = graph_vertex.UnWeightedGraphVertex(a_graph, 'A')
        v_1 = graph_vertex.UnWeightedGraphVertex(a_graph, 'B')
        v_2 = graph_vertex.UnWeightedGraphVertex(a_graph, 'C')
        v_3 = graph_vertex.UnWeightedGraphVertex(a_graph, 'D')
        v_4 = graph_vertex.UnWeightedGraphVertex(a_graph, 'E')

        # Add vertices to the graph.
        a_graph.add_vertex(v_0)
        a_graph.add_vertex(v_1)
        a_graph.add_vertex(v_2)
        a_graph.add_vertex(v_3)
        a_graph.add_vertex(v_4)

        # Add edges to the graph.
        a_graph.add_edge(v_0, v_1, 10)
        a_graph.add_edge(v_0, v_2, 20)
        a_graph.add_edge(v_0, v_3, 30)
        a_graph.add_edge(v_0, v_4, 40)
        a_graph.add_edge(v_1, v_2, 50)
        a_graph.add_edge(v_1, v_3, 60)
        a_graph.add_edge(v_1, v_4, 70)
        a_graph.add_edge(v_2, v_3, 80)
        a_graph.add_edge(v_2, v_4, 90)
        a_graph.add_edge(v_3, v_4, 100)

        # Create a reference graph used to compare the result after a vertex has been removed.
        g_ref = graph.UnDirectedWeightedGraph(4)

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
        a_graph.remove_vertex(v_4)
        self.assertEqual(g_ref, a_graph)

    def test_un_directed_weighted_graph_remove_edge(self):
        """
        Test method "remove_edge".
        """
        # https://reference.wolfram.com/mathematica/ref/VertexDelete.html
        a_graph = graph.UnDirectedWeightedGraph(5)
        v_1 = graph_vertex.UnWeightedGraphVertex(a_graph, 'A')
        v_2 = graph_vertex.UnWeightedGraphVertex(a_graph, 'B')
        v_3 = graph_vertex.UnWeightedGraphVertex(a_graph, 'C')
        v_4 = graph_vertex.UnWeightedGraphVertex(a_graph, 'D')
        v_5 = graph_vertex.UnWeightedGraphVertex(a_graph, 'E')

        a_graph.add_vertex(v_1)
        a_graph.add_vertex(v_2)
        a_graph.add_vertex(v_3)
        a_graph.add_vertex(v_4)
        a_graph.add_vertex(v_5)

        a_graph.add_edge(v_1, v_2, 10)
        a_graph.add_edge(v_1, v_3, 20)
        a_graph.add_edge(v_1, v_4, 30)
        a_graph.add_edge(v_1, v_5, 40)
        a_graph.add_edge(v_2, v_3, 50)
        a_graph.add_edge(v_2, v_4, 60)
        a_graph.add_edge(v_2, v_5, 70)
        a_graph.add_edge(v_3, v_4, 80)
        a_graph.add_edge(v_3, v_5, 90)
        a_graph.add_edge(v_4, v_5, 100)

        a_graph.remove_edge(v_1, v_2)
        a_graph.remove_edge(v_1, v_3)
        a_graph.remove_edge(v_1, v_4)
        a_graph.remove_edge(v_1, v_5)
        a_graph.remove_edge(v_2, v_3)
        a_graph.remove_edge(v_2, v_4)
        a_graph.remove_edge(v_2, v_5)
        a_graph.remove_edge(v_3, v_4)
        a_graph.remove_edge(v_3, v_5)
        a_graph.remove_edge(v_4, v_5)
        ref = []
        res = a_graph.get_edges()
        self.assertEqual(ref, res)

    def test_un_directed_weighted_graph_is_connected(self):
        """
        Test method "is_connected".
        """
        self.assertTrue(self.g_1.is_connected())

    def test_un_directed_weighted_graph_is_cyclic(self):
        """
        Test method "is_cyclic".
        """
        a_graph = graph.UnDirectedWeightedGraph(4)
        v_1 = graph_vertex.UnWeightedGraphVertex(a_graph, 'A')
        v_2 = graph_vertex.UnWeightedGraphVertex(a_graph, 'B')
        v_3 = graph_vertex.UnWeightedGraphVertex(a_graph, 'C')
        v_4 = graph_vertex.UnWeightedGraphVertex(a_graph, 'D')

        a_graph.add_vertex(v_1)
        a_graph.add_vertex(v_2)
        a_graph.add_vertex(v_3)
        a_graph.add_vertex(v_4)

        a_graph.add_edge(v_1, v_2, 1)
        a_graph.add_edge(v_2, v_3, 2)
        a_graph.add_edge(v_1, v_3, 3)
        self.assertTrue(a_graph.is_cyclic())

    def test_un_directed_weighted_graph_is_cyclic_not(self):
        """
        Test method "is_cyclic" - inverted.
        """
        a_graph = graph.UnDirectedWeightedGraph(4)
        v_1 = graph_vertex.UnWeightedGraphVertex(a_graph, 'A')
        v_2 = graph_vertex.UnWeightedGraphVertex(a_graph, 'B')
        v_3 = graph_vertex.UnWeightedGraphVertex(a_graph, 'C')
        v_4 = graph_vertex.UnWeightedGraphVertex(a_graph, 'D')

        a_graph.add_vertex(v_1)
        a_graph.add_vertex(v_2)
        a_graph.add_vertex(v_3)
        a_graph.add_vertex(v_4)

        a_graph.add_edge(v_1, v_2, 1)
        a_graph.add_edge(v_2, v_3, 2)
        a_graph.add_edge(v_2, v_4, 3)
        self.assertFalse(a_graph.is_cyclic())

    def test_un_directed_weighted_graph_get_vertex_at_index(self):
        """
        Test method "get_vertex_at_index".
        """
        self.assertEqual(self.v_4, self.g_1.get_vertex_at_index(3))

    def test_un_directed_weighted_graph_get_emanating_edges(self):
        """
        Test method "get_emanating_edges".
        """
        res = []
        ref = []
        ref.append(self.e12)
        ref.append(self.e14)
        res = self.g_1.get_emanating_edges(self.v_1.get_vertex_number())
        self.assertEqual(ref, res)

    def test_un_directed_weighted_graph_get_incident_edges(self):
        """
        Test method "get_incident_edges".
        """
        res = []
        ref = []
        ref.append(self.e12)
        ref.append(self.e14)
        res = self.g_1.get_incident_edges(self.v_1.get_vertex_number())
        self.assertEqual(ref, res)

    def test_un_directed_weighted_graph_get_out_degree(self):
        """
        Test method "get_out_degree".
        """
        self.assertEqual(2, self.g_1.get_out_degree(self.v_1))

    def test_un_directed_weighted_graph_get_in_degree(self):
        """
        Test method "get_in_degree".
        """
        self.assertEqual(2, self.g_1.get_in_degree(self.v_1))

    def test_un_directed_weighted_graph_breadth_first_traversal(self):
        """
        Test method "breadth_first_traversal".
        """
        ref = []
        ref.append(self.g_1[0])
        ref.append(self.g_1[1])
        ref.append(self.g_1[3])
        ref.append(self.g_1[2])
        ref.append(self.g_1[4])
        ref.append(self.g_1[5])
        ref.append(self.g_1[6])
        visitor = vertex_visitor.VertexVisitor()
        self.g_1.breadth_first_traversal(visitor, 0)
        visited = visitor.get_visited()
        self.assertEqual(ref, visited)

    def test_un_directed_weighted_graph_depth_first_traversal(self):
        """
        Test method "depth_first_traversal".
        """
        ref = []
        ref.append(self.g_1[0])
        ref.append(self.g_1[1])
        ref.append(self.g_1[3])
        ref.append(self.g_1[4])
        ref.append(self.g_1[5])
        ref.append(self.g_1[6])
        ref.append(self.g_1[2])
        visitor = vertex_visitor.VertexVisitor()
        self.g_1.depth_first_traversal(visitor, 0)
        visited = visitor.get_visited()
        self.assertEqual(ref, visited)

    def test_un_directed_weighted_graph_classify_edges_cyclic(self):
        """
        Test classify edges - cyclic graph
        """
        # Create an undirected weighted cyclic graph
        a_graph = graph.UnDirectedWeightedGraph(4)
        v_1 = graph_vertex.UnWeightedGraphVertex(a_graph, 'A')
        v_2 = graph_vertex.UnWeightedGraphVertex(a_graph, 'B')
        v_3 = graph_vertex.UnWeightedGraphVertex(a_graph, 'C')
        v_4 = graph_vertex.UnWeightedGraphVertex(a_graph, 'D')

        a_graph.add_vertex(v_1)
        a_graph.add_vertex(v_2)
        a_graph.add_vertex(v_3)
        a_graph.add_vertex(v_4)

        a_graph.add_edge(v_1, v_2, 10)
        a_graph.add_edge(v_2, v_3, 20)
        a_graph.add_edge(v_1, v_3, 30)

        res = a_graph.classify_edges().get_edges()
        ref = dfs_edge_classification.DFSEdgeClassification(a_graph).get_edges()
        e12 = graph_edge.UnDirectedWeightedGraphEdge(a_graph, v_1, v_2, 10)
        e23 = graph_edge.UnDirectedWeightedGraphEdge(a_graph, v_2, v_3, 20)
        e13 = graph_edge.UnDirectedWeightedGraphEdge(a_graph, v_3, v_1, 30)
        ref[e12] = graph_edge.EdgeClassification.TREE_EDGE
        ref[e23] = graph_edge.EdgeClassification.TREE_EDGE
        ref[e13] = graph_edge.EdgeClassification.BACK_EDGE
        self.assertEqual(res, ref)

    def test_un_directed_weighted_graph_classify_edges_acyclic(self):
        """
        Test classify edges - acyclic graph.
        """
        # Create an undirected unweighted acyclic graph
        a_graph = graph.UnDirectedWeightedGraph(4)
        v_1 = graph_vertex.UnWeightedGraphVertex(a_graph, 'A')
        v_2 = graph_vertex.UnWeightedGraphVertex(a_graph, 'B')
        v_3 = graph_vertex.UnWeightedGraphVertex(a_graph, 'C')
        v_4 = graph_vertex.UnWeightedGraphVertex(a_graph, 'D')

        a_graph.add_vertex(v_1)
        a_graph.add_vertex(v_2)
        a_graph.add_vertex(v_3)
        a_graph.add_vertex(v_4)

        a_graph.add_edge(v_1, v_2, 10)
        a_graph.add_edge(v_2, v_3, 20)
        a_graph.add_edge(v_2, v_4, 30)

        res = a_graph.classify_edges().get_edges()
        ref = dfs_edge_classification.DFSEdgeClassification(a_graph).get_edges()
        e12 = graph_edge.UnDirectedWeightedGraphEdge(a_graph, v_1, v_2, 10)
        e23 = graph_edge.UnDirectedWeightedGraphEdge(a_graph, v_2, v_3, 20)
        e24 = graph_edge.UnDirectedWeightedGraphEdge(a_graph, v_2, v_4, 30)
        ref[e12] = graph_edge.EdgeClassification.TREE_EDGE
        ref[e23] = graph_edge.EdgeClassification.TREE_EDGE
        ref[e24] = graph_edge.EdgeClassification.TREE_EDGE
        self.assertEqual(res, ref)



