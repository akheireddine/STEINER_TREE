#!/usr/bin/env py.test

"""
Test GraphEdge class.
"""

import unittest

from py_alg_dat import graph
from py_alg_dat import graph_vertex

class TestGraphEdge(unittest.TestCase):

    """
    Test GraphEdge class.
    """

    def setUp(self):
        # Create an empty undirected unweighted graph.
        self.graph1 = graph.UnDirectedUnWeightedGraph(3)
        # Create graph vertices and add them to the graph.
        self.vertex1 = graph_vertex.GraphVertex(self.graph1, "A")
        self.vertex2 = graph_vertex.GraphVertex(self.graph1, "B")
        self.vertex3 = graph_vertex.GraphVertex(self.graph1, "C")
        self.graph1.add_vertex(self.vertex1)
        self.graph1.add_vertex(self.vertex2)
        self.graph1.add_vertex(self.vertex3)
        # Create graph edges and add them to the graph.
        self.graph1.add_edge(self.vertex1, self.vertex2)
        self.graph1.add_edge(self.vertex2, self.vertex3)
        self.graph1.add_edge(self.vertex1, self.vertex3)
        # Extract the edges.
        self.e12_g1 = self.graph1.get_edge(self.vertex1, self.vertex2)
        self.e21_g1 = self.graph1.get_edge(self.vertex2, self.vertex1)
        self.e23_g1 = self.graph1.get_edge(self.vertex2, self.vertex3)
        self.e32_g1 = self.graph1.get_edge(self.vertex3, self.vertex2)
        self.e13_g1 = self.graph1.get_edge(self.vertex1, self.vertex3)
        self.e31_g1 = self.graph1.get_edge(self.vertex3, self.vertex1)

    def test_graph_edge_equal(self):
        """
        Test operator "equal".
        """
        a_graph = graph.UnDirectedUnWeightedGraph(3)
        vertex1 = graph_vertex.GraphVertex(a_graph, "A")
        vertex2 = graph_vertex.GraphVertex(a_graph, "B")
        vertex3 = graph_vertex.GraphVertex(a_graph, "C")
        a_graph.add_vertex(vertex1)
        a_graph.add_vertex(vertex2)
        a_graph.add_vertex(vertex3)
        a_graph.add_edge(vertex1, vertex2)
        a_graph.add_edge(vertex2, vertex3)
        a_graph.add_edge(vertex1, vertex3)
        e12_g = a_graph.get_edge(vertex1, vertex2)
        e21_g = a_graph.get_edge(vertex2, vertex1)
        e23_g = a_graph.get_edge(vertex2, vertex3)
        e32_g = a_graph.get_edge(vertex3, vertex2)
        e13_g = a_graph.get_edge(vertex1, vertex3)
        e31_g = a_graph.get_edge(vertex3, vertex1)
        test1 = e12_g == self.e12_g1
        test2 = e21_g == self.e21_g1
        test3 = e23_g == self.e23_g1
        test4 = e32_g == self.e32_g1
        test5 = e13_g == self.e13_g1
        test6 = e31_g == self.e31_g1
        test = test1 == test2 == test3 == test4 == test5 == test6
        self.assertTrue(test)

    def test_graph_edge_not_equal(self):
        """
        Test operator "inequal".
        """
        a_graph = graph.UnDirectedUnWeightedGraph(3)
        vertex1 = graph_vertex.GraphVertex(a_graph, "AA")
        vertex2 = graph_vertex.GraphVertex(a_graph, "BB")
        vertex3 = graph_vertex.GraphVertex(a_graph, "CC")
        a_graph.add_vertex(vertex1)
        a_graph.add_vertex(vertex2)
        a_graph.add_vertex(vertex3)
        a_graph.add_edge(vertex1, vertex2)
        a_graph.add_edge(vertex2, vertex3)
        a_graph.add_edge(vertex1, vertex3)
        e12_g = a_graph.get_edge(vertex1, vertex2)
        e21_g = a_graph.get_edge(vertex2, vertex1)
        e23_g = a_graph.get_edge(vertex2, vertex3)
        e32_g = a_graph.get_edge(vertex3, vertex2)
        e13_g = a_graph.get_edge(vertex1, vertex3)
        e31_g = a_graph.get_edge(vertex3, vertex1)
        test1 = e12_g == self.e12_g1
        test2 = e21_g == self.e21_g1
        test3 = e23_g == self.e23_g1
        test4 = e32_g == self.e32_g1
        test5 = e13_g == self.e13_g1
        test6 = e31_g == self.e31_g1
        test = test1 and test2 and test3 and test4 and test5 and test6
        self.assertFalse(test)

    def test_graph_edge_is_directed(self):
        """
        Test method "is_directed".
        """
        self.assertFalse(self.e12_g1.is_directed())

    def test_graph_edge_get_v0(self):
        """
        Test method "get_v0".
        """
        self.assertEqual(self.vertex1, self.e12_g1.get_head_vertex())

    def test_graph_edge_get_v1(self):
        """
        Test method "get_v1"-
        """
        self.assertEqual(self.vertex2, self.e12_g1.get_tail_vertex())

    def test_graph_edge_get_mate(self):
        """
        Test method "get_mate".
        """
        test1 = self.vertex2 == self.e12_g1.get_mate(self.vertex1)
        test2 = self.vertex1 == self.e12_g1.get_mate(self.vertex2)
        test = test1 == test2
        self.assertTrue(test)
