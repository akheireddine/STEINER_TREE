#!/usr/bin/env py.test

"""
Test of various graph algorithms.
"""

import unittest

from py_alg_dat import array_list
from py_alg_dat import entry
from py_alg_dat import graph
from py_alg_dat import graph_algorithms
from py_alg_dat import graph_edge
from py_alg_dat import graph_path
from py_alg_dat import graph_vertex
from py_alg_dat import minimum_spanning_tree

class TestGraphAlgorithms(unittest.TestCase):

    """
    Test of various graph algorithms.
    """

    def setUp(self):
        # Graph from http://en.wikipedia.org/wiki/Prim%27s_algorithm
        self.graph1 = graph.UnDirectedWeightedGraph(7)

        self.v1_g1 = graph_vertex.UnWeightedGraphVertex(self.graph1, "A")
        self.v2_g1 = graph_vertex.UnWeightedGraphVertex(self.graph1, "B")
        self.v3_g1 = graph_vertex.UnWeightedGraphVertex(self.graph1, "C")
        self.v4_g1 = graph_vertex.UnWeightedGraphVertex(self.graph1, "D")
        self.v5_g1 = graph_vertex.UnWeightedGraphVertex(self.graph1, "E")
        self.v6_g1 = graph_vertex.UnWeightedGraphVertex(self.graph1, "F")
        self.v7_g1 = graph_vertex.UnWeightedGraphVertex(self.graph1, "G")

        self.graph1.add_vertex(self.v1_g1)
        self.graph1.add_vertex(self.v2_g1)
        self.graph1.add_vertex(self.v3_g1)
        self.graph1.add_vertex(self.v4_g1)
        self.graph1.add_vertex(self.v5_g1)
        self.graph1.add_vertex(self.v6_g1)
        self.graph1.add_vertex(self.v7_g1)

        self.graph1.add_edge(self.v1_g1, self.v2_g1, 7)    # ( A - B, 7 )
        self.graph1.add_edge(self.v1_g1, self.v4_g1, 5)    # ( A - D, 5 )
        self.graph1.add_edge(self.v2_g1, self.v3_g1, 8)    # ( B - C, 8 )
        self.graph1.add_edge(self.v2_g1, self.v4_g1, 9)    # ( B - D, 9 )
        self.graph1.add_edge(self.v2_g1, self.v5_g1, 7)    # ( B - E, 7 )
        self.graph1.add_edge(self.v3_g1, self.v5_g1, 5)    # ( C - E, 5 )
        self.graph1.add_edge(self.v4_g1, self.v5_g1, 15)   # ( D - E, 15 )
        self.graph1.add_edge(self.v4_g1, self.v6_g1, 6)    # ( D - F, 6 )
        self.graph1.add_edge(self.v5_g1, self.v6_g1, 8)    # ( E - F, 8 )
        self.graph1.add_edge(self.v5_g1, self.v7_g1, 9)    # ( E - G, 9 )
        self.graph1.add_edge(self.v6_g1, self.v7_g1, 11)   # ( F - G, 11 )

        # Directed weighted graph from:
        # http://compalg.inf.elte.hu/~tony/Oktatas/TDK/FINAL/Chap%2013.PDF
        self.graph2 = graph.DirectedWeightedGraph(7)

        self.v0_g2 = graph_vertex.UnWeightedGraphVertex(self.graph2, "A")
        self.v1_g2 = graph_vertex.UnWeightedGraphVertex(self.graph2, "B")
        self.v2_g2 = graph_vertex.UnWeightedGraphVertex(self.graph2, "C")
        self.v3_g2 = graph_vertex.UnWeightedGraphVertex(self.graph2, "D")
        self.v4_g2 = graph_vertex.UnWeightedGraphVertex(self.graph2, "E")
        self.v5_g2 = graph_vertex.UnWeightedGraphVertex(self.graph2, "F")
        self.v6_g2 = graph_vertex.UnWeightedGraphVertex(self.graph2, "G")

        self.graph2.add_vertex(self.v0_g2)
        self.graph2.add_vertex(self.v1_g2)
        self.graph2.add_vertex(self.v2_g2)
        self.graph2.add_vertex(self.v3_g2)
        self.graph2.add_vertex(self.v4_g2)
        self.graph2.add_vertex(self.v5_g2)
        self.graph2.add_vertex(self.v6_g2)

        self.graph2.add_edge(self.v0_g2, self.v1_g2, 7)   # ( A - B, 7 )
        self.graph2.add_edge(self.v1_g2, self.v2_g2, 2)   # ( B - C, 2 )
        self.graph2.add_edge(self.v1_g2, self.v6_g2, 3)   # ( B - G, 3 )
        self.graph2.add_edge(self.v2_g2, self.v3_g2, 2)   # ( C - D, 2 )
        self.graph2.add_edge(self.v2_g2, self.v6_g2, 4)   # ( C - G, 4 )
        self.graph2.add_edge(self.v3_g2, self.v4_g2, 5)   # ( D - E, 5 )
        self.graph2.add_edge(self.v3_g2, self.v6_g2, 1)   # ( D - G, 5 )
        self.graph2.add_edge(self.v4_g2, self.v5_g2, 6)   # ( E - F, 6 )
        self.graph2.add_edge(self.v5_g2, self.v0_g2, 1)   # ( F - A, 1 )
        self.graph2.add_edge(self.v5_g2, self.v6_g2, 4)   # ( F - G, 4 )
        self.graph2.add_edge(self.v6_g2, self.v0_g2, 7)   # ( G - A, 7 )
        self.graph2.add_edge(self.v6_g2, self.v4_g2, 1)   # ( G - E, 1 )

    def test_graph_algorithms_prim_0(self):
        """
        Test of Prims algorithm.
        """
        elem1 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v2_g1, self.v1_g1, 7)
        elem2 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v3_g1, self.v5_g1, 5)
        elem3 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v4_g1, self.v1_g1, 5)
        elem4 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v5_g1, self.v2_g1, 7)
        elem5 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v6_g1, self.v4_g1, 6)
        elem6 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v7_g1, self.v5_g1, 9)
        mst_ref = minimum_spanning_tree.MinimumSpanningTree(self.graph1)
        mst_ref.add_edge(elem1)
        mst_ref.add_edge(elem2)
        mst_ref.add_edge(elem3)
        mst_ref.add_edge(elem4)
        mst_ref.add_edge(elem5)
        mst_ref.add_edge(elem6)
        mst_res = graph_algorithms.GraphAlgorithms.prims_algorithm(self.graph1, self.v1_g1)
        self.assertEqual(mst_ref, mst_res)

    def test_graph_algorithms_prim_1(self):
        """
        Test of Prims algorithm.
        """
        elem1 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v1_g1, self.v2_g1, 7)
        elem2 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v3_g1, self.v5_g1, 5)
        elem3 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v4_g1, self.v1_g1, 5)
        elem4 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v5_g1, self.v2_g1, 7)
        elem5 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v6_g1, self.v4_g1, 6)
        elem6 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v7_g1, self.v5_g1, 9)
        mst_ref = minimum_spanning_tree.MinimumSpanningTree(self.graph1)
        mst_ref.add_edge(elem1)
        mst_ref.add_edge(elem2)
        mst_ref.add_edge(elem3)
        mst_ref.add_edge(elem4)
        mst_ref.add_edge(elem5)
        mst_ref.add_edge(elem6)
        mst_res = graph_algorithms.GraphAlgorithms.prims_algorithm(self.graph1, self.v2_g1)
        self.assertEqual(mst_ref, mst_res)

    def test_graph_algorithms_prim_2(self):
        """
        Test of Prims algorithm.
        """
        elem1 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v1_g1, self.v2_g1, 7)
        elem2 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v2_g1, self.v5_g1, 7)
        elem3 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v4_g1, self.v1_g1, 5)
        elem4 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v5_g1, self.v3_g1, 5)
        elem5 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v6_g1, self.v4_g1, 6)
        elem6 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v7_g1, self.v5_g1, 9)
        mst_ref = minimum_spanning_tree.MinimumSpanningTree(self.graph1)
        mst_ref.add_edge(elem1)
        mst_ref.add_edge(elem2)
        mst_ref.add_edge(elem3)
        mst_ref.add_edge(elem4)
        mst_ref.add_edge(elem5)
        mst_ref.add_edge(elem6)
        mst_res = graph_algorithms.GraphAlgorithms.prims_algorithm(self.graph1, self.v3_g1)
        self.assertEqual(mst_ref, mst_res)

    def test_graph_algorithms_prim_3(self):
        """
        Test of Prims algorithm.
        """
        elem1 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v1_g1, self.v4_g1, 5)
        elem2 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v2_g1, self.v1_g1, 7)
        elem3 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v3_g1, self.v5_g1, 5)
        elem4 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v5_g1, self.v2_g1, 7)
        elem5 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v6_g1, self.v4_g1, 6)
        elem6 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v7_g1, self.v5_g1, 9)
        mst_ref = minimum_spanning_tree.MinimumSpanningTree(self.graph1)
        mst_ref.add_edge(elem1)
        mst_ref.add_edge(elem2)
        mst_ref.add_edge(elem3)
        mst_ref.add_edge(elem4)
        mst_ref.add_edge(elem5)
        mst_ref.add_edge(elem6)
        mst_res = graph_algorithms.GraphAlgorithms.prims_algorithm(self.graph1, self.v4_g1)
        self.assertEqual(mst_ref, mst_res)

    def test_graph_algorithms_prim_4(self):
        """
        Test of Prims algorithm.
        """
        elem1 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v1_g1, self.v2_g1, 7)
        elem2 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v2_g1, self.v5_g1, 7)
        elem3 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v3_g1, self.v5_g1, 5)
        elem4 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v4_g1, self.v1_g1, 5)
        elem5 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v6_g1, self.v4_g1, 6)
        elem6 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v7_g1, self.v5_g1, 9)
        mst_ref = minimum_spanning_tree.MinimumSpanningTree(self.graph1)
        mst_ref.add_edge(elem1)
        mst_ref.add_edge(elem2)
        mst_ref.add_edge(elem3)
        mst_ref.add_edge(elem4)
        mst_ref.add_edge(elem5)
        mst_ref.add_edge(elem6)
        mst_res = graph_algorithms.GraphAlgorithms.prims_algorithm(self.graph1, self.v5_g1)
        self.assertEqual(mst_ref, mst_res)

    def test_graph_algorithms_prim_5(self):
        """
        Test of Prims algorithm.
        """
        elem1 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v1_g1, self.v4_g1, 5)
        elem2 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v2_g1, self.v1_g1, 7)
        elem3 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v3_g1, self.v5_g1, 5)
        elem4 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v4_g1, self.v6_g1, 6)
        elem5 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v5_g1, self.v2_g1, 7)
        elem6 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v7_g1, self.v5_g1, 9)
        mst_ref = minimum_spanning_tree.MinimumSpanningTree(self.graph1)
        mst_ref.add_edge(elem1)
        mst_ref.add_edge(elem2)
        mst_ref.add_edge(elem3)
        mst_ref.add_edge(elem4)
        mst_ref.add_edge(elem5)
        mst_ref.add_edge(elem6)
        mst_res = graph_algorithms.GraphAlgorithms.prims_algorithm(self.graph1, self.v6_g1)
        self.assertEqual(mst_ref, mst_res)

    def test_graph_algorithms_prim_6(self):
        """
        Test of Prims algorithm.
        """
        elem1 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v1_g1, self.v2_g1, 7)
        elem2 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v2_g1, self.v5_g1, 7)
        elem3 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v3_g1, self.v5_g1, 5)
        elem4 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v4_g1, self.v1_g1, 5)
        elem5 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v5_g1, self.v7_g1, 9)
        elem6 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v6_g1, self.v4_g1, 6)
        mst_ref = minimum_spanning_tree.MinimumSpanningTree(self.graph1)
        mst_ref.add_edge(elem1)
        mst_ref.add_edge(elem2)
        mst_ref.add_edge(elem3)
        mst_ref.add_edge(elem4)
        mst_ref.add_edge(elem5)
        mst_ref.add_edge(elem6)
        mst_res = graph_algorithms.GraphAlgorithms.prims_algorithm(self.graph1, self.v7_g1)
        self.assertEqual(mst_ref, mst_res)

    def test_graph_algorithms_kruskal(self):
        """
        Test of Kruskals algorithm.
        """
        elem1 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v1_g1, self.v4_g1, 5)
        elem2 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v3_g1, self.v5_g1, 5)
        elem3 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v6_g1, self.v4_g1, 6)
        elem4 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v1_g1, self.v2_g1, 7)
        elem5 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v2_g1, self.v5_g1, 7)
        elem6 = graph_edge.UnDirectedWeightedGraphEdge(self.graph1, self.v5_g1, self.v7_g1, 9)
        mst_ref = minimum_spanning_tree.MinimumSpanningTree(self.graph1)
        mst_ref.add_edge(elem1)
        mst_ref.add_edge(elem2)
        mst_ref.add_edge(elem3)
        mst_ref.add_edge(elem4)
        mst_ref.add_edge(elem5)
        mst_ref.add_edge(elem6)
        mst_res = graph_algorithms.GraphAlgorithms.kruskals_algorithm(self.graph1)
        self.assertEqual(mst_ref, mst_res)

    def test_graph_algorithms_dijkstra_v0(self):
        """
        Test of Dijkstras algorithm.
        """
        e01 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v0_g2, self.v1_g2, 7)
        e12 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v1_g2, self.v2_g2, 2)
        e23 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v2_g2, self.v3_g2, 2)
        e64 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v6_g2, self.v4_g2, 1)
        e45 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v4_g2, self.v5_g2, 6)
        e16 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v1_g2, self.v6_g2, 3)

        elem0 = entry.Entry(True, 0, None, None)
        elem1 = entry.Entry(True, 7, self.v0_g2, e01)
        elem2 = entry.Entry(True, 9, self.v1_g2, e12)
        elem3 = entry.Entry(True, 11, self.v2_g2, e23)
        elem4 = entry.Entry(True, 11, self.v6_g2, e64)
        elem5 = entry.Entry(True, 17, self.v4_g2, e45)
        elem6 = entry.Entry(True, 10, self.v1_g2, e16)

        ref = array_list.ArrayList(self.graph2.get_number_of_vertices())
        ref[0] = elem0
        ref[1] = elem1
        ref[2] = elem2
        ref[3] = elem3
        ref[4] = elem4
        ref[5] = elem5
        ref[6] = elem6
        res = graph_algorithms.GraphAlgorithms.dijkstras_algorithm(self.graph2, self.v0_g2)
        self.assertEqual(ref, res)

    def test_graph_algorithms_dijkstra_v1(self):
        """
        Test of Dijkstras algorithm.
        """
        e60 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v6_g2, self.v0_g2, 7)
        e21 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v1_g2, self.v2_g2, 2)
        e23 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v2_g2, self.v3_g2, 2)
        e64 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v6_g2, self.v4_g2, 1)
        e45 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v4_g2, self.v5_g2, 6)
        e16 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v1_g2, self.v6_g2, 3)

        elem0 = entry.Entry(True, 10, self.v6_g2, e60)
        elem1 = entry.Entry(True, 0, None, None)
        elem2 = entry.Entry(True, 2, self.v1_g2, e21)
        elem3 = entry.Entry(True, 4, self.v2_g2, e23)
        elem4 = entry.Entry(True, 4, self.v6_g2, e64)
        elem5 = entry.Entry(True, 10, self.v4_g2, e45)
        elem6 = entry.Entry(True, 3, self.v1_g2, e16)

        ref = array_list.ArrayList(self.graph2.get_number_of_vertices())
        ref[0] = elem0
        ref[1] = elem1
        ref[2] = elem2
        ref[3] = elem3
        ref[4] = elem4
        ref[5] = elem5
        ref[6] = elem6
        res = graph_algorithms.GraphAlgorithms.dijkstras_algorithm(self.graph2, self.v1_g2)
        self.assertEqual(ref, res)

    def test_graph_algorithms_dijkstra_v2(self):
        """
        Test of Dijkstras algorithm.
        """
        e60 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v6_g2, self.v0_g2, 7)
        e01 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v0_g2, self.v1_g2, 7)
        e23 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v2_g2, self.v3_g2, 2)
        e64 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v6_g2, self.v4_g2, 1)
        e45 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v4_g2, self.v5_g2, 6)
        e36 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v3_g2, self.v6_g2, 1)

        elem0 = entry.Entry(True, 10, self.v6_g2, e60)
        elem1 = entry.Entry(True, 17, self.v0_g2, e01)
        elem2 = entry.Entry(True, 0, None, None)
        elem3 = entry.Entry(True, 2, self.v2_g2, e23)
        elem4 = entry.Entry(True, 4, self.v6_g2, e64)
        elem5 = entry.Entry(True, 10, self.v4_g2, e45)
        elem6 = entry.Entry(True, 3, self.v3_g2, e36)

        ref = array_list.ArrayList(self.graph2.get_number_of_vertices())
        ref[0] = elem0
        ref[1] = elem1
        ref[2] = elem2
        ref[3] = elem3
        ref[4] = elem4
        ref[5] = elem5
        ref[6] = elem6
        res = graph_algorithms.GraphAlgorithms.dijkstras_algorithm(self.graph2, self.v2_g2)
        self.assertEqual(ref, res)

    def test_graph_algorithms_dijkstra_v3(self):
        """
        Test of Dijkstras algorithm.
        """
        e60 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v6_g2, self.v0_g2, 7)
        e01 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v0_g2, self.v1_g2, 7)
        e12 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v1_g2, self.v2_g2, 2)
        e64 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v6_g2, self.v4_g2, 1)
        e45 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v4_g2, self.v5_g2, 6)
        e36 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v3_g2, self.v6_g2, 1)

        elem0 = entry.Entry(True, 8, self.v6_g2, e60)
        elem1 = entry.Entry(True, 15, self.v0_g2, e01)
        elem2 = entry.Entry(True, 17, self.v1_g2, e12)
        elem3 = entry.Entry(True, 0, None, None)
        elem4 = entry.Entry(True, 2, self.v6_g2, e64)
        elem5 = entry.Entry(True, 8, self.v4_g2, e45)
        elem6 = entry.Entry(True, 1, self.v3_g2, e36)

        ref = array_list.ArrayList(self.graph2.get_number_of_vertices())
        ref[0] = elem0
        ref[1] = elem1
        ref[2] = elem2
        ref[3] = elem3
        ref[4] = elem4
        ref[5] = elem5
        ref[6] = elem6
        res = graph_algorithms.GraphAlgorithms.dijkstras_algorithm(self.graph2, self.v3_g2)
        self.assertEqual(ref, res)

    def test_graph_algorithms_dijkstra_v4(self):
        """
        Test of Dijkstras algorithm.
        """
        e50 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v5_g2, self.v0_g2, 1)
        e01 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v0_g2, self.v1_g2, 7)
        e12 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v1_g2, self.v2_g2, 2)
        e23 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v2_g2, self.v3_g2, 2)
        e45 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v4_g2, self.v5_g2, 6)
        e56 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v5_g2, self.v6_g2, 4)

        elem0 = entry.Entry(True, 7, self.v5_g2, e50)
        elem1 = entry.Entry(True, 14, self.v0_g2, e01)
        elem2 = entry.Entry(True, 16, self.v1_g2, e12)
        elem3 = entry.Entry(True, 18, self.v2_g2, e23)
        elem4 = entry.Entry(True, 0, None, None)
        elem5 = entry.Entry(True, 6, self.v4_g2, e45)
        elem6 = entry.Entry(True, 10, self.v5_g2, e56)

        ref = array_list.ArrayList(self.graph2.get_number_of_vertices())
        ref[0] = elem0
        ref[1] = elem1
        ref[2] = elem2
        ref[3] = elem3
        ref[4] = elem4
        ref[5] = elem5
        ref[6] = elem6
        res = graph_algorithms.GraphAlgorithms.dijkstras_algorithm(self.graph2, self.v4_g2)
        self.assertEqual(ref, res)

    def test_graph_algorithms_dijkstra_v5(self):
        """
        Test of Dijkstras algorithm.
        """
        e50 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v5_g2, self.v0_g2, 1)
        e01 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v0_g2, self.v1_g2, 7)
        e12 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v1_g2, self.v2_g2, 2)
        e23 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v2_g2, self.v3_g2, 2)
        e64 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v6_g2, self.v4_g2, 1)
        e56 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v5_g2, self.v6_g2, 4)

        elem0 = entry.Entry(True, 1, self.v5_g2, e50)
        elem1 = entry.Entry(True, 8, self.v0_g2, e01)
        elem2 = entry.Entry(True, 10, self.v1_g2, e12)
        elem3 = entry.Entry(True, 12, self.v2_g2, e23)
        elem4 = entry.Entry(True, 5, self.v6_g2, e64)
        elem5 = entry.Entry(True, 0, None, None)
        elem6 = entry.Entry(True, 4, self.v5_g2, e56)

        ref = array_list.ArrayList(self.graph2.get_number_of_vertices())
        ref[0] = elem0
        ref[1] = elem1
        ref[2] = elem2
        ref[3] = elem3
        ref[4] = elem4
        ref[5] = elem5
        ref[6] = elem6
        res = graph_algorithms.GraphAlgorithms.dijkstras_algorithm(self.graph2, self.v5_g2)
        self.assertEqual(ref, res)

    def test_graph_algorithms_dijkstra_v6(self):
        """
        Test of Dijkstras algorithm.
        """
        e60 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v6_g2, self.v0_g2, 7)
        e01 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v0_g2, self.v1_g2, 7)
        e12 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v1_g2, self.v2_g2, 2)
        e23 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v2_g2, self.v3_g2, 2)
        e64 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v6_g2, self.v4_g2, 1)
        e45 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v4_g2, self.v5_g2, 6)

        elem0 = entry.Entry(True, 7, self.v6_g2, e60)
        elem1 = entry.Entry(True, 14, self.v0_g2, e01)
        elem2 = entry.Entry(True, 16, self.v1_g2, e12)
        elem3 = entry.Entry(True, 18, self.v2_g2, e23)
        elem4 = entry.Entry(True, 1, self.v6_g2, e64)
        elem5 = entry.Entry(True, 7, self.v4_g2, e45)
        elem6 = entry.Entry(True, 0, None, None)

        ref = array_list.ArrayList(self.graph2.get_number_of_vertices())
        ref[0] = elem0
        ref[1] = elem1
        ref[2] = elem2
        ref[3] = elem3
        ref[4] = elem4
        ref[5] = elem5
        ref[6] = elem6
        res = graph_algorithms.GraphAlgorithms.dijkstras_algorithm(self.graph2, self.v6_g2)
        self.assertEqual(ref, res)

    def test_graph_algorithms_shortest_path_v0_v1(self):
        """
        Test of shortest path algorithm.
        """
        ref = graph_path.GraphPath(self.graph2)

        # Reference vertices are added in reverse order
        # in order to make comparison easier.
        ref.add_vertex(self.v1_g2)
        ref.add_vertex(self.v0_g2)

        e01 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v0_g2, self.v1_g2, 7)

        # Reference edges are added in reverse order
        # in order to make comparison easier.
        ref.add_edge(e01)

        res = graph_algorithms.GraphAlgorithms.shortest_path(self.graph2, self.v0_g2, self.v1_g2)
        self.assertEqual(ref, res)

    def test_graph_algorithms_shortest_path_v0_v2(self):
        """
        Test of shortest path algorithm.
        """
        ref = graph_path.GraphPath(self.graph2)

        # Reference vertices are added in reverse order
        # in order to make comparison easier.
        ref.add_vertex(self.v2_g2)
        ref.add_vertex(self.v1_g2)
        ref.add_vertex(self.v0_g2)

        e01 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v0_g2, self.v1_g2, 7)
        e02 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v1_g2, self.v2_g2, 2)

        # Reference edges are added in reverse order
        # in order to make comparison easier.
        ref.add_edge(e02)
        ref.add_edge(e01)

        res = graph_algorithms.GraphAlgorithms.shortest_path(self.graph2, self.v0_g2, self.v2_g2)
        self.assertEqual(ref, res)

    def test_graph_algorithms_shortest_path_v0_v3(self):
        """
        Test of shortest path algorithm.
        """
        ref = graph_path.GraphPath(self.graph2)

        # Reference vertices are added in reverse order
        # in order to make comparison easier.
        ref.add_vertex(self.v3_g2)
        ref.add_vertex(self.v2_g2)
        ref.add_vertex(self.v1_g2)
        ref.add_vertex(self.v0_g2)

        e01 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v0_g2, self.v1_g2, 7)
        e02 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v1_g2, self.v2_g2, 2)
        e03 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v2_g2, self.v3_g2, 2)

        # Reference edges are added in reverse order
        # in order to make comparison easier.
        ref.add_edge(e03)
        ref.add_edge(e02)
        ref.add_edge(e01)

        res = graph_algorithms.GraphAlgorithms.shortest_path(self.graph2, self.v0_g2, self.v3_g2)
        self.assertEqual(ref, res)

    def test_graph_algorithms_shortest_path_v0_v4(self):
        """
        Test of shortest path algorithm.
        """
        ref = graph_path.GraphPath(self.graph2)

        # Reference vertices are added in reverse order
        # in order to make comparison easier.
        ref.add_vertex(self.v4_g2)
        ref.add_vertex(self.v6_g2)
        ref.add_vertex(self.v1_g2)
        ref.add_vertex(self.v0_g2)

        e01 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v0_g2, self.v1_g2, 7)
        e16 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v1_g2, self.v6_g2, 3)
        e64 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v6_g2, self.v4_g2, 1)

        # Reference edges are added in reverse order
        # in order to make comparison easier.
        ref.add_edge(e64)
        ref.add_edge(e16)
        ref.add_edge(e01)

        res = graph_algorithms.GraphAlgorithms.shortest_path(self.graph2, self.v0_g2, self.v4_g2)
        self.assertEqual(ref, res)

    def test_graph_algorithms_shortest_path_v0_v5(self):
        """
        Test of shortest path algorithm.
        """
        ref = graph_path.GraphPath(self.graph2)

        # Reference vertices are added in reverse order
        # in order to make comparison easier.
        ref.add_vertex(self.v5_g2)
        ref.add_vertex(self.v4_g2)
        ref.add_vertex(self.v6_g2)
        ref.add_vertex(self.v1_g2)
        ref.add_vertex(self.v0_g2)

        e01 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v0_g2, self.v1_g2, 7)
        e16 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v1_g2, self.v6_g2, 3)
        e64 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v6_g2, self.v4_g2, 1)
        e65 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v4_g2, self.v5_g2, 6)

        # Reference edges are added in reverse order
        # in order to make comparison easier.
        ref.add_edge(e65)
        ref.add_edge(e64)
        ref.add_edge(e16)
        ref.add_edge(e01)

        res = graph_algorithms.GraphAlgorithms.shortest_path(self.graph2, self.v0_g2, self.v5_g2)
        self.assertEqual(ref, res)

    def test_graph_algorithms_shortest_path_v0_v6(self):
        """
        Test of shortest path algorithm.
        """
        ref = graph_path.GraphPath(self.graph2)

        # Reference vertices are added in reverse order
        # in order to make comparison easier.
        ref.add_vertex(self.v6_g2)
        ref.add_vertex(self.v1_g2)
        ref.add_vertex(self.v0_g2)

        e01 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v0_g2, self.v1_g2, 7)
        e16 = graph_edge.DirectedWeightedGraphEdge(self.graph2, self.v1_g2, self.v6_g2, 3)

        # Reference edges are added in reverse order
        # in order to make comparison easier.
        ref.add_edge(e16)
        ref.add_edge(e01)

        res = graph_algorithms.GraphAlgorithms.shortest_path(self.graph2, self.v0_g2, self.v6_g2)
        self.assertEqual(ref, res)

    def test_graph_algorithms_bellman_ford(self):
        """
        Test of Belleman-Fords algorithm.
        """
        # Create directed weighted graph - reference: http://algs4.cs.princeton.edu
        test_graph = graph.DirectedWeightedGraph(8)

        vertex0 = graph_vertex.UnWeightedGraphVertex(test_graph, "0")
        vertex1 = graph_vertex.UnWeightedGraphVertex(test_graph, "1")
        vertex2 = graph_vertex.UnWeightedGraphVertex(test_graph, "2")
        vertex3 = graph_vertex.UnWeightedGraphVertex(test_graph, "3")
        vertex4 = graph_vertex.UnWeightedGraphVertex(test_graph, "4")
        vertex5 = graph_vertex.UnWeightedGraphVertex(test_graph, "5")
        vertex6 = graph_vertex.UnWeightedGraphVertex(test_graph, "6")
        vertex7 = graph_vertex.UnWeightedGraphVertex(test_graph, "7")

        test_graph.add_vertex(vertex0)
        test_graph.add_vertex(vertex1)
        test_graph.add_vertex(vertex2)
        test_graph.add_vertex(vertex3)
        test_graph.add_vertex(vertex4)
        test_graph.add_vertex(vertex5)
        test_graph.add_vertex(vertex6)
        test_graph.add_vertex(vertex7)

        test_graph.add_edge(vertex0, vertex1, 5)
        test_graph.add_edge(vertex0, vertex4, 9)
        test_graph.add_edge(vertex0, vertex7, 8)
        test_graph.add_edge(vertex1, vertex2, 12)
        test_graph.add_edge(vertex1, vertex3, 15)
        test_graph.add_edge(vertex1, vertex7, 4)
        test_graph.add_edge(vertex2, vertex3, 3)
        test_graph.add_edge(vertex2, vertex6, 11)
        test_graph.add_edge(vertex3, vertex6, 9)
        test_graph.add_edge(vertex4, vertex5, 4)
        test_graph.add_edge(vertex4, vertex6, 20)
        test_graph.add_edge(vertex4, vertex7, 5)
        test_graph.add_edge(vertex5, vertex2, 1)
        test_graph.add_edge(vertex5, vertex6, 13)
        test_graph.add_edge(vertex7, vertex2, 7)
        test_graph.add_edge(vertex7, vertex5, 6)

        # Find the shortest path from vertex v0 to all other vertices
        # using Bellman-Ford's algorithm.
        res_tuple = graph_algorithms.GraphAlgorithms.bellman_ford_algorithm(test_graph, vertex0)

        # Create a reference map holding the result.
        ref_map = {}
        ref_map[vertex7] = 8
        ref_map[vertex1] = 5
        ref_map[vertex2] = 14
        ref_map[vertex3] = 17
        ref_map[vertex6] = 25
        ref_map[vertex0] = 0
        ref_map[vertex5] = 13
        ref_map[vertex4] = 9
        ref_directed = False
        ref_tuple = (ref_directed, ref_map)
        self.assertEqual(ref_tuple, res_tuple)

