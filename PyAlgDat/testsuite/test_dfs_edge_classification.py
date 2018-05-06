#!/usr/bin/env py.test

"""
Test of edge classification in graph structures.
"""

import unittest
from py_alg_dat import dfs_edge_classification
from py_alg_dat import graph
from py_alg_dat import graph_edge
from py_alg_dat import graph_vertex

class TestDFSEdgeClassification(unittest.TestCase):

    """
    Test of edge classification in graph structures.

    NOTE: When performing edge classification in a graph the order of
    adding vertices matters.

    The order of adding vertices matter, since it affects the structure,
    of the adjacency list -and thereby the order in which DFS traverses
    the vertices, causing edge classification to be dependent of the
    order in which the vertices are added to the graph.
    """

    def setUp(self):
        # Create an undirected cyclic graph
        self.g_undirected_cyclic = graph.UnDirectedGraph(4)
        self.a_g_undirected_cyclic = graph_vertex.UnWeightedGraphVertex(self.g_undirected_cyclic, 'a')
        self.b_g_undirected_cyclic = graph_vertex.UnWeightedGraphVertex(self.g_undirected_cyclic, 'b')
        self.c_g_undirected_cyclic = graph_vertex.UnWeightedGraphVertex(self.g_undirected_cyclic, 'c')
        self.d_g_undirected_cyclic = graph_vertex.UnWeightedGraphVertex(self.g_undirected_cyclic, 'd')

        self.g_undirected_cyclic.add_vertex(self.a_g_undirected_cyclic)
        self.g_undirected_cyclic.add_vertex(self.b_g_undirected_cyclic)
        self.g_undirected_cyclic.add_vertex(self.c_g_undirected_cyclic)
        self.g_undirected_cyclic.add_vertex(self.d_g_undirected_cyclic)

        self.g_undirected_cyclic.add_edge(self.a_g_undirected_cyclic, self.b_g_undirected_cyclic)
        self.g_undirected_cyclic.add_edge(self.b_g_undirected_cyclic, self.c_g_undirected_cyclic)
        self.g_undirected_cyclic.add_edge(self.a_g_undirected_cyclic, self.c_g_undirected_cyclic)
        self.classification_undirected_cyclic = self.g_undirected_cyclic.classify_edges()

        # Create a directed cyclic graph (Cormen page 542)
        self.g_directed_cyclic = graph.DirectedGraph(6)
        self.u_g_directed_cyclic = graph_vertex.UnWeightedGraphVertex(self.g_directed_cyclic, 'u')
        self.v_g_directed_cyclic = graph_vertex.UnWeightedGraphVertex(self.g_directed_cyclic, 'v')
        self.w_g_directed_cyclic = graph_vertex.UnWeightedGraphVertex(self.g_directed_cyclic, 'w')
        self.x_g_directed_cyclic = graph_vertex.UnWeightedGraphVertex(self.g_directed_cyclic, 'x')
        self.y_g_directed_cyclic = graph_vertex.UnWeightedGraphVertex(self.g_directed_cyclic, 'y')
        self.z_g_directed_cyclic = graph_vertex.UnWeightedGraphVertex(self.g_directed_cyclic, 'z')

        self.g_directed_cyclic.add_vertex(self.u_g_directed_cyclic)
        self.g_directed_cyclic.add_vertex(self.v_g_directed_cyclic)
        self.g_directed_cyclic.add_vertex(self.w_g_directed_cyclic)
        self.g_directed_cyclic.add_vertex(self.x_g_directed_cyclic)
        self.g_directed_cyclic.add_vertex(self.y_g_directed_cyclic)
        self.g_directed_cyclic.add_vertex(self.z_g_directed_cyclic)

        self.g_directed_cyclic.add_edge(self.u_g_directed_cyclic, self.v_g_directed_cyclic)
        self.g_directed_cyclic.add_edge(self.u_g_directed_cyclic, self.x_g_directed_cyclic)
        self.g_directed_cyclic.add_edge(self.v_g_directed_cyclic, self.y_g_directed_cyclic)
        self.g_directed_cyclic.add_edge(self.w_g_directed_cyclic, self.y_g_directed_cyclic)
        self.g_directed_cyclic.add_edge(self.w_g_directed_cyclic, self.z_g_directed_cyclic)
        self.g_directed_cyclic.add_edge(self.x_g_directed_cyclic, self.v_g_directed_cyclic)
        self.g_directed_cyclic.add_edge(self.y_g_directed_cyclic, self.x_g_directed_cyclic)
        self.g_directed_cyclic.add_edge(self.z_g_directed_cyclic, self.z_g_directed_cyclic)
        self.classification_directed_cyclic = self.g_directed_cyclic.classify_edges()

    def test_dfs_edge_classification_undirected_cyclic_clear(self):
        """
        Test method "clear" using an undirected graph.
        """
        res = self.classification_undirected_cyclic.clear()
        ref = dfs_edge_classification.DFSEdgeClassification(self.g_undirected_cyclic).clear()
        self.assertEqual(ref, res)

    def test_dfs_edge_classification_undirected_cyclic_get_parent(self):
        """
        Test method "get_parent" using an undirected graph.
        """
        res = self.classification_undirected_cyclic.get_parent()
        ref = dfs_edge_classification.DFSEdgeClassification(self.g_undirected_cyclic).get_parent()
        ref[self.a_g_undirected_cyclic] = None
        ref[self.b_g_undirected_cyclic] = self.a_g_undirected_cyclic
        ref[self.c_g_undirected_cyclic] = self.b_g_undirected_cyclic
        ref[self.d_g_undirected_cyclic] = None
        self.assertEqual(ref, res)

    def test_dfs_edge_classification_undirected_cyclic_get_discovery_time(self):
        """
        Test method "get_discovery_time" using an undirected graph.
        """
        res = self.classification_undirected_cyclic.get_discovery_time()
        ref = dfs_edge_classification.DFSEdgeClassification(self.g_undirected_cyclic).get_discovery_time()
        ref[self.a_g_undirected_cyclic] = 1
        ref[self.b_g_undirected_cyclic] = 2
        ref[self.c_g_undirected_cyclic] = 3
        ref[self.d_g_undirected_cyclic] = 7
        self.assertEqual(ref, res)

    def test_dfs_edge_classification_undirected_cyclic_get_finishing_time(self):
        """
        Test method "get_finishing_time" using an undirected graph.
        """
        res = self.classification_undirected_cyclic.get_finishing_time()
        ref = dfs_edge_classification.DFSEdgeClassification(self.g_undirected_cyclic).get_finishing_time()
        ref[self.a_g_undirected_cyclic] = 6
        ref[self.b_g_undirected_cyclic] = 5
        ref[self.c_g_undirected_cyclic] = 4
        ref[self.d_g_undirected_cyclic] = 8
        self.assertEqual(ref, res)

    def test_dfs_edge_classification_undirected_cyclic_get_order(self):
        """
        Test method "get_order" using an undirected graph.
        """
        res = self.classification_undirected_cyclic.get_order()
        ref = dfs_edge_classification.DFSEdgeClassification(self.g_undirected_cyclic).get_order()
        ref.append(self.c_g_undirected_cyclic)
        ref.append(self.b_g_undirected_cyclic)
        ref.append(self.a_g_undirected_cyclic)
        ref.append(self.d_g_undirected_cyclic)
        self.assertEqual(ref, res)

    def test_dfs_edge_classification_undirected_cyclic_get_edges(self):
        """
        Test method "get_edges" using an undirected graph.
        """
        res = self.classification_undirected_cyclic.get_edges()
        ref = dfs_edge_classification.DFSEdgeClassification(self.g_undirected_cyclic).get_edges()
        e31 = graph_edge.UnDirectedGraphEdge(self.g_undirected_cyclic, self.c_g_undirected_cyclic, self.a_g_undirected_cyclic)
        e23 = graph_edge.UnDirectedGraphEdge(self.g_undirected_cyclic, self.b_g_undirected_cyclic, self.c_g_undirected_cyclic)
        e12 = graph_edge.UnDirectedGraphEdge(self.g_undirected_cyclic, self.a_g_undirected_cyclic, self.b_g_undirected_cyclic)
        ref[e31] = graph_edge.EdgeClassification.BACK_EDGE
        ref[e23] = graph_edge.EdgeClassification.TREE_EDGE
        ref[e12] = graph_edge.EdgeClassification.TREE_EDGE
        self.assertEqual(ref, res)

    def test_dfs_edge_classification_undirected_cyclic_get_parent_of_vertex(self):
        """
        Test method "get_parent_of_vertex" using an undirected graph.
        """
        res = self.classification_undirected_cyclic.get_parent_of_vertex(self.b_g_undirected_cyclic)
        self.assertEqual(self.a_g_undirected_cyclic, res)

    def test_dfs_edge_classification_undirected_cyclic_get_discovery_time_of_vertex(self):
        """
        Test method "get_discovery_time_of_vertex" using an undirected graph.
        """
        res = self.classification_undirected_cyclic.get_discovery_time_of_vertex(self.d_g_undirected_cyclic)
        self.assertEqual(7, res)

    def test_dfs_edge_classification_undirected_cyclic_get_finishing_time_of_vertex(self):
        """
        Test method "get_finishing_time_of_vertex" using an undirected graph.
        """
        res = self.classification_undirected_cyclic.get_finishing_time_of_vertex(self.d_g_undirected_cyclic)
        self.assertEqual(8, res)

    def test_dfs_edge_classification_undirected_cyclic_get_tree_edges(self):
        """
        Test method "get_tree_edges" using an undirected graph.
        """
        res = self.classification_undirected_cyclic.get_tree_edges()
        ref = dfs_edge_classification.DFSEdgeClassification(self.g_undirected_cyclic).get_tree_edges()
        e23 = graph_edge.UnDirectedGraphEdge(self.g_undirected_cyclic, self.b_g_undirected_cyclic, self.c_g_undirected_cyclic)
        e12 = graph_edge.UnDirectedGraphEdge(self.g_undirected_cyclic, self.a_g_undirected_cyclic, self.b_g_undirected_cyclic)
        ref.add(e23)
        ref.add(e12)
        self.assertEqual(ref, res)

    def test_dfs_edge_classification_undirected_cyclic_get_back_edges(self):
        """
        Test method "get_back_edges" using an undirected graph.
        """
        res = self.classification_undirected_cyclic.get_back_edges()
        ref = dfs_edge_classification.DFSEdgeClassification(self.g_undirected_cyclic).get_back_edges()
        e31 = graph_edge.UnDirectedGraphEdge(self.g_undirected_cyclic, self.c_g_undirected_cyclic, self.a_g_undirected_cyclic)
        ref.add(e31)
        self.assertEqual(ref, res)

    def test_dfs_edge_classification_undirected_cyclic_get_forward_edges(self):
        """
        Test method "get_forward_edges" using an undirected graph.
        """
        res = self.classification_undirected_cyclic.get_forward_edges()
        ref = dfs_edge_classification.DFSEdgeClassification(self.g_undirected_cyclic).get_forward_edges()
        self.assertEqual(ref, res)

    def test_dfs_edge_classification_undirected_cyclic_get_cross_edges(self):
        """
        Test method "get_cross_edges" using an undirected graph.
        """
        res = self.classification_undirected_cyclic.get_cross_edges()
        ref = dfs_edge_classification.DFSEdgeClassification(self.g_undirected_cyclic).get_cross_edges()
        self.assertEqual(ref, res)

    def test_dfs_edge_classification_undirected_cyclic_get_number_of_tree_edges(self):
        """
        Test method "get_number_of_tree_edges" using an undirected graph.
        """
        res = self.classification_undirected_cyclic.get_number_of_tree_edges()
        self.assertEqual(2, res)

    def test_dfs_edge_classification_undirected_cyclic_get_number_of_back_edges(self):
        """
        Test method "get_number_of_back_edges" using an undirected graph.
        """
        res = self.classification_undirected_cyclic.get_number_of_back_edges()
        self.assertEqual(1, res)

    def test_dfs_edge_classification_undirected_cyclic_get_number_of_forward_edges(self):
        """
        Test method "get_number_of_forward_edges" using an undirected graph.
        """
        res = self.classification_undirected_cyclic.get_number_of_forward_edges()
        self.assertEqual(0, res)

    def test_dfs_edge_classification_undirected_cyclic_get_number_of_cross_edges(self):
        """
        Test method "get_number_of_cross_edges" using an undirected graph.
        """
        res = self.classification_undirected_cyclic.get_number_of_cross_edges()
        self.assertEqual(0, res)

    def test_dfs_edge_classification_undirected_cyclic_has_tree_edges(self):
        """
        Test method "has_tree_edges" using an undirected graph.
        """
        res = self.classification_undirected_cyclic.has_tree_edges()
        self.assertTrue(res)

    def test_dfs_edge_classification_undirected_cyclic_has_back_edges(self):
        """
        Test method "has_back_edges" using an undirected graph.
        """
        res = self.classification_undirected_cyclic.has_back_edges()
        self.assertTrue(res)

    def test_dfs_edge_classification_undirected_cyclic_has_forward_edges(self):
        """
        Test method "has_forward_edges" using an undirected graph.
        """
        res = self.classification_undirected_cyclic.has_forward_edges()
        self.assertFalse(res)

    def test_dfs_edge_classification_undirected_cyclic_has_cross_edges(self):
        """
        Test method "has_cross_edges" using an undirected graph.
        """
        res = self.classification_undirected_cyclic.has_cross_edges()
        self.assertFalse(res)

    def test_dfs_edge_classification_directed_cyclic_clear(self):
        """
        Test method "clear" using a directed graph.
        """
        res = self.classification_directed_cyclic.clear()
        ref = dfs_edge_classification.DFSEdgeClassification(self.g_directed_cyclic).clear()
        self.assertEqual(ref, res)

    def test_dfs_edge_classification_directed_cyclic_get_parent(self):
        """
        Test method "get_parent" using a directed graph.
        """
        res = self.classification_directed_cyclic.get_parent()
        ref = dfs_edge_classification.DFSEdgeClassification(self.g_directed_cyclic).get_parent()

        # Add vertices to map of parent for vertices
        ref[self.u_g_directed_cyclic] = None
        ref[self.v_g_directed_cyclic] = self.u_g_directed_cyclic
        ref[self.w_g_directed_cyclic] = None
        ref[self.x_g_directed_cyclic] = self.y_g_directed_cyclic
        ref[self.y_g_directed_cyclic] = self.v_g_directed_cyclic
        ref[self.z_g_directed_cyclic] = self.w_g_directed_cyclic
        self.assertEqual(ref, res)

    def test_dfs_edge_classification_directed_cyclic_get_edges(self):
        """
        Test method "get_edges" using a directed graph.
        """
        res = self.classification_directed_cyclic.get_edges()
        # Create reference from (Cormen page 542)
        ref = dfs_edge_classification.DFSEdgeClassification(self.g_directed_cyclic).get_edges()
        e_uv = self.g_directed_cyclic.get_edge(self.u_g_directed_cyclic, self.v_g_directed_cyclic)
        e_ux = self.g_directed_cyclic.get_edge(self.u_g_directed_cyclic, self.x_g_directed_cyclic)
        e_vy = self.g_directed_cyclic.get_edge(self.v_g_directed_cyclic, self.y_g_directed_cyclic)
        e_wy = self.g_directed_cyclic.get_edge(self.w_g_directed_cyclic, self.y_g_directed_cyclic)
        e_wz = self.g_directed_cyclic.get_edge(self.w_g_directed_cyclic, self.z_g_directed_cyclic)
        e_xv = self.g_directed_cyclic.get_edge(self.x_g_directed_cyclic, self.v_g_directed_cyclic)
        e_yx = self.g_directed_cyclic.get_edge(self.y_g_directed_cyclic, self.x_g_directed_cyclic)
        e_zz = self.g_directed_cyclic.get_edge(self.z_g_directed_cyclic, self.z_g_directed_cyclic)

        # Add the edges to map of edges
        ref[e_uv] = graph_edge.EdgeClassification.TREE_EDGE
        ref[e_ux] = graph_edge.EdgeClassification.FORWARD_EDGE
        ref[e_vy] = graph_edge.EdgeClassification.TREE_EDGE
        ref[e_wy] = graph_edge.EdgeClassification.CROSS_EDGE
        ref[e_wz] = graph_edge.EdgeClassification.TREE_EDGE
        ref[e_xv] = graph_edge.EdgeClassification.BACK_EDGE
        ref[e_yx] = graph_edge.EdgeClassification.TREE_EDGE
        ref[e_zz] = graph_edge.EdgeClassification.BACK_EDGE
        self.assertEqual(ref, res)

    def test_dfs_edge_classification_directed_cyclic_get_discovery_time(self):
        """
        Test method "get_discovery_time" using a directed graph.
        """
        res = self.classification_directed_cyclic.get_discovery_time()
        ref = dfs_edge_classification.DFSEdgeClassification(self.g_directed_cyclic).get_discovery_time()

        # Add vertices to map of discovering times for vertices
        ref[self.u_g_directed_cyclic] = 1
        ref[self.v_g_directed_cyclic] = 2
        ref[self.y_g_directed_cyclic] = 3
        ref[self.x_g_directed_cyclic] = 4
        ref[self.w_g_directed_cyclic] = 9
        ref[self.z_g_directed_cyclic] = 10
        self.assertEqual(ref, res)

    def test_dfs_edge_classification_directed_cyclic_get_finishing_time(self):
        """
        Test method "get_finishing_time" using a directed graph.
        """
        res = self.classification_directed_cyclic.get_finishing_time()
        ref = dfs_edge_classification.DFSEdgeClassification(self.g_directed_cyclic).get_finishing_time()

        # Add vertices to map of finishing times for vertices
        ref[self.x_g_directed_cyclic] = 5
        ref[self.y_g_directed_cyclic] = 6
        ref[self.v_g_directed_cyclic] = 7
        ref[self.u_g_directed_cyclic] = 8
        ref[self.z_g_directed_cyclic] = 11
        ref[self.w_g_directed_cyclic] = 12
        self.assertEqual(ref, res)

    def test_dfs_edge_classification_directed_cyclic_get_order(self):
        """
        Test method "get_order" using a directed graph.
        """
        res = self.classification_directed_cyclic.get_order()
        ref = dfs_edge_classification.DFSEdgeClassification(self.g_directed_cyclic).get_order()
        ref.append(self.x_g_directed_cyclic)
        ref.append(self.y_g_directed_cyclic)
        ref.append(self.v_g_directed_cyclic)
        ref.append(self.u_g_directed_cyclic)
        ref.append(self.z_g_directed_cyclic)
        ref.append(self.w_g_directed_cyclic)
        self.assertEqual(ref, res)

    def test_dfs_edge_classification_directed_cyclic_get_parent_of_vertex(self):
        """
        Test method "get_parent_of_vertex" using a directed graph.
        """
        res = self.classification_directed_cyclic.get_parent_of_vertex(self.v_g_directed_cyclic)
        self.assertEqual(self.u_g_directed_cyclic, res)

    def test_dfs_edge_classification_directed_cyclic_get_discovery_time_of_vertex(self):
        """
        Test method "get_discovery_time_of_vertex" using a directed graph.
        """
        res = self.classification_directed_cyclic.get_discovery_time_of_vertex(self.w_g_directed_cyclic)
        self.assertEqual(9, res)

    def test_dfs_edge_classification_directed_cyclic_get_tree_edges(self):
        """
        Test method "get_tree_edges" using a directed graph.
        """
        res = self.classification_directed_cyclic.get_tree_edges()
        ref = dfs_edge_classification.DFSEdgeClassification(self.g_directed_cyclic).get_tree_edges()
        edge_vy = self.g_directed_cyclic.get_edge(self.v_g_directed_cyclic, self.y_g_directed_cyclic)
        edge_uv = self.g_directed_cyclic.get_edge(self.u_g_directed_cyclic, self.v_g_directed_cyclic)
        edge_yx = self.g_directed_cyclic.get_edge(self.y_g_directed_cyclic, self.x_g_directed_cyclic)
        edge_wz = self.g_directed_cyclic.get_edge(self.w_g_directed_cyclic, self.z_g_directed_cyclic)
        ref.add(edge_vy)
        ref.add(edge_uv)
        ref.add(edge_yx)
        ref.add(edge_wz)
        self.assertEqual(ref, res)

    def test_dfs_edge_classification_directed_cyclic_get_back_edges(self):
        """
        Test method "get_back_edges" using a directed graph.
        """
        res = self.classification_directed_cyclic.get_back_edges()
        ref = dfs_edge_classification.DFSEdgeClassification(self.g_directed_cyclic).get_back_edges()
        edge_xv = self.g_directed_cyclic.get_edge(self.x_g_directed_cyclic, self.v_g_directed_cyclic)
        edge_zz = self.g_directed_cyclic.get_edge(self.z_g_directed_cyclic, self.z_g_directed_cyclic)
        ref.add(edge_xv)
        ref.add(edge_zz)
        self.assertEqual(ref, res)

    def test_dfs_edge_classification_directed_cyclic_get_forward_edges(self):
        """
        Test method "get_forward_edges" using a directed graph.
        """
        res = self.classification_directed_cyclic.get_forward_edges()
        ref = dfs_edge_classification.DFSEdgeClassification(self.g_directed_cyclic).get_forward_edges()
        edge_ux = self.g_directed_cyclic.get_edge(self.u_g_directed_cyclic, self.x_g_directed_cyclic)
        ref.add(edge_ux)
        self.assertEqual(ref, res)

    def test_dfs_edge_classification_directed_cyclic_get_cross_edges(self):
        """
        Test method "get_cross_edges" using a directed graph.
        """
        res = self.classification_directed_cyclic.get_cross_edges()
        ref = dfs_edge_classification.DFSEdgeClassification(self.g_directed_cyclic).get_cross_edges()
        edge_wy = self.g_directed_cyclic.get_edge(self.w_g_directed_cyclic, self.y_g_directed_cyclic)
        ref.add(edge_wy)
        self.assertEqual(ref, res)

    def test_dfs_edge_classification_directed_cyclic_get_number_of_tree_edges(self):
        """
        Test method "get_number_of_tree_edges" using a directed graph.
        """
        res = self.classification_directed_cyclic.get_number_of_tree_edges()
        self.assertEqual(4, res)

    def test_dfs_edge_classification_directed_cyclic_get_number_of_back_edges(self):
        """
        Test method "get_number_of_back_edges" using a directed graph.
        """
        res = self.classification_directed_cyclic.get_number_of_back_edges()
        self.assertEqual(2, res)

    def test_dfs_edge_classification_directed_cyclic_get_number_of_forward_edges(self):
        """
        Test method "get_number_of_forward_edges" using a directed graph.
        """
        res = self.classification_directed_cyclic.get_number_of_forward_edges()
        self.assertEqual(1, res)

    def test_dfs_edge_classification_directed_cyclic_get_number_of_cross_edges(self):
        """
        Test method "get_number_of_cross_edges" using a directed graph.
        """
        res = self.classification_directed_cyclic.get_number_of_cross_edges()
        self.assertEqual(1, res)

    def test_dfs_edge_classification_directed_cyclic_has_tree_edges(self):
        """
        Test method "has_tree_edges" using a directed graph.
        """
        res = self.classification_undirected_cyclic.has_tree_edges()
        self.assertTrue(res)

    def test_dfs_edge_classification_directed_cyclic_has_back_edges(self):
        """
        Test method "has_back_edges" using a directed graph.
        """
        res = self.classification_directed_cyclic.has_back_edges()
        self.assertTrue(res)

    def test_dfs_edge_classification_directed_cyclic_has_forward_edges(self):
        """
        Test method "has_forward_edges" using a directed graph.
        """
        res = self.classification_directed_cyclic.has_forward_edges()
        self.assertTrue(res)

    def test_dfs_edge_classification_directed_cyclic_has_cross_edges(self):
        """
        Test method "has_cross_edges using a directed graph.
        """
        res = self.classification_directed_cyclic.has_cross_edges()
        self.assertTrue(res)
