#!/usr/bin/env python

"""
Test of Prims's algorithm for a UnDirected Weighted Graph.
"""

def create_graph():
    """
    Creates an UnDirected Weighted Graph
    """
    # Create an empty undirected weighted graph
    graph = UnDirectedWeightedGraph(7)

    # Create vertices
    vertex1 = UnWeightedGraphVertex(graph, "A")
    vertex2 = UnWeightedGraphVertex(graph, "B")
    vertex3 = UnWeightedGraphVertex(graph, "C")
    vertex4 = UnWeightedGraphVertex(graph, "D")
    vertex5 = UnWeightedGraphVertex(graph, "E")
    vertex6 = UnWeightedGraphVertex(graph, "F")
    vertex7 = UnWeightedGraphVertex(graph, "G")

    # Add vertices
    graph.add_vertex(vertex1)
    graph.add_vertex(vertex2)
    graph.add_vertex(vertex3)
    graph.add_vertex(vertex4)
    graph.add_vertex(vertex5)
    graph.add_vertex(vertex6)
    graph.add_vertex(vertex7)

    # Add edges
    graph.add_edge(vertex1, vertex2, 7)    # (A - B, 7)
    graph.add_edge(vertex1, vertex4, 5)    # (A - D, 5)
    graph.add_edge(vertex2, vertex3, 8)    # (B - C, 8)
    graph.add_edge(vertex2, vertex4, 9)    # (B - D, 9)
    graph.add_edge(vertex2, vertex5, 7)    # (B - E, 7)
    graph.add_edge(vertex3, vertex5, 5)    # (C - E, 5)
    graph.add_edge(vertex4, vertex5, 15)   # (D - E, 1)
    graph.add_edge(vertex4, vertex6, 6)    # (D - F, 6)
    graph.add_edge(vertex5, vertex6, 8)    # (E - F, 8)
    graph.add_edge(vertex5, vertex7, 9)    # (E - G, 9)
    graph.add_edge(vertex6, vertex7, 11)   # (F - G, 11)
    return graph

if __name__ == "__main__":
    # Make it possible to use py_alg_dat without performing
    # an installation. This is needed in order to be able
    # to run: python prim_test.py, without having
    # performed an installation of the package. The is
    # neccessary due to Python's handling of relative
    # imports.
    if __package__ is None:
        import sys
        from os import path
        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
        from py_alg_dat.graph import UnDirectedWeightedGraph
        from py_alg_dat.graph_vertex import UnWeightedGraphVertex
        from py_alg_dat.graph_algorithms import GraphAlgorithms
    else:
        from ..py_alg_dat.graph import UnDirectedWeightedGraph
        from ..py_alg_dat.graph_vertex import UnWeightedGraphVertex
        from ..py_alg_dat.graph_algorithms import GraphAlgorithms

    # Create the graph
    GRAPH = create_graph()
    # Run Prim's algorithm
    MST = GraphAlgorithms.prims_algorithm(GRAPH, GRAPH[0])
    print MST

