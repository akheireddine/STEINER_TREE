#!/usr/bin/env python

"""
Test of shortest path algorithm for a Directed Weighted Graph.
"""

def create_graph():
    """
    Creates a Directed Weighted Graph
    """
    # Create an empty directed weighted graph
    graph = DirectedWeightedGraph(7)

    # Create vertices
    vertex0 = UnWeightedGraphVertex(graph, "A")
    vertex1 = UnWeightedGraphVertex(graph, "B")
    vertex2 = UnWeightedGraphVertex(graph, "C")
    vertex3 = UnWeightedGraphVertex(graph, "D")
    vertex4 = UnWeightedGraphVertex(graph, "E")
    vertex5 = UnWeightedGraphVertex(graph, "F")
    vertex6 = UnWeightedGraphVertex(graph, "G")

    # Add vertices
    graph.add_vertex(vertex0)
    graph.add_vertex(vertex1)
    graph.add_vertex(vertex2)
    graph.add_vertex(vertex3)
    graph.add_vertex(vertex4)
    graph.add_vertex(vertex5)
    graph.add_vertex(vertex6)

    # Add edges
    graph.add_edge(vertex0, vertex1, 7)   # ( A <- B, 7 )
    graph.add_edge(vertex1, vertex2, 2)   # ( B <- C, 2 )
    graph.add_edge(vertex1, vertex6, 3)   # ( B -> G, 3 )
    graph.add_edge(vertex2, vertex3, 2)   # ( C -> D, 2 )
    graph.add_edge(vertex2, vertex6, 4)   # ( C -> G, 4 )
    graph.add_edge(vertex3, vertex4, 5)   # ( D -> E, 5 )
    graph.add_edge(vertex3, vertex6, 1)   # ( D -> G, 1 )
    graph.add_edge(vertex4, vertex5, 6)   # ( E -> F, 6 )
    graph.add_edge(vertex5, vertex0, 1)   # ( F <- A, 1 )
    graph.add_edge(vertex5, vertex6, 4)   # ( F <- G, 4 )
    graph.add_edge(vertex6, vertex0, 7)   # ( G -> A, 7 )
    graph.add_edge(vertex6, vertex4, 1)   # ( G -> E, 1 )

    #       B--<--7--<--A
    #      / \         / \
    #     /   \       /   \
    #    2     3     7     1
    #   /       \   /       \
    #  /         \ /         \
    # C-->--4-->--G--<--4--<--F
    #  \         / \         /
    #   \       /   \       /
    #    2     1     1     6
    #     \   /       \   /
    #      \ /         \ /
    #       D-->--5-->--E

    return graph


if __name__ == "__main__":

    # Make it possible to use py_alg_dat without performing
    # an installation. This is needed in order to be able
    # to run: python shortest_path_test.py, without having
    # performed an installation of the package. The is
    # neccessary due to Python's handling of relative
    # imports.
    if __package__ is None:
        import sys
        from os import path
        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
        from py_alg_dat.graph import DirectedWeightedGraph
        from py_alg_dat.graph_vertex import UnWeightedGraphVertex
        from py_alg_dat.graph_algorithms import GraphAlgorithms
    else:
        from ..py_alg_dat.graph import DirectedWeightedGraph
        from ..py_alg_dat.graph_vertex import UnWeightedGraphVertex
        from ..py_alg_dat.graph_algorithms import GraphAlgorithms

    # Create the graph
    GRAPH = create_graph()
    # Find shortest path from vertex vertex "A" to vertex "G"
    PATH = GraphAlgorithms.shortest_path(GRAPH, GRAPH[0], GRAPH[5])
    # Find the edges in the Spanning Tree and its total weight
    print PATH

