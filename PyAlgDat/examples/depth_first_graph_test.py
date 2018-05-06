#!/usr/bin/env python

"""
Test of Depth First traversal in an UnDirected Weighted Graph.
"""

def create_graph():
    """
    Creates an UnDirected Weighted Graph
    """
    graph = UnDirectedWeightedGraph(7)

    vertex1 = UnWeightedGraphVertex(graph, 'A')
    vertex2 = UnWeightedGraphVertex(graph, 'B')
    vertex3 = UnWeightedGraphVertex(graph, 'C')
    vertex4 = UnWeightedGraphVertex(graph, 'D')
    vertex5 = UnWeightedGraphVertex(graph, 'E')
    vertex6 = UnWeightedGraphVertex(graph, 'F')
    vertex7 = UnWeightedGraphVertex(graph, 'G')

    graph.add_vertex(vertex1)
    graph.add_vertex(vertex2)
    graph.add_vertex(vertex3)
    graph.add_vertex(vertex4)
    graph.add_vertex(vertex5)
    graph.add_vertex(vertex6)
    graph.add_vertex(vertex7)

    graph.add_edge(vertex1, vertex2, 7)
    graph.add_edge(vertex1, vertex4, 5)
    graph.add_edge(vertex2, vertex3, 8)
    graph.add_edge(vertex2, vertex4, 9)
    graph.add_edge(vertex2, vertex5, 7)
    graph.add_edge(vertex3, vertex5, 5)
    graph.add_edge(vertex4, vertex5, 15)
    graph.add_edge(vertex4, vertex6, 6)
    graph.add_edge(vertex5, vertex6, 8)
    graph.add_edge(vertex5, vertex7, 9)
    graph.add_edge(vertex6, vertex7, 11)
    return graph

if __name__ == "__main__":
    # Make it possible to use py_alg_dat without performing
    # an installation. This is needed in order to be able
    # to run: python depth_first_graph_test.py, without
    # having performed an installation of the package.
    # This is neccessary due to Python's handling of
    # relative imports.
    if __package__ is None:
        import sys
        from os import path
        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
        from py_alg_dat.graph import UnDirectedWeightedGraph
        from py_alg_dat.graph_vertex import UnWeightedGraphVertex
        from py_alg_dat.vertex_visitor import VertexVisitor
    else:
        from ..py_alg_dat.graph import UnDirectedWeightedGraph
        from ..py_alg_dat.graph_vertex import UnWeightedGraphVertex
        from ..py_alg_dat.vertex_visitor import VertexVisitor

    # Create the graph
    GRAPH = create_graph()
    # Create vertex visitor
    VISITOR = VertexVisitor()
    # Run Depth First traversal
    GRAPH.depth_first_traversal(VISITOR, 0)
    # Show result
    print str("Depth First traversal: ") + str(VISITOR.get_visited())

