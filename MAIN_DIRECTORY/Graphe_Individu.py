from os import path
import sys

sys.path.append(path.dirname('../PyAlgDat/py_alg_dat'))

from py_alg_dat.graph import UnDirectedWeightedGraph
from py_alg_dat.graph_vertex import UnWeightedGraphVertex
from py_alg_dat.graph_algorithms import GraphAlgorithms



M = 9223372036854775807


class Graphe_Individu:


        def __init__(self, wholeGraph, dictSteinerNodes): 

            self.wholeGraph = wholeGraph                                              #ATTRIBUT
            self.nodes = set()                                                        #ATTRIBUT
            for n, bit in dictSteinerNodes.items():
                if bit == 1:
                    self.nodes.add(n)

            self.dictSteinerNodes = dictSteinerNodes                                  #ATTRIBUT

            self.nodes = self.nodes | wholeGraph.get_setTerminals()
#            self.ind_dictAdjacence = {node : list() for node in self.nodes}
            self.ind_dictAdjacence = {node : list() for node in range(1, wholeGraph.NumNodes + 1)}       #ATTRIBUT

            self.graphe = UnDirectedWeightedGraph(len(self.nodes))                    #ATTRIBUT
            self.m_a_j_GrapheETAdjacence()  #ET MAJ_dict ADJACENCE

            self.MST = self.MST_by_Kruskal()                                          #ATTRIBUT
            self.my_fitness = self.fitness()                                          #ATTRIBUT
            

        def get_dictSteinerNodes(self):
            return self.dictSteinerNodes

        def get_graphe(self):
            return self.graphe
            
        def get_ind_dict_adjacence(self):
            return self.ind_dictAdjacence
            
        def get_fitness(self):
            return self.fitness()
            
            
        def getVertex(self,vertex):
            return self.dictVertices[vertex]
            
        def get_MST(self):
            return self.MST
            

        def m_a_j_GrapheETAdjacence(self):
            # Ajouter les nodes
            self.dictVertices = {n : UnWeightedGraphVertex(self.graphe, n) for n in self.nodes}
            for idd, Vertex in self.dictVertices.items():
                self.graphe.add_vertex(Vertex)
            # Ajouter les edges
            for nodes, weight in self.wholeGraph.get_dictValuations().items():
                node1, node2 = nodes
                if node1 in self.nodes and node2 in self.nodes:
                    self.ind_dictAdjacence[node1].append(node2)
                    self.ind_dictAdjacence[node2].append(node1)
                    self.graphe.add_edge(self.dictVertices[node1], self.dictVertices[node2], weight)

            
        def MST_by_Prim(self, startVertex):                                                             ########---------------------
            return GraphAlgorithms.prims_algorithm(self.graphe, startVertex)


        def MST_by_Kruskal(self):
            return GraphAlgorithms.kruskals_algorithm(self.graphe)

        def fitness(self):
            total_weight = self.MST.get_total_weight() #Poids de la foret
            #Chercher a savoir si tous les noeuds terminaux sont couverts
            return total_weight + M*(len(self.nodes) - 1 - len(self.MST.get_edges()))
            
            
        def __str__(self):
            # return "Steiner Nodes : " + str(self.dictSteinerNodes) + " Fitness : " + str(self.my_fitness)
            return " Fitness : " + str(self.my_fitness)


        
        
        
        
        
        
        
        
        
        
        
