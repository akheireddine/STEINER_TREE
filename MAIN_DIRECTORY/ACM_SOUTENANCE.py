import random
import  numpy as np
import time
#import pydotplus
from threading import Event
import matplotlib.pyplot as plt

from threading import Thread, Event

import sys
import os
from os import path
sys.path.append(path.dirname('../PyAlgDat/py_alg_dat'))
from heapq import heappush, heappop

from py_alg_dat.graph_algorithms import GraphAlgorithms
from Graphe_Individu import Graphe_Individu

BASE = list()

class Whole_Graph:



    def __init__(self, stpFile):
        self.NumNodes = 0  #ATTRIBUT
        self.NumEdges = 0  #ATTRIBUT
        self.dictAdjacence = dict()  #ATTRIBUT
        self.dictValuations = dict() #ATTRIBUT
        self.dictValuationsOriginel = dict()  #ATTRIBUT
        self.NumTerminals = 0  #ATTRIBUT
        self.setTerminals = set()  #ATTRIBUT

        self.steinerNodes = set()  #ATTRIBUT
        self.ListeIndividus = list()  #ATTRIBUT
        self.wholeGraphDict = set()   #ATTRIBUT
        self.ListeDesDicosPopDeBase = list()


        self.start = time.time()
        self.last_time_bestIndividu = self.start

        f = open(stpFile, "r")
        line = f.readline()

        self.flux_reel = list()


        while line != 'EOF\n':
            line = f.readline()

            lineSplit = line.split(' ')
            if lineSplit[0] == 'SECTION':
                if lineSplit[1][:-1] == 'Graph':
                    #NumNodes : nombre de noeuds
                    self.NumNodes = int(f.readline().split(' ')[1])
                    #initialiser diction des noeuds : cle : node, Valeur : liste des noeuds adjacent
                    #initialiser dict des valuations: cle : couple de nodes , valeur : poids

                    for i in range(1, self.NumNodes+1):
                        self.dictAdjacence[i] = set()

                    #NumEdges : nombre d'aretes
                    self.NumEdges = int(f.readline().split(' ')[1])
                    #Stocker les edges
                    for i in range(self.NumEdges):
                        edgeLine = f.readline()
                        edgeTuple = tuple(edgeLine.split(' ')[1:])
                        node1, node2, weight = edgeTuple
                        node1 = int(node1)
                        node2 = int(node2)
                        weight = float(weight)
                        self.dictAdjacence[node1].add(node2)
                        self.dictAdjacence[node2].add(node1)
                        n1 = min(node1, node2)
                        n2 = max(node1, node2)

                        self.dictValuations[(n1, n2)] = weight
                        self.dictValuationsOriginel[(n1,n2)] = weight

                elif lineSplit[1][:-1] == 'Terminals':
                    #NumTerminals : nombre de noeuds terminaux
                    self.NumTerminals = int(f.readline().split(' ')[1])

                    for i in range(self.NumTerminals):
                        self.setTerminals.add(int(f.readline().split(' ')[1]))
                    self.steinerNodes = {i for i in range(1, self.NumNodes+1)} - self.setTerminals
                    self.wholeGraphDict = {n : 1 for n in self.steinerNodes}

                elif lineSplit[1][:-1] == 'Comment':
                    line = f.readline().split(' ')[-1][:-1]
                    name = line.split("\"")
                    self.name = name[1]
                    self.dirname = name[1][0]+"/"+name[1][1:]                    #ATTRIBUTE
                    try:
                        os.makedirs(name[1][0])
                        os.makedirs(self.dirname)
                    except:
                        pass

        self.best_individu = Graphe_Individu(self,self.wholeGraphDict)


    def get_name(self):
        return self.name

    def get_setTerminals(self):
        return self.setTerminals

    def get_dictValuations(self):
        return self.dictValuations

    def get_listIndividus(self):
        return self.ListeIndividus

    def reset_listIndividus(self):
        self.ListeIndividus = list()


    def generer_individu_aleatoire(self, proba):
        def zeroOuUn(value):
            if value <= proba:
                return 1
            return 0
        steinerNodesDict = {n : zeroOuUn(random.random()) for n in self.steinerNodes}
        self.ListeDesDicosPopDeBase.append(steinerNodesDict.copy())
        currentIndividu = Graphe_Individu(self,steinerNodesDict)
        self.ListeIndividus.append(currentIndividu)

    def generer_N_individus_aleatoire(self, N, probaMin, probaMax,event=Event()):
        for i in range(N):
            self.generer_individu_aleatoire(random.uniform(probaMin, probaMax))
            if event.is_set():
                break

    def calculer_proba_associee(self):
        total_cout = 0
        for individu in self.ListeIndividus:
            total_cout += individu.get_fitness()
        ListeProba = [1.*(total_cout -individu.get_fitness())/total_cout for individu in self.ListeIndividus]
        ListeProba = [1.*v/sum(ListeProba) for v in ListeProba]
        return ListeProba

    def restaurer_population_de_base(self):
        self.ListeIndividus = [Graphe_Individu(self, d) for d in self.ListeDesDicosPopDeBase]

    def selectionner_P_parents(self, nb_parents):
        parents = np.random.choice([i for i in range(len(self.ListeIndividus))], nb_parents, replace=True, p=self.calculer_proba_associee())
        return parents

#########################################Ce 11/05#############################################################################""
    def selectionner_2_parents_roulette(self):
        Somme_fitness_pop = sum([I.get_fitness() for I in self.ListeIndividus])
        eval_pop = [Somme_fitness_pop - I.get_fitness() for I in self.ListeIndividus]
        S1 = sum(eval_pop)
        r = random.randint(0, S1)

        somme = 0
        parent1 = -1#0
        while somme < r:
            somme += eval_pop[parent1]
            parent1 += 1

        r = random.randint(0,S1)
        somme = 0
        parent2 = -1#0
        while somme < r:
            somme += eval_pop[parent2]
            parent2 += 1
        return [parent1, parent2]



    def croisement_a_un_point(self, id_parent1, id_parent2, proba):
        parent1 = self.ListeIndividus[id_parent1]
        parent2 = self.ListeIndividus[id_parent2]

        id_point = 0
        for st_node in self.steinerNodes:
            if id_point != 0: #il n'y a pas de node 0
                break
            if random.random() < proba:
                id_point = st_node

        dict_fils1 = dict()
        for st_node, bit in parent1.get_dictSteinerNodes().items():
            if st_node >= id_point:
                dict_fils1[st_node] = parent2.get_dictSteinerNodes()[st_node]
            else:
                dict_fils1[st_node] = bit

        dict_fils2 = dict()
        for st_node, bit in parent2.get_dictSteinerNodes().items():
            if st_node >= id_point:
                dict_fils2[st_node] = parent1.get_dictSteinerNodes()[st_node]
            else:
                dict_fils2[st_node] = bit

        return Graphe_Individu(self, dict_fils1), Graphe_Individu(self, dict_fils2)         #Pour eviter les erreurs dues a la randomisation

    def selection_2_parents_dif(self):
        L_fitness = [I.get_fitness() for I in self.ListeIndividus]
        max_fit = max(L_fitness)
        L_dif_fitness = [max_fit - val for val in L_fitness]
        eval_pop = L_dif_fitness
        S1 = sum(eval_pop)
        r = random.randint(0, S1)

        somme = 0
        parent1 = -1#0
        while somme < r:
            somme += eval_pop[parent1]
            parent1 += 1

        r = random.randint(0,S1)
        somme = 0
        parent2 = -1#0
        while somme < r:
            somme += eval_pop[parent2]
            parent2 += 1
        print (parent1, parent2)
        return [parent1, parent2]

#########################################Ce 11/05#############################################################################""

    def realiser_croisement(self, id_parent1, id_parent2, proba):

        parent1 = self.ListeIndividus[id_parent1]
        parent2 = self.ListeIndividus[id_parent2]

        id_debut = 0
        id_fin = self.NumNodes + 1

        for st_node in self.steinerNodes:
            if id_debut != 0:
                break
            if random.random() < proba:
                id_debut = st_node

        for st_node in self.steinerNodes:
            if id_fin != self.NumNodes + 1:
                break
            if st_node > id_debut:
                if random.random() < proba:
                    id_fin = st_node


        if id_debut == 0:
            id_debut = min(self.steinerNodes)
        if id_fin == self.NumNodes + 1:                  #AU PIRE ON RETOURNE DES INDIVIDUS IDENTIQUES AUX PARENTS
            id_fin = max(self.steinerNodes)

        dict_fils1 = dict()
        for st_node, bit in parent1.get_dictSteinerNodes().items():
            if st_node <= id_fin and st_node >= id_debut:
                dict_fils1[st_node] = parent2.get_dictSteinerNodes()[st_node]
            else:
                dict_fils1[st_node] = bit

        dict_fils2 = dict()
        for st_node, bit in parent2.get_dictSteinerNodes().items():
            if st_node <= id_fin and st_node >= id_debut:
                dict_fils2[st_node] = parent1.get_dictSteinerNodes()[st_node]
            else:
                dict_fils2[st_node] = bit

        return Graphe_Individu(self,dict_fils1), Graphe_Individu(self,dict_fils2)



    def realiser_mutation(self, id_parent, proba):
        parent = self.ListeIndividus[id_parent]
        def oppose(bit):
            if bit == 0:
                return 1
            return 0
        dict_fils = dict()
        for st_node, bit in parent.get_dictSteinerNodes().items():
            if random.random() < proba :
                dict_fils[st_node] = oppose(bit)
            else:
                dict_fils[st_node] = bit

        return Graphe_Individu(self, dict_fils)



################################################################CE 23/04 ############################################################################################


    def calculer_meilleur_fitness(self):
        minimumFitness = self.ListeIndividus[0].get_fitness()
        maximumFitness = self.ListeIndividus[0].get_fitness()
        i = 0
        i_meilleur = 0
        for individu in self.ListeIndividus:
            if individu.get_fitness() < minimumFitness:
                minimumFitness = individu.get_fitness()
                i_meilleur = i
            elif individu.get_fitness() > maximumFitness:
                maximumFitness = individu.get_fitness()
            i += 1

        # print "minFitness", minimumFitness
        # print "---------------------------------------------------------------"
        return i_meilleur

    def stocker_best_individu(self, i):
        self.flux_reel.append(self.ListeIndividus[i].get_fitness())
        if len(self.flux_reel) == 500:
            print self.flux_reel
            exit(-2)
        if self.best_individu.get_fitness() > self.ListeIndividus[i].get_fitness():
            self.best_individu = Graphe_Individu(self, self.ListeIndividus[i].get_dictSteinerNodes().copy())
            self.last_time_bestIndividu = time.time()
            # print "============== MAJ BEST INDIVIDU (%.2f"%(self.last_time_bestIndividu-self.start)+" sec) : %d"%self.best_individu.get_fitness()+" ============== \n "
            print "Boom  Trouve Ameliore " , self.best_individu.get_fitness()

    def get_time_max_fitness(self):
        return round(self.last_time_bestIndividu - self.start,2)



    def remplacement_generationnel(self, best_fitness, probaMutation, probaCroisement, event):
        self.best_individu = Graphe_Individu(self, self.wholeGraphDict)
        self.start = time.time()
        self.last_time_bestIndividu = self.start
        taillePopulation = len(self.ListeIndividus)
        self.stocker_best_individu(self.calculer_meilleur_fitness())
        if best_fitness == self.best_individu.get_fitness():
            return

        def nouvelle_generation():
            G_prim = list()
            for i in range(taillePopulation/2):
                L_parents = self.selectionner_P_parents(2)
                id_parent1 = L_parents[0]
                id_parent2 = L_parents[1]

                # Fils1, Fils2 = self.realiser_croisement(id_parent1, id_parent2, probaCroisement)
                Fils1, Fils2 = self.croisement_a_un_point(id_parent1, id_parent2, probaCroisement)

                G_prim.append(Fils1)
                G_prim.append(Fils2)

                #Mutation des parents
                self.ListeIndividus[id_parent1] = self.realiser_mutation(id_parent1, probaMutation)
                self.ListeIndividus[id_parent2] = self.realiser_mutation(id_parent2, probaMutation)
            return G_prim

#        for nbG in range(nombreDeGenerations):
        while True:
            self.ListeIndividus = nouvelle_generation()
            # print "-----new generation ---------"
            # for I in self.ListeIndividus:
            #     print I
            self.stocker_best_individu(self.calculer_meilleur_fitness())
            if best_fitness == self.best_individu.get_fitness():
                return

            if event.is_set():
                break
#            self.calculer_meilleur_fitness()






    def remplacement_elitiste(self, best_fitness, probaMutation, probaCroisement, event):

        self.best_individu = Graphe_Individu(self, self.wholeGraphDict)
        self.start = time.time()
        self.last_time_bestIndividu = self.start
        taillePopulation = len(self.ListeIndividus)
        self.stocker_best_individu(self.calculer_meilleur_fitness())

        if best_fitness == self.best_individu.get_fitness():
            return

        def nouvelle_generation():
            G_prim = list()
            for i in range(taillePopulation/2):
                L_parents = self.selectionner_P_parents(2)
                id_parent1 = L_parents[0]
                id_parent2 = L_parents[1]

                Fils1, Fils2 = self.croisement_a_un_point(id_parent1, id_parent2, probaCroisement)
                G_prim.append(Fils1)
                G_prim.append(Fils2)

                #Mutation des parents
                self.ListeIndividus[id_parent1] = self.realiser_mutation(id_parent1, probaMutation)
                self.ListeIndividus[id_parent2] = self.realiser_mutation(id_parent2, probaMutation)
            return G_prim

#        for nbG in range(nombreDeGenerations):
        while True:
            self.ListeIndividus = self.ListeIndividus + nouvelle_generation()
            self.ListeIndividus.sort(key=lambda Individu: Individu.get_fitness())
            self.ListeIndividus = self.ListeIndividus[:taillePopulation]
            # print "-----new generation ---------"
            # for I in self.ListeIndividus:
            #     print I
            self.stocker_best_individu(self.calculer_meilleur_fitness())
            if best_fitness == self.best_individu.get_fitness():
                return


            if event.is_set():
                break
#            self.calculer_meilleur_fitness()


#        return self.ListeIndividus[self.calculer_meilleur_fitness()]





##########################################################################____RANDOMISATION_DES_HEURISTIQUES____############################################################################################"




    def getIdVerticesOfEdge(self,edge):
        s = int(edge.get_head_vertex().get_vertex_name())
        t = int(edge.get_tail_vertex().get_vertex_name())
        return s,t

    def dijkstra(self, source):
        def valeur_arete(n1, n2):
            node1 = min(n1, n2)
            node2 = max(n1, n2)
            return self.dictValuations[(node1, node2)]

        TabPredecesseurs = [0 for i in range(self.NumNodes + 1)]
        TabDistanceALaSource = [float("+inf") for i in range(self.NumNodes + 1)]
        TAS = []
        Fermes = set()
        # INITIALISATION
        TabDistanceALaSource[source] = 0
        TabPredecesseurs[source] = -1

        heappush(TAS, (0, source))
        while len(TAS) != 0 and self.setTerminals not in Fermes:
            d, noeud_courant = heappop(TAS)
            if noeud_courant not in Fermes:
                for noeud_adjacent_noeud_courant in self.dictAdjacence[noeud_courant] - Fermes:
                    poids_arete = valeur_arete(noeud_adjacent_noeud_courant, noeud_courant)
                    nouvelle_distance = d + poids_arete

                    if TabDistanceALaSource[noeud_adjacent_noeud_courant] > nouvelle_distance:
                        TabDistanceALaSource[noeud_adjacent_noeud_courant] = nouvelle_distance
                        TabPredecesseurs[noeud_adjacent_noeud_courant] = noeud_courant
                        heappush(TAS, (nouvelle_distance, noeud_adjacent_noeud_courant))

                Fermes.add(noeud_courant)

        return TabPredecesseurs, TabDistanceALaSource


    def chemin_entre_noeud_terminal1_et_noeud_terminal2(self, noeud_terminal1, noeud_terminal2, TabPredecesseurs):
        if TabPredecesseurs[noeud_terminal1] == -1:
            Chemin = [noeud_terminal2]
        # elif TabPredecesseurs[noeud_terminal2] == -1:
        #     Chemin = [noeud_terminal1]
        while TabPredecesseurs[Chemin[-1]] != -1:
            Chemin.append(TabPredecesseurs[Chemin[-1]])
        # Chemin.reverse()
        return Chemin

    def construireGrapheDistance(self):
        ListeTerminals = list(self.setTerminals)
        paths = dict()
        dictValuationDistance = dict()
        dictAdjacenceDistance = {n : (self.setTerminals - {n}) for n in self.setTerminals}
        for v,i in zip(ListeTerminals[:-1],range(len(ListeTerminals)-1)):
            TabPredecesseurs, TabDistanceALaSource = self.dijkstra(v)
            for u in ListeTerminals[i+1:]:
                l = [v,u]
                l.sort()
                l = tuple(l)
                paths[l] = self.chemin_entre_noeud_terminal1_et_noeud_terminal2(v,u,TabPredecesseurs)
                dictValuationDistance[l] = TabDistanceALaSource[u]

        self.dictValuations = dictValuationDistance  #modifier le dictionnaire des valuations des aretes
        dictAdjacenceOriginel = self.dictAdjacence.copy()
        self.dictAdjacence = dictAdjacenceDistance
        G = Graphe_Individu(self,dict())
        self.reinitialiser_dictValuations()
        self.dictAdjacence = dictAdjacenceOriginel

        return G,paths



    def reconstruireGrapheChemins(self, edgesACPM, paths):
        """
            Etape 3
            Renvoie Objet Individu
        """

        # Recuperer l'ensemble  de noeuds de l'ACPM
        set_Nodes = set()
        for stpath in edgesACPM:
            node1, node2 = self.getIdVerticesOfEdge(stpath)
            l = [node1, node2]
            l.sort()
            path = paths[tuple(l)]
            set_Nodes |= set(path)

        dictSteinerNodesPath = {n: 1 for n in set_Nodes}

        G = Graphe_Individu(self, dictSteinerNodesPath)
        return G



    def eliminationFeuilles(self,edges,vertices):
        """
            Etape 5
            Renvoie un dictionnaire contenant les noeuds de steiner
        """
        dictAdjacenceACPM = {n : set() for n in vertices}
        for edge in edges:
            s,t = self.getIdVerticesOfEdge(edge)
            if not(s in self.setTerminals) :
                dictAdjacenceACPM[s].add(t)
            if not(t in self.setTerminals):
                dictAdjacenceACPM[t].add(s)
        vertices_degree_one = { n for n in vertices if len(dictAdjacenceACPM[n])==1}

        dictSteinerNodeDelete = {n : 1 for n in (set(vertices) - vertices_degree_one) }
        return dictSteinerNodeDelete



    def heuristique_PCM(self,draw=False):
        """
            Renvoie un dictionnaire contenant les noeuds de steiner
        """
        if draw:
            try:
                os.makedirs(self.dirname+"/H_ShortestPath")
            except:
                pass

        #Graphe de depart contenant tout les noeuds
        # Individu = Graphe_Individu(self,self.wholeGraphDict)
        # G = Individu.get_graphe()
        #
        # if draw:
        #     self.drawGraph("/H_ShortestPath/G0",G.get_edges())

        G1,graphPaths = self.construireGrapheDistance()
        if draw:
            self.drawGraph("/H_ShortestPath/G1",G1.graphe.get_edges())

        """
            Etape 2
        """
        G2 = G1.get_MST()
        if draw:
            self.drawGraph("/H_ShortestPath/G2_%d"%G2.get_total_weight(),G2.get_edges())

        G3 = self.reconstruireGrapheChemins(G2.get_edges(),graphPaths)
        if draw:
            self.drawGraph("/H_ShortestPath/G3_%d"%G3.get_MST().get_total_weight(),G3.graphe.get_edges())

        """
            Etape 4
        """
        G4 = G3.get_MST()
        if draw:
            self.drawGraph("/H_ShortestPath/G4_%d"%G4.get_total_weight(),G4.get_edges())

        dictSteinerNodes = self.eliminationFeuilles(G4.get_edges(),G3.get_dictSteinerNodes().keys())

        G5 = Graphe_Individu(self,dictSteinerNodes)
        if draw:
            self.drawGraph("/H_ShortestPath/G5_%d"%G5.get_MST().get_total_weight(),G5.graphe.get_edges())

        for i in self.steinerNodes:
            if i not in dictSteinerNodes.keys():
                dictSteinerNodes[i] = 0

        for i in self.steinerNodes :
            if i not in dictSteinerNodes.keys():
                dictSteinerNodes[i] = 0

        return dictSteinerNodes



    def heuristique_ACPM(self,draw=False):
        """
            Renvoie un dictionnaire contenant les noeuds de steiner
        """

        def getVertricesOfPath(edges):
            set_node = set()
            for e in edges:
                id1,id2 = self.getIdVerticesOfEdge(e)
                set_node.add(id1)
                set_node.add(id2)
            return set_node


        if draw:
            try:
                os.makedirs(self.dirname+"/H_ACPM")
            except:
                pass


        #Graphe de depart contenant tout les noeuds
        G = Graphe_Individu(self,self.wholeGraphDict)
        if draw:
            self.drawGraph("/H_ACPM/G0",G.get_graphe().get_edges())


        nb_vertices = 1
        nb_tmp_vertices = 0
        i=0

        #Tant que le nombre de noeuds du graphe decroit, appliquer kruskal
        while nb_vertices > nb_tmp_vertices :
            g_acpm = G.get_MST()


            nb_vertices = len(G.get_dictSteinerNodes())
            set_Node = set(G.get_dictSteinerNodes().keys())
            dictSteinerNodez = self.eliminationFeuilles(g_acpm.get_edges(),set_Node)
            G = Graphe_Individu(self,dictSteinerNodez)
            nb_tmp_vertices = len(dictSteinerNodez)

            if draw:
                i+= 1
                self.drawGraph("/H_ACPM/G%d"%i+"_%d"%g_acpm.get_total_weight(),g_acpm.get_edges())


        if draw:
            self.drawGraph("/H_ACPM/GFinal_%d"%G.get_MST().get_total_weight(),G.get_MST().get_edges())

        for i in self.steinerNodes:
            if i not in dictSteinerNodez.keys():
                dictSteinerNodez[i] = 0

        # return Graphe_Individu(self,dictSteinerNodez)
        return dictSteinerNodez



    def drawGraph(self,filename,edges):
        dgraph= pydotplus.Dot(graph_type='graph')
        nodes= dict()
        visited_edge = set()
        for edge in edges :
            node1Id,node2Id = self.getIdVerticesOfEdge(edge)
            if not(visited_edge.__contains__((node1Id,node2Id))) and not(visited_edge.__contains__((node2Id,node1Id))):
                visited_edge.add((node1Id,node2Id))
                color1='black'
                color2='black'
                if node1Id in self.setTerminals:     #identifier les noeuds terminaux par une couleur
                    color1 = 'red'
                if node2Id in self.setTerminals:      #identifier les noeuds terminaux par une couleur
                    color2 = 'red'
                #Creation et ajout des noeuds dans le graphe
                if not(nodes.has_key(node1Id)):
                    nodes[node1Id] = pydotplus.Node(name=node1Id,label=node1Id,color=color1)
                    dgraph.add_node(nodes[node1Id])
                if not(nodes.has_key(node2Id)):
                    nodes[node2Id] = pydotplus.Node(name=node2Id,label=node2Id,color=color2)
                    dgraph.add_node(nodes[node2Id])
                #Creation et ajout des aretes dans le graphe
                edge=pydotplus.Edge(src=nodes[node1Id],dst=nodes[node2Id],label=edge.get_weight())
                dgraph.add_edge(edge)

        #sauvegarder l'image
        dgraph.write_png(self.dirname+"/"+filename)



#######################################################____RANDOMISATION_____##############################################################################################

    def reinitialiser_dictValuations(self):
        self.dictValuations = self.dictValuationsOriginel.copy()


    def randomisation_des_donnees_initiales(self, probaMinRandomisation, probaMaxRandomisation):

        for key in self.dictValuations.keys():
            p = random.uniform(probaMinRandomisation, probaMaxRandomisation)
            if random.random()<= 0.5:          #hausse
                self.dictValuations[key] *= (1 + p)
            else:                              #baisse
                self.dictValuations[key] *= (1 - p)

            self.dictValuations[key] = int(self.dictValuations[key])


    def generer_N_individus_heuristique(self, N, heuristique, probaMinRandomisaton, probaMaxRandomisation):

        self.ListeIndividus = list()
        for i in range(N):
            self.randomisation_des_donnees_initiales(probaMinRandomisaton, probaMaxRandomisation)

#            ListeDesDictSteinerNodes.append(heuristique())
            dict_indiv = heuristique()
            self.reinitialiser_dictValuations()
            new_indiv = Graphe_Individu(self,dict_indiv)
            self.ListeIndividus.append(new_indiv)

            # if event.is_set():
            #     self.reinitialiser_dictValuations()
            #     print " I PASSED IN EVENT STOP HEURISTIC "
            #     break




######################################################_____RECHERCHE LOCALE_____#############################################################################################


    def premier_voisinage_ameliorant_individu(self, Individu):

        copy_dico_steiner_individu = Individu.get_dictSteinerNodes().copy()
        for st_node, bit in copy_dico_steiner_individu.items():
            if bit == 1:
                copy_dico_steiner_individu[st_node] = 0
                Voisin = Graphe_Individu(self,copy_dico_steiner_individu)

                if Voisin.get_fitness() < Individu.get_fitness():
                    return Voisin

                copy_dico_steiner_individu[st_node] = 1                    #RETABLISSEMENT

            elif len(Individu.get_ind_dict_adjacence()[st_node]) >= 2:    #Si le noeud a ajoute n'est pas une feuille
                copy_dico_steiner_individu[st_node] = 1
                Voisin = Graphe_Individu(self,copy_dico_steiner_individu)

                if Voisin.get_fitness() < Individu.get_fitness():
                    return Voisin
                copy_dico_steiner_individu[st_node] = 0                   #RETABLISSEMENT

        return None







    def algorithme_genetique_generationnel(self, N, heuristic, probaMutation, probaCroisement, L, CoupleInstanceHeuristic, event):

        self.generer_N_individus_heuristique(N,heuristic,probaMinRandomisaton,probaMaxRandomisation)
        meilleur_fitness = self.ListeIndividus[self.calculer_meilleur_fitness()].get_fitness()
        L.append(meilleur_fitness)
        instance, nom_heuristic, opt = CoupleInstanceHeuristic
        if meilleur_fitness == opt:
            event.set()
        def nouvelle_generation():
            G_prim = list()
            for i in range(N/2):
                L_parents = self.selectionner_2_parents_roulette()
                id_parent1 = L_parents[0]
                id_parent2 = L_parents[1]

                Fils1, Fils2 = self.croisement_a_un_point(id_parent1, id_parent2, probaCroisement)
                G_prim.append(Fils1)
                G_prim.append(Fils2)

                #Mutation des parents
                self.ListeIndividus[id_parent1] = self.realiser_mutation(id_parent1, probaMutation)
                self.ListeIndividus[id_parent2] = self.realiser_mutation(id_parent2, probaMutation)
            return G_prim

#        for nbG in range(nombreDeGenerations):
        while True:
            self.ListeIndividus = nouvelle_generation()
            meilleur_fitness = self.ListeIndividus[self.calculer_meilleur_fitness()].get_fitness()
            L.append(meilleur_fitness)
            if meilleur_fitness == opt:
                BASE.append((instance, opt, nom_heuristic, L))
                event.set()
            if event.is_set():
                BASE.append((instance, opt, nom_heuristic, L))
                break




    def algorithme_genetique_elitiste(self, N, heuristic, probaMutation, probaCroisement, L, CoupleInstanceHeuristic, event=Event()):
            debut = time.time()
            dic_heuristic = heuristic()
            I = Graphe_Individu(self, dic_heuristic)
            self.ListeIndividus.append(I)
            meilleur_fitness = self.ListeIndividus[self.calculer_meilleur_fitness()].get_fitness()
            self.generer_N_individus_heuristique(N-1,heuristic,probaMinRandomisaton,probaMaxRandomisation)

            L.append(meilleur_fitness)
            instance, nom_heuristic, opt = CoupleInstanceHeuristic
            def nouvelle_generation():
                G_prim = list()
                for i in range(N/2):
                    L_parents = self.selectionner_2_parents_roulette()
                    id_parent1 = L_parents[0]
                    id_parent2 = L_parents[1]

                    Fils1, Fils2 = self.croisement_a_un_point(id_parent1, id_parent2, probaCroisement)
                    G_prim.append(Fils1)
                    G_prim.append(Fils2)

                    #Mutation des parents
                    self.ListeIndividus[id_parent1] = self.realiser_mutation(id_parent1, probaMutation)
                    self.ListeIndividus[id_parent2] = self.realiser_mutation(id_parent2, probaMutation)
                return G_prim

    #        for nbG in range(nombreDeGenerations):
            while True:
                if time.time() - debut >= 60*timeout:
                    duree = (time.time() - debut)/60.
                    f.write(instance + " " + str(min(L)) + " " + str(duree) + "\n")
                    print instance, min(L), duree
                    BASE.append((instance, opt, nom_heuristic, L))
                    break
                self.ListeIndividus = self.ListeIndividus + nouvelle_generation()
                self.ListeIndividus.sort(key=lambda Individu: Individu.get_fitness())
                self.ListeIndividus = self.ListeIndividus[:N]
                meilleur_fitness = self.ListeIndividus[self.calculer_meilleur_fitness()].get_fitness()
                L.append(meilleur_fitness)
                if meilleur_fitness == opt:
                    duree = (time.time() - debut)/60.
                    f.write(instance + " " + str(min(L)) + " " + str(duree) + "\n")
                    print instance, min(L), duree
                    BASE.append((instance, opt, nom_heuristic, L))
                    break #event.set()
#                if event.is_set():
#                    duree = timeout
#                    f.write(instance + " " + str(min(L)) + " " + str(duree) + "\n")
#                    print instance, min(L), duree
#                    BASE.append((instance, opt, nom_heuristic, L))
#                    break



    def local_search(self, heuristic, probaMinRandomisaton, probaMaxRandomisation, L, CoupleInstanceHeuristic, event):
        debut = time.time()
        dic_heuristic = heuristic()
        I = Graphe_Individu(self, dic_heuristic)
        self.ListeIndividus.append(I)
        meilleur_fitness = self.ListeIndividus[self.calculer_meilleur_fitness()].get_fitness()
        instance, nom_heuristic, opt = CoupleInstanceHeuristic
        duree = (time.time() - debut)/60.
        if meilleur_fitness == opt:
            f.write(instance + " " + str(meilleur_fitness) + " " + str(duree) + "\n")
            print instance, meilleur_fitness, duree
            return
        while True:
            L.append(self.ListeIndividus[-1].get_fitness())
            # if time.time() - deb >= 60*5:
            Voisin = self.premier_voisinage_ameliorant_individu(self.ListeIndividus[-1])
            if Voisin != None:
                self.ListeIndividus.append(Voisin)
            else:
                self.generer_N_individus_heuristique(1, heuristic, probaMinRandomisaton, probaMaxRandomisation)
            meilleur_fitness = self.ListeIndividus[self.calculer_meilleur_fitness()].get_fitness()
            if meilleur_fitness == opt:
                duree = (time.time() - debut)/60.
                f.write(instance + " " + str(meilleur_fitness) + " " + str(duree) + "\n")
                print instance, meilleur_fitness, duree
                BASE.append((instance, opt, nom_heuristic, L))
                break

            if event.is_set():
                duree = timeout
                f.write(instance + " " + str(meilleur_fitness) + " " + str(duree) + "\n")
                print instance, meilleur_fitness, duree
                BASE.append((instance, opt, nom_heuristic, L))
                break
                # affiche(L,nom_heuristic,instance,opt)
                # print instance, L
                break

#    def recherche_locale(self):
#            self.ListeIndividusRLocale = list()
#
#            for individu in self.ListeIndividus:
#                Voisin = self.premier_voisinage_ameliorant_individu(individu)
#                if Voisin == None:
#                    self.ListeIndividusRLocale.append(individu.get_fitness())
#                else:
#                    while Voisin !=None:
#                        final_result = Voisin
#                        Voisin = self.premier_voisinage_ameliorant_individu(Voisin)
#
#                    self.ListeIndividusRLocale.append(final_result.get_fitness())




    def recherche_locale(self, best_fitness,event):
        self.best_individu = Graphe_Individu(self, self.wholeGraphDict)
        self.start = time.time()
        self.last_time_bestIndividu = self.start
        self.stocker_best_individu(self.calculer_meilleur_fitness())
        if best_fitness == self.best_individu.get_fitness():
            return

        i = 0
        for individu in self.ListeIndividus:
            Voisin = self.premier_voisinage_ameliorant_individu(individu)
            if Voisin == None:
                self.ListeIndividus.insert(i,individu.get_fitness())
            else:
                while Voisin !=None:
                    final_result = Voisin
                    Voisin = self.premier_voisinage_ameliorant_individu(Voisin)

                self.ListeIndividus.insert(i,final_result.get_fitness())
                self.stocker_best_individu(self.calculer_meilleur_fitness())
                if best_fitness == self.best_individu.get_fitness():
                    return

                if event.is_set():
                    break

            i += 1




    def visio_genetique(self, N, heuristique, probaMinRandomisaton, probaMaxRandomisation):
        self.generer_N_individus_heuristique(N, heuristique, probaMinRandomisaton, probaMaxRandomisation)
        # for I in self.ListeIndividus:
        #     print I
        print "debut remplacement"
        # self.remplacement_generationnel(0, probaMutation, probaCroisement, Event())
        self.remplacement_elitiste(0, probaMutation, probaCroisement, Event())

######################################################________FIN RECHERCHE LOCALE_________#################################################################################

def comparer_croisements(probaMinRandomisaton, probaMaxRandomisation, probaCroisement):
    g = Whole_Graph("../C/c" + str(13) + ".stp")
    L_1P = list()
    L_2P = list()
    for i in range(50):
        g.generer_N_individus_heuristique(2, g.heuristique_ACPM, probaMinRandomisaton, probaMaxRandomisation)
        F1_1P, F2_1P = g.realiser_croisement(0, 1, probaCroisement)
        L_1P.append(abs(F1_1P.get_fitness() - F2_1P.get_fitness()))
        # L_2P.append()
        F1_2P, F2_2P = g.croisement_a_un_point(0, 1, probaCroisement)
        L_2P.append(abs(F1_2P.get_fitness() - F2_2P.get_fitness()))
        # L_2P.append()

    plt.plot(range(50), L_1P, color="red")
    plt.plot(range(50), L_2P, color="blue")
    plt.show()

def affiche(L, nom_heuristic, nom_instance, opt, timeOut):
    colors = ["tomato", "navy", "hotpink", "khaki"]
    # plt.figure()
    g1 = plt.plot(range(len(L)),L, color=colors[random.randint(0,3)])
    g2 = plt.plot(range(len(L)), [opt for i in range(len(L))], color="gold")
    plt.legend([g1[0], g2[0]], ['Premier Voisin Ameliorant RL', "OPT"])
    plt.ylim(opt*9./10, max(L)*11./10)
    plt.suptitle("Instance " + nom_instance + " Heuristic " + nom_heuristic)
    plt.savefig("RL_"+nom_instance+"_"+nom_heuristic+"(~" +str(timeOut)+"min).png")
    # plt.show()
    plt.close()

def thread_timeout(target, args, timeLimit):
    stop_event = Event()
    args.append(stop_event)
    # RES = list()
    # args.append(RES)
    try:
        thread = Thread(target=target,args=tuple(args))
        thread.start()
        thread.join(timeout=timeLimit)
        stop_event.set()
        # print RES
    except Exception:
        pass

def affiche_AG(L, nom_heuristic, nom_instance, opt, timeOut):
    colors = ["deepskyblue", "cornflowerblue", "steelblue", "chocolate"]
    g1 = plt.plot(range(len(L)),L, color=colors[random.randint(0,3)])
    g2 = plt.plot(range(len(L)), [opt for i in range(len(L))], color="gold")
    plt.legend([g1[0], g2[0]], ['Meilleure fitness de generation', "OPT"])
    plt.ylim(opt*9./10, max(L)*11./10)
    plt.suptitle("Instance " + nom_instance + " Heuristic " + nom_heuristic)
    plt.savefig("AG_"+nom_instance+"_"+nom_heuristic+"(~" +str(timeOut)+"min).png")
    plt.close()


filename = "../B/b04.stp"
WG = Whole_Graph(filename)
t1 = time.time()
dictIndividu = WG.heuristique_ACPM()
t2 = time.time()
Ind = Graphe_Individu(WG, dictIndividu)
print "Fitness ACM ", Ind.get_fitness(), "  temps : ",t2-t1
