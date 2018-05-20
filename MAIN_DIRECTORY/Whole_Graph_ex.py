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







    def algorithme_genetique(self, N, heuristic, probaMutation, probaCroisement, L, CoupleInstanceHeuristic, event):

        self.generer_N_individus_heuristique(N,heuristic,probaMinRandomisaton,probaMaxRandomisation)
        L.append(self.ListeIndividus[self.calculer_meilleur_fitness()].get_fitness())
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
            self.ListeIndividus = self.ListeIndividus + nouvelle_generation()
            self.ListeIndividus.sort(key=lambda Individu: Individu.get_fitness())
            self.ListeIndividus = self.ListeIndividus[:N]
            L.append(self.ListeIndividus[self.calculer_meilleur_fitness()].get_fitness())

            if event.is_set():
                BASE.append((instance, opt, nom_heuristic, L))
                break








    def local_search(self, heuristic, probaMinRandomisaton, probaMaxRandomisation, L, CoupleInstanceHeuristic, event):
        dic_heuristic = heuristic()
        I = Graphe_Individu(self, dic_heuristic)
        self.ListeIndividus.append(I)

        instance, nom_heuristic, opt = CoupleInstanceHeuristic
        while True:
            L.append(self.ListeIndividus[-1].get_fitness())
            # if time.time() - deb >= 60*5:
            Voisin = self.premier_voisinage_ameliorant_individu(self.ListeIndividus[-1])
            if Voisin != None:
                self.ListeIndividus.append(Voisin)
            else:
                self.generer_N_individus_heuristique(1, heuristic, probaMinRandomisaton, probaMaxRandomisation)

            if event.is_set():
                BASE.append((instance, opt, nom_heuristic, L))
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


probaMutation = 0.05#random.uniform(0.01, 0.04)
probaCroisement = 0.2

probaMinRandomisaton = 0.05
probaMaxRandomisation = 0.20
timeout = 0.3 #minutes
N = 10

Liste_Instances = [5, 11, 13, 16]
Liste_OPT = [61, 88, 165, 127]


METH = "AG"

for i in range(len(Liste_Instances)):
    n_str = str(Liste_Instances[i])
    if Liste_Instances[i] < 10:
        n_str = str(0) + str(Liste_Instances[i])

    RES = list()
    g = Whole_Graph("../B/b" + n_str + ".stp")
    instance = "B"+str(Liste_Instances[i])
    heuristic = "PCC"
    couple_instance_heuristic_opt = instance, heuristic, Liste_OPT[i]
    # thread_timeout(g.local_search, [g.heuristique_ACPM, probaMinRandomisaton, probaMaxRandomisation, RES, couple_instance_heuristic_opt],timeout*60)
    # thread_timeout(g.local_search, [g.heuristique_PCM, probaMinRandomisaton, probaMaxRandomisation, RES, couple_instance_heuristic_opt],timeout*60)
    # thread_timeout(g.algorithme_genetique, [N, g.heuristique_ACPM, probaMutation, probaCroisement, RES, couple_instance_heuristic_opt], timeout*60)
    thread_timeout(g.algorithme_genetique, [N, g.heuristique_PCM, probaMutation, probaCroisement, RES, couple_instance_heuristic_opt], timeout*60)

# comparer_croisements(probaMinRandomisaton, probaMaxRandomisation, probaCroisement)

print BASE
if METH == "RL":
    for ELT in BASE:
        instance, opt, nom_heuristic, L = ELT
        affiche(L, nom_heuristic, instance, opt, timeout)
else:
    for ELT in BASE:
        instance, opt, nom_heuristic, L = ELT
        affiche_AG(L, nom_heuristic, instance, opt, timeout)


# probaMinRandomisaton = 0.05
# probaMaxRandomisation = 0.20
# probaMutation = random.uniform(0.01, 0.04)
# probaCroisement = 0.2
# best_fitness = 127
#
# for i in range(15,16):
#     n_str = str(i)
#     if i < 10:
#         n_str = str(0) + str(i)
#
#
#     g = Whole_Graph("../C/c" + n_str + ".stp")
#     # g.visio_genetique(30, g.heuristique_ACPM, probaMinRandomisaton, probaMaxRandomisation)
#     g.local_search(g.heuristique_ACPM, probaMinRandomisaton, probaMaxRandomisation)

#######################COMPARAISON METHODES DE SELECTION
 # R = list()
 #    R_ = list()
 #    R_D =list()
 #    for j in range(30):
    #     LP = g.selectionner_2_parents_roulette()
    #     R.append((g.ListeIndividus[LP[0]].get_fitness()+g.ListeIndividus[LP[1]].get_fitness())/2)
    #
    #     LP_ = g.selectionner_P_parents(2)
    #     R_.append((g.ListeIndividus[LP_[0]].get_fitness()+g.ListeIndividus[LP_[1]].get_fitness())/2)
    #
    #     LP_D = g.selection_2_parents_dif()
    #     R_D.append((g.ListeIndividus[LP_D[0]].get_fitness()+g.ListeIndividus[LP_D[1]].get_fitness())/2)
    #
    #
    #
    # print "R" ,sum(R)/30
    # print "R_", sum(R_)/30
    # print "R_D", sum(R_D)/30
    # plt.plot(range(30), R, color="red")
    # plt.plot(range(30), R_, color="yellow")
    # plt.plot(range(30), R_D, color="green")

    # plt.show()

#######################COMPARAISON METHODES DE SELECTION
    # print "parent1", g.ListeIndividus[LP[0]]
    # print "parent2", g.ListeIndividus[LP[1]]
    # print "--------------------------------"
    # print "parent1", g.ListeIndividus[LP_[0]]
    # print "parent2", g.ListeIndividus[LP_[1]]

    # g.remplacement_generationnel(best_fitness, probaMutation, probaCroisement, Event())

# L = list()
# Duree = list()
# nb_test = 21
# for i in range(1, nb_test):
#     n_str = str(i)
#     if i < 10:
#         n_str = str(0) + str(i)
#
#     g = Whole_Graph("../E/e" + n_str + ".stp")
#     deb = time.time()
#     dic = g.heuristique_PCM()
#     # dic = g.heuristique_ACPM()
#     Duree.append(time.time()-deb)
#     print Duree
#     fitness = Graphe_Individu(g, dic).get_fitness()
#     L.append(fitness)
#
# print "Val"
# print L

# OpT = [82,83,138,59,61,122,111,104,220,86,88,174,165,235,318,127,131,218]
# T = [9,13,25,9,13,25,9,13,25,9,13,25,9,13,25,9,13,25]
#
# LimPCC = [round((2. -2./T[i])*OpT[i], 2) for i in range(nb_test-1)]
# g1 = plt.scatter(range(1, nb_test), L, color='yellow', marker='o')
# g2 = plt.scatter(range(1,nb_test), OpT,color='green', marker='*')
# g3 = plt.scatter(range(1,nb_test), LimPCC , color='red', marker='1')
# plt.legend([g1, g2, g3], ["acm", "opt", "limPCM"])
# plt.suptitle("Evaluation empirique : Heuristique de l'arbre couvrant minimum(TestSet B)")
# plt.show()
# print "Duree"
# print Duree
# print "Lim PCC"
# print LimPCC

#for i in range(1, 2):
#    n_str = str(i)
#    if i < 10:
#        n_str = str(0) + str(i)
#    
#    g = Whole_Graph("../B/b" + n_str + ".stp")
#    g.generer_N_individus_aleatoire(taillePopulation, probaGeneMin, probaGeneMax)
##    indMeilleur = g.remplacement_elitiste(taillePopulation, nombreDeGenerations, probaGeneMin, probaGeneMax, probaMutation, pCroisement)
#    
##    print indMeilleur
#    g.recherche_locale(60*3) #3 minutes
#    g.calculer_meilleur_fitness()
    
    
    
#    ind = g.Graphe_Individu(g, g.wholeGraphDict)
#    arb = g.heuristique_ACPM(ind.graphe) 
#    print "Cout " + "(b" + n_str +")" , g.get_total_weight(arb)
#g = Whole_Graph("../B/b01.stp")
#g.heuristique_PCM(True)
#g.heuristique_ACPM(True)
#
#
#
#pmutation = 0.2 #between 1 and 4%
#pcroisement = 0.2
#NGene = 1000000
#pGeneMin = 0.2
#pGeneMax = 0.5
#taillePopulation = 500
#g.generer_N_individus_aleatoire(100, 0.2, 0.5)
#g.calculer_proba_associee()
#g.remplacement_generationnel(taillePopulation,NGene,pGeneMin,pGeneMax,pmutation,pcroisement)



# graph = UnDirectedWeightedGraph(4)
# vertex1 = UnWeightedGraphVertex(graph, "A")
# vertex2 = UnWeightedGraphVertex(graph, "B")
# vertex3 = UnWeightedGraphVertex(graph, "C")
# vertex4 = UnWeightedGraphVertex(graph, "D")
#
#
# graph.add_vertex(vertex1)
# graph.add_vertex(vertex2)
# graph.add_vertex(vertex3)
# graph.add_vertex(vertex4)
#
# graph.add_edge(vertex1, vertex2, 7)
# graph.add_edge(vertex3, vertex4, 5)
#
# MST = GraphAlgorithms.kruskals_algorithm(graph)
# print MST

##############################################################PCC###################################################################################
#B
#VAL = [82.0, 90.0, 138.0, 64.0, 61.0, 124.0, 111.0, 104.0, 222.0, 94.0, 90.0, 174.0, 175.0, 237.0, 318.0, 136.0, 133.0, 223.0]
# Opt = [82,83,138,59,61,122,111,104,220,86,88,174,165,235,318,127,131,218]
# T = [9,13,25,9,13,25,9,13,25,9,13,25,9,13,25,9,13,25]
# Duree = [0.0067441463470458984, 0.009167909622192383, 0.03881406784057617, 0.005869865417480469, 0.009519815444946289, 0.028523921966552734, 0.0101318359375, 0.01814413070678711, 0.06713700294494629, 0.01135396957397461, 0.01992201805114746, 0.06897497177124023, 0.017975807189941406, 0.03352999687194824, 0.11906194686889648, 0.0196230411529541, 0.036271095275878906, 0.12414884567260742]

#C
# Opt = [85,144,754,1079,1579,55,102,509,707,1093,32,46,258,323,556,11,18,113,146,267]
# T = [5,10,83,125,250,5,10,83,125,250,5,10,83,125,250,5,10,83,125,250]
#VAL = [88.0, 144.0, 772.0, 1107.0, 1585.0, 60.0, 109.0, 526.0, 717.0, 1105.0, 35.0, 49.0, 270.0, 334.0, 560.0, 12.0, 20.0, 122.0, 158.0, 268.0]
# Duree = [0.014037847518920898, 0.025249958038330078, 0.5575900077819824, 1.204524040222168, 5.612712860107422, 0.015146017074584961, 0.030779123306274414, 0.6883840560913086, 1.2457640171051025, 5.663992881774902, 0.022114992141723633, 0.04326295852661133, 0.6979119777679443, 1.4847121238708496, 6.275766849517822, 0.059844970703125, 0.18312788009643555, 1.340282917022705, 2.3900628089904785, 8.216739892959595]

#D
#D = [107.0, 237.0, 1605.0, 1972.0, 3266.0, 73.0, 105.0, 1122.0, 1510.0, 2132.0, 32.0, 43.0, 522.0, 683.0, 1135.0, 16.0, 25.0, 241.0, 330.0, 543.0]
# OpT = [106,220,1565,1935,3250,67,103,1072,1448,2110,29,42,500,667,1116,13,23,223,310,537]
# T = [5,10,167,250,500,5,10,167,250,500,5,10,167,250,500,5,10,167,250,500]
# DUREE = [0.03395509719848633, 0.060617923736572266, 2.8724329471588135, 6.517707109451294, 33.14359092712402, 0.038352012634277344, 0.06200885772705078, 3.3733420372009277, 6.889219045639038, 33.98662304878235, 0.051223039627075195, 0.09311485290527344, 3.7876269817352295, 7.88662314414978, 36.90067791938782, 0.12939810752868652, 0.25954103469848633, 6.851513147354126, 12.401702165603638, 45.72596001625061]


#E
# OpT = [111,214,4013,5101,8128,73,145,2640,3604,5600,34,67,1280,1732,2784,15,25,564,758,1342]
# T = [5,10,417,625,1250,5,10,417,625,1250,5,10,417,625,1250,5,10,417,625,1250]
# Duree = [0.13910889625549316, 0.1913158893585205, 28.635748863220215, 73.50468301773071, 422.3781008720398, 0.17519903182983398, 0.2401590347290039, 30.457620859146118, 85.25490498542786, 424.2287199497223, 0.197861909866333, 0.3028738498687744, 42.011751890182495, 89.78154182434082, 463.3807199001312, 0.4454929828643799, 0.7944211959838867, 61.441736936569214, 127.60712885856628, 554.1239531040192]
# VAL = [125.0, 255.0, 4181.0, 5207.0, 8168.0, 86.0, 169.0, 2751.0, 3715.0, 5643.0, 39.0, 71.0, 1345.0, 1784.0, 2818.0, 18.0, 29.0, 615.0, 791.0, 1355.0]

######################################################################ACPM##############################################################
#E
#DUREE = [7.140773057937622, 7.037506103515625, 11.982561111450195, 11.954309940338135, 15.766960859298706, 7.75667405128479, 8.061851978302002, 17.69463300704956, 15.25546407699585, 16.82558798789978, 8.671695947647095, 9.042387008666992, 14.376343011856079, 19.70241403579712, 23.805142879486084, 15.3387131690979, 15.227991104125977, 19.941237926483154, 22.0925190448761, 48.52823305130005]
#VAL = [189.0, 476.0, 4473.0, 5587.0, 8335.0, 157.0, 259.0, 3115.0, 4051.0, 5929.0, 52.0, 125.0, 1488.0, 1937.0, 2961.0, 48.0, 82.0, 646.0, 856.0, 1393.0]

#D
#DUREE= [1.2949848175048828, 1.516611099243164, 1.9345390796661377, 2.6601998805999756, 2.944054126739502, 1.4807639122009277, 1.4854459762573242, 2.507891893386841, 2.707580804824829, 2.6715660095214844, 1.9177579879760742, 1.7862749099731445, 3.221176862716675, 4.342092990875244, 3.945751905441284, 4.3657519817352295, 4.683380126953125, 6.297979116439819, 7.1460089683532715, 11.694835186004639]
#VAL = [143.0, 323.0, 1828.0, 2129.0, 3365.0, 119.0, 268.0, 1259.0, 1625.0, 2209.0, 70.0, 57.0, 583.0, 762.0, 1165.0, 33.0, 56.0, 278.0, 358.0, 555.0]
# OpT = [106,220,1565,1935,3250,67,103,1072,1448,2110,29,42,500,667,1116,13,23,223,310,537]
#LIM PCC = [169.60000000000002, 396.0, 3111.25748502994, 3854.52, 6487.0, 107.2, 185.4, 2131.1616766467064, 2884.416, 4211.56, 46.400000000000006, 75.60000000000001, 994.0119760479041, 1328.664, 2227.536, 20.8, 41.4, 443.3293413173653, 617.52, 1071.852]

#C
#DUREE = [0.390887975692749, 0.40096378326416016, 0.6392171382904053, 0.6557068824768066, 0.722858190536499, 0.49837207794189453, 0.4541139602661133, 0.6867339611053467, 0.9120938777923584, 0.8778960704803467, 0.6388888359069824, 0.6904468536376953, 0.8906960487365723, 1.1255741119384766, 1.1423509120941162, 1.7973871231079102, 1.9670391082763672, 2.4836370944976807, 2.5768940448760986, 4.023965120315552]
#VAL = [134.0, 197.0, 861.0, 1153.0, 1632.0, 98.0, 166.0, 636.0, 798.0, 1153.0, 58.0, 77.0, 314.0, 365.0, 595.0, 24.0, 32.0, 129.0, 163.0, 280.0]
# OPT = [85,144,754,1079,1579,55,102,509,707,1093,32,46,258,323,556,11,18,113,146,267]
#LIM PCC = [136.0, 259.2, 1489.83, 2140.74, 3145.37, 88.0, 183.6, 1005.73, 1402.69, 2177.26, 51.2, 82.8, 509.78, 640.83, 1107.55, 17.6, 32.4, 223.28, 289.66, 531.86]

#B
#DUREE = [0.012931108474731445, 0.017551898956298828, 0.025040149688720703, 0.020088911056518555, 0.02048206329345703, 0.020239830017089844, 0.025690793991088867, 0.023764848709106445, 0.03489494323730469, 0.03699898719787598, 0.038271188735961914, 0.04003310203552246, 0.046106815338134766, 0.04832005500793457, 0.06488204002380371, 0.048120975494384766, 0.05086493492126465, 0.04585981369018555]
#VAL = [83.0, 86.0, 144.0, 63.0, 68.0, 130.0, 112.0, 107.0, 220.0, 91.0, 103.0, 180.0, 189.0, 243.0, 333.0, 164.0, 133.0, 222.0]
#LIM PCC = [145.78, 153.23, 264.96, 104.89, 112.62, 234.24, 197.33, 192.0, 422.4, 152.89, 162.46, 334.08, 293.33, 433.85, 610.56, 225.78, 241.85, 418.56]


###############################################################################################################################################
#ACPM/INSTANCE B13/ELITISTE/OPT :165/TAIL_POP:30
#FLUX-REEL = [177.0, 177.0, 177.0, 177.0, 177.0, 173.0, 173.0, 173.0, 173.0, 173.0, 173.0, 173.0, 173.0, 173.0, 179.0, 175.0, 175.0, 175.0, 175.0, 175.0, 175.0, 175.0, 175.0, 175.0, 175.0, 175.0, 175.0, 175.0, 175.0, 175.0, 175.0, 175.0, 175.0, 175.0, 175.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0, 177.0]
#GEN FLUX-REEL = [174.0, 174.0, 174.0, 174.0, 181.0, 193.0, 193.0, 193.0, 193.0, 216.0, 216.0, 218.0, 218.0, 218.0, 218.0, 220.0, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 1.8446744073709552e+19, 1.8446744073709552e+19, 1.8446744073709552e+19, 1.8446744073709552e+19, 1.8446744073709552e+19, 1.8446744073709552e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 2.7670116110564327e+19, 3.6893488147419103e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 3.6893488147419103e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 6.456360425798343e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 6.456360425798343e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 5.5340232221128655e+19, 7.378697629483821e+19, 7.378697629483821e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 8.301034833169298e+19, 9.223372036854776e+19, 8.301034833169298e+19, 6.456360425798343e+19, 8.301034833169298e+19, 7.378697629483821e+19, 6.456360425798343e+19, 9.223372036854776e+19, 8.301034833169298e+19, 8.301034833169298e+19, 4.611686018427388e+19, 4.611686018427388e+19, 6.456360425798343e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 6.456360425798343e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 8.301034833169298e+19, 9.223372036854776e+19, 7.378697629483821e+19, 8.301034833169298e+19, 7.378697629483821e+19, 6.456360425798343e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 7.378697629483821e+19, 6.456360425798343e+19, 7.378697629483821e+19, 6.456360425798343e+19, 7.378697629483821e+19, 7.378697629483821e+19, 8.301034833169298e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 5.5340232221128655e+19, 4.611686018427388e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 7.378697629483821e+19, 7.378697629483821e+19, 6.456360425798343e+19, 7.378697629483821e+19, 7.378697629483821e+19, 9.223372036854776e+19, 9.223372036854776e+19, 9.223372036854776e+19, 9.223372036854776e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 6.456360425798343e+19, 7.378697629483821e+19, 6.456360425798343e+19, 7.378697629483821e+19, 6.456360425798343e+19, 7.378697629483821e+19, 7.378697629483821e+19, 6.456360425798343e+19, 6.456360425798343e+19, 7.378697629483821e+19, 6.456360425798343e+19, 7.378697629483821e+19, 8.301034833169298e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 8.301034833169298e+19, 7.378697629483821e+19, 7.378697629483821e+19, 5.5340232221128655e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 7.378697629483821e+19, 7.378697629483821e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 6.456360425798343e+19, 7.378697629483821e+19, 6.456360425798343e+19, 8.301034833169298e+19, 8.301034833169298e+19, 6.456360425798343e+19, 8.301034833169298e+19, 9.223372036854776e+19, 7.378697629483821e+19, 9.223372036854776e+19, 7.378697629483821e+19, 9.223372036854776e+19, 9.223372036854776e+19, 9.223372036854776e+19, 7.378697629483821e+19, 7.378697629483821e+19, 8.301034833169298e+19, 9.223372036854776e+19, 8.301034833169298e+19, 9.223372036854776e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 9.223372036854776e+19, 9.223372036854776e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 1.1068046444225731e+20, 1.1068046444225731e+20, 1.1990383647911209e+20, 1.0145709240540253e+20, 1.0145709240540253e+20, 7.378697629483821e+19, 9.223372036854776e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 7.378697629483821e+19, 6.456360425798343e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 7.378697629483821e+19, 9.223372036854776e+19, 8.301034833169298e+19, 8.301034833169298e+19, 6.456360425798343e+19, 9.223372036854776e+19, 7.378697629483821e+19, 8.301034833169298e+19, 8.301034833169298e+19, 6.456360425798343e+19, 6.456360425798343e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 1.8446744073709552e+19, 1.8446744073709552e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 6.456360425798343e+19, 6.456360425798343e+19, 8.301034833169298e+19, 7.378697629483821e+19, 7.378697629483821e+19, 8.301034833169298e+19, 9.223372036854776e+19, 1.0145709240540253e+20, 1.0145709240540253e+20, 1.1068046444225731e+20, 1.1990383647911209e+20, 1.0145709240540253e+20, 1.1068046444225731e+20, 1.0145709240540253e+20, 1.0145709240540253e+20, 8.301034833169298e+19, 7.378697629483821e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 9.223372036854776e+19, 9.223372036854776e+19, 6.456360425798343e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 6.456360425798343e+19, 7.378697629483821e+19, 8.301034833169298e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 9.223372036854776e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 7.378697629483821e+19, 9.223372036854776e+19, 9.223372036854776e+19, 8.301034833169298e+19, 8.301034833169298e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 8.301034833169298e+19, 9.223372036854776e+19, 9.223372036854776e+19, 9.223372036854776e+19, 8.301034833169298e+19, 7.378697629483821e+19, 6.456360425798343e+19, 4.611686018427388e+19, 6.456360425798343e+19, 7.378697629483821e+19, 7.378697629483821e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 7.378697629483821e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 7.378697629483821e+19, 8.301034833169298e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 6.456360425798343e+19, 6.456360425798343e+19, 8.301034833169298e+19, 6.456360425798343e+19, 6.456360425798343e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 5.5340232221128655e+19, 7.378697629483821e+19, 8.301034833169298e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 7.378697629483821e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 4.611686018427388e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 4.611686018427388e+19, 3.6893488147419103e+19, 4.611686018427388e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 4.611686018427388e+19, 4.611686018427388e+19, 3.6893488147419103e+19, 2.7670116110564327e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 7.378697629483821e+19, 6.456360425798343e+19, 7.378697629483821e+19, 8.301034833169298e+19, 9.223372036854776e+19, 9.223372036854776e+19, 1.1068046444225731e+20, 1.1068046444225731e+20, 1.1068046444225731e+20, 1.1068046444225731e+20, 9.223372036854776e+19, 9.223372036854776e+19, 1.0145709240540253e+20, 1.1068046444225731e+20, 1.0145709240540253e+20, 9.223372036854776e+19, 9.223372036854776e+19, 8.301034833169298e+19, 8.301034833169298e+19, 1.0145709240540253e+20, 8.301034833169298e+19, 9.223372036854776e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 7.378697629483821e+19, 7.378697629483821e+19, 8.301034833169298e+19, 8.301034833169298e+19, 6.456360425798343e+19, 8.301034833169298e+19, 7.378697629483821e+19, 6.456360425798343e+19, 8.301034833169298e+19, 8.301034833169298e+19, 9.223372036854776e+19, 9.223372036854776e+19, 9.223372036854776e+19, 9.223372036854776e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 9.223372036854776e+19, 9.223372036854776e+19, 9.223372036854776e+19, 9.223372036854776e+19, 8.301034833169298e+19, 7.378697629483821e+19, 8.301034833169298e+19, 9.223372036854776e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 9.223372036854776e+19, 8.301034833169298e+19, 6.456360425798343e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 1.0145709240540253e+20, 9.223372036854776e+19, 8.301034833169298e+19, 7.378697629483821e+19, 9.223372036854776e+19, 7.378697629483821e+19, 9.223372036854776e+19, 7.378697629483821e+19, 9.223372036854776e+19, 8.301034833169298e+19, 5.5340232221128655e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 9.223372036854776e+19, 8.301034833169298e+19, 9.223372036854776e+19, 9.223372036854776e+19, 1.0145709240540253e+20, 9.223372036854776e+19, 9.223372036854776e+19, 9.223372036854776e+19, 7.378697629483821e+19, 8.301034833169298e+19, 8.301034833169298e+19, 7.378697629483821e+19, 7.378697629483821e+19, 8.301034833169298e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 7.378697629483821e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 6.456360425798343e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 8.301034833169298e+19, 8.301034833169298e+19, 1.0145709240540253e+20, 8.301034833169298e+19, 8.301034833169298e+19, 7.378697629483821e+19, 7.378697629483821e+19, 8.301034833169298e+19, 1.1068046444225731e+20, 1.1068046444225731e+20, 1.1068046444225731e+20, 1.2912720851596686e+20, 1.1068046444225731e+20, 1.1068046444225731e+20, 1.0145709240540253e+20, 1.0145709240540253e+20, 1.1990383647911209e+20, 1.1990383647911209e+20, 1.1990383647911209e+20, 1.1068046444225731e+20, 1.1068046444225731e+20, 9.223372036854776e+19, 9.223372036854776e+19, 9.223372036854776e+19, 8.301034833169298e+19, 8.301034833169298e+19, 1.1068046444225731e+20, 1.1068046444225731e+20, 1.2912720851596686e+20, 1.1068046444225731e+20, 9.223372036854776e+19, 9.223372036854776e+19, 1.0145709240540253e+20, 1.0145709240540253e+20, 1.0145709240540253e+20, 9.223372036854776e+19, 9.223372036854776e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 9.223372036854776e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 6.456360425798343e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 1.1990383647911209e+20, 1.1990383647911209e+20, 1.1990383647911209e+20, 1.1068046444225731e+20, 1.1990383647911209e+20, 1.0145709240540253e+20, 6.456360425798343e+19, 7.378697629483821e+19, 6.456360425798343e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 6.456360425798343e+19, 9.223372036854776e+19, 1.0145709240540253e+20, 1.0145709240540253e+20, 1.0145709240540253e+20, 1.0145709240540253e+20, 1.0145709240540253e+20, 1.1068046444225731e+20, 1.0145709240540253e+20, 9.223372036854776e+19, 9.223372036854776e+19, 1.0145709240540253e+20, 9.223372036854776e+19, 9.223372036854776e+19, 1.0145709240540253e+20, 1.0145709240540253e+20, 9.223372036854776e+19, 1.0145709240540253e+20, 1.0145709240540253e+20, 1.0145709240540253e+20, 9.223372036854776e+19, 1.0145709240540253e+20, 9.223372036854776e+19, 8.301034833169298e+19, 8.301034833169298e+19, 1.0145709240540253e+20, 1.0145709240540253e+20, 8.301034833169298e+19, 9.223372036854776e+19, 9.223372036854776e+19, 1.0145709240540253e+20, 1.1990383647911209e+20, 1.1990383647911209e+20, 1.1068046444225731e+20, 1.1990383647911209e+20, 1.0145709240540253e+20, 1.0145709240540253e+20, 1.0145709240540253e+20, 1.0145709240540253e+20, 1.0145709240540253e+20, 9.223372036854776e+19, 9.223372036854776e+19, 1.0145709240540253e+20, 1.0145709240540253e+20, 9.223372036854776e+19, 1.0145709240540253e+20, 1.1068046444225731e+20, 1.1068046444225731e+20, 1.1068046444225731e+20, 1.0145709240540253e+20, 1.1068046444225731e+20, 1.1068046444225731e+20, 1.1068046444225731e+20, 1.1990383647911209e+20, 1.1068046444225731e+20, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 9.223372036854776e+19, 1.1068046444225731e+20, 1.1068046444225731e+20, 1.1068046444225731e+20, 1.1068046444225731e+20, 1.0145709240540253e+20, 9.223372036854776e+19, 1.0145709240540253e+20, 1.0145709240540253e+20, 1.0145709240540253e+20, 1.0145709240540253e+20, 9.223372036854776e+19, 9.223372036854776e+19, 9.223372036854776e+19, 9.223372036854776e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 9.223372036854776e+19, 8.301034833169298e+19, 8.301034833169298e+19, 1.0145709240540253e+20, 1.1990383647911209e+20, 1.1990383647911209e+20, 1.1990383647911209e+20, 1.1990383647911209e+20, 1.1068046444225731e+20, 1.1990383647911209e+20, 1.1068046444225731e+20, 1.1068046444225731e+20, 1.0145709240540253e+20, 1.0145709240540253e+20, 1.0145709240540253e+20, 1.0145709240540253e+20, 1.0145709240540253e+20, 1.1068046444225731e+20, 1.1990383647911209e+20, 1.1990383647911209e+20, 1.1990383647911209e+20, 1.1990383647911209e+20, 1.1068046444225731e+20, 1.0145709240540253e+20, 1.0145709240540253e+20, 1.2912720851596686e+20, 1.2912720851596686e+20, 1.2912720851596686e+20, 1.0145709240540253e+20, 1.2912720851596686e+20, 1.1068046444225731e+20, 1.0145709240540253e+20, 1.1068046444225731e+20, 1.1068046444225731e+20, 1.0145709240540253e+20, 1.0145709240540253e+20, 1.1068046444225731e+20, 1.0145709240540253e+20, 1.0145709240540253e+20, 1.0145709240540253e+20, 1.1068046444225731e+20, 1.1068046444225731e+20, 1.1990383647911209e+20, 1.1068046444225731e+20, 1.0145709240540253e+20, 1.0145709240540253e+20, 1.0145709240540253e+20, 1.0145709240540253e+20, 1.0145709240540253e+20, 1.1990383647911209e+20, 1.2912720851596686e+20, 9.223372036854776e+19, 1.0145709240540253e+20, 1.1990383647911209e+20, 9.223372036854776e+19, 9.223372036854776e+19, 7.378697629483821e+19, 1.1068046444225731e+20, 1.0145709240540253e+20, 9.223372036854776e+19, 9.223372036854776e+19, 9.223372036854776e+19, 7.378697629483821e+19, 7.378697629483821e+19, 9.223372036854776e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 9.223372036854776e+19, 8.301034833169298e+19, 9.223372036854776e+19, 5.5340232221128655e+19, 3.6893488147419103e+19, 7.378697629483821e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 5.5340232221128655e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 7.378697629483821e+19, 7.378697629483821e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 9.223372036854776e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 6.456360425798343e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 8.301034833169298e+19, 1.1990383647911209e+20, 1.0145709240540253e+20, 1.1990383647911209e+20, 1.1068046444225731e+20, 1.0145709240540253e+20, 1.0145709240540253e+20, 7.378697629483821e+19, 8.301034833169298e+19, 7.378697629483821e+19, 6.456360425798343e+19, 8.301034833169298e+19, 7.378697629483821e+19, 6.456360425798343e+19, 7.378697629483821e+19, 6.456360425798343e+19, 6.456360425798343e+19, 7.378697629483821e+19, 7.378697629483821e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 4.611686018427388e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 7.378697629483821e+19, 7.378697629483821e+19, 8.301034833169298e+19, 7.378697629483821e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 7.378697629483821e+19, 7.378697629483821e+19, 6.456360425798343e+19, 6.456360425798343e+19, 8.301034833169298e+19, 6.456360425798343e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 6.456360425798343e+19, 7.378697629483821e+19, 7.378697629483821e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 6.456360425798343e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 9.223372036854776e+19, 9.223372036854776e+19, 9.223372036854776e+19, 9.223372036854776e+19, 7.378697629483821e+19, 7.378697629483821e+19, 6.456360425798343e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 6.456360425798343e+19, 7.378697629483821e+19, 6.456360425798343e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 6.456360425798343e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 6.456360425798343e+19, 1.0145709240540253e+20, 9.223372036854776e+19, 8.301034833169298e+19, 8.301034833169298e+19, 9.223372036854776e+19, 8.301034833169298e+19, 8.301034833169298e+19, 7.378697629483821e+19, 9.223372036854776e+19, 7.378697629483821e+19, 9.223372036854776e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 4.611686018427388e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 3.6893488147419103e+19, 4.611686018427388e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 6.456360425798343e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 7.378697629483821e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 7.378697629483821e+19, 8.301034833169298e+19, 7.378697629483821e+19, 7.378697629483821e+19, 6.456360425798343e+19, 7.378697629483821e+19, 8.301034833169298e+19, 9.223372036854776e+19, 9.223372036854776e+19, 8.301034833169298e+19, 9.223372036854776e+19, 1.0145709240540253e+20, 1.0145709240540253e+20, 8.301034833169298e+19, 9.223372036854776e+19, 9.223372036854776e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 7.378697629483821e+19, 9.223372036854776e+19, 8.301034833169298e+19, 7.378697629483821e+19]

#ACPM/INSTANCE B1/GEN/OPT :82/TAIL_POP:30
#FLUX-REEL = [83.0, 83.0, 83.0, 83.0, 83.0, 83.0, 83.0, 83.0, 83.0, 83.0, 87.0, 87.0, 87.0, 87.0, 89.0, 87.0, 89.0, 89.0, 102.0, 102.0, 9.223372036854776e+18, 9.223372036854776e+18, 101.0, 102.0, 101.0, 102.0, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 132.0, 132.0, 135.0, 135.0, 9.223372036854776e+18, 132.0, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 145.0, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 159.0, 150.0, 158.0, 159.0, 159.0, 159.0, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 122.0, 122.0, 122.0, 122.0, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 154.0, 133.0, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 1.8446744073709552e+19, 1.8446744073709552e+19, 1.8446744073709552e+19, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 149.0, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 141.0, 9.223372036854776e+18, 132.0, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 143.0, 1.8446744073709552e+19, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 1.8446744073709552e+19, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 1.8446744073709552e+19, 9.223372036854776e+18, 1.8446744073709552e+19, 1.8446744073709552e+19, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 1.8446744073709552e+19, 1.8446744073709552e+19, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 9.223372036854776e+18, 1.8446744073709552e+19, 9.223372036854776e+18, 1.8446744073709552e+19, 1.8446744073709552e+19, 1.8446744073709552e+19, 1.8446744073709552e+19, 1.8446744073709552e+19, 1.8446744073709552e+19, 1.8446744073709552e+19, 1.8446744073709552e+19, 2.7670116110564327e+19, 1.8446744073709552e+19, 1.8446744073709552e+19, 1.8446744073709552e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 1.8446744073709552e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 6.456360425798343e+19, 6.456360425798343e+19, 4.611686018427388e+19, 4.611686018427388e+19, 5.5340232221128655e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 5.5340232221128655e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 6.456360425798343e+19, 6.456360425798343e+19, 4.611686018427388e+19, 4.611686018427388e+19, 5.5340232221128655e+19, 4.611686018427388e+19, 4.611686018427388e+19, 5.5340232221128655e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 7.378697629483821e+19, 6.456360425798343e+19, 3.6893488147419103e+19, 2.7670116110564327e+19, 5.5340232221128655e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 4.611686018427388e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 3.6893488147419103e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 3.6893488147419103e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 4.611686018427388e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 4.611686018427388e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 4.611686018427388e+19, 4.611686018427388e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 1.8446744073709552e+19, 1.8446744073709552e+19, 2.7670116110564327e+19, 1.8446744073709552e+19, 1.8446744073709552e+19, 1.8446744073709552e+19, 1.8446744073709552e+19, 2.7670116110564327e+19, 8.301034833169298e+19, 8.301034833169298e+19, 6.456360425798343e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 7.378697629483821e+19, 7.378697629483821e+19, 8.301034833169298e+19, 8.301034833169298e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 9.223372036854776e+19, 9.223372036854776e+19, 8.301034833169298e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 6.456360425798343e+19, 8.301034833169298e+19, 7.378697629483821e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 7.378697629483821e+19, 7.378697629483821e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 7.378697629483821e+19, 8.301034833169298e+19, 8.301034833169298e+19, 7.378697629483821e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 6.456360425798343e+19, 6.456360425798343e+19, 8.301034833169298e+19, 7.378697629483821e+19, 7.378697629483821e+19, 9.223372036854776e+19, 9.223372036854776e+19, 8.301034833169298e+19, 9.223372036854776e+19, 9.223372036854776e+19, 9.223372036854776e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 9.223372036854776e+19, 8.301034833169298e+19, 6.456360425798343e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 4.611686018427388e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 4.611686018427388e+19, 4.611686018427388e+19, 5.5340232221128655e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 5.5340232221128655e+19, 4.611686018427388e+19, 5.5340232221128655e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 4.611686018427388e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 4.611686018427388e+19, 4.611686018427388e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 4.611686018427388e+19, 6.456360425798343e+19, 6.456360425798343e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 6.456360425798343e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 6.456360425798343e+19, 5.5340232221128655e+19, 6.456360425798343e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 4.611686018427388e+19, 4.611686018427388e+19, 5.5340232221128655e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 2.7670116110564327e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 2.7670116110564327e+19, 1.8446744073709552e+19, 2.7670116110564327e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 2.7670116110564327e+19, 3.6893488147419103e+19, 4.611686018427388e+19, 4.611686018427388e+19, 3.6893488147419103e+19, 4.611686018427388e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 3.6893488147419103e+19, 4.611686018427388e+19, 4.611686018427388e+19, 3.6893488147419103e+19, 5.5340232221128655e+19, 4.611686018427388e+19, 6.456360425798343e+19, 4.611686018427388e+19, 4.611686018427388e+19, 4.611686018427388e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19, 9.223372036854776e+19, 8.301034833169298e+19, 8.301034833169298e+19, 7.378697629483821e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 9.223372036854776e+19, 9.223372036854776e+19, 1.0145709240540253e+20, 8.301034833169298e+19, 9.223372036854776e+19, 9.223372036854776e+19, 9.223372036854776e+19, 9.223372036854776e+19, 9.223372036854776e+19, 9.223372036854776e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 8.301034833169298e+19, 6.456360425798343e+19, 6.456360425798343e+19, 3.6893488147419103e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 5.5340232221128655e+19, 6.456360425798343e+19, 7.378697629483821e+19, 7.378697629483821e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 6.456360425798343e+19, 7.378697629483821e+19, 7.378697629483821e+19, 7.378697629483821e+19]
# ELI = [83.0, 83.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0, 82.0]
