import matplotlib.pyplot as plt
from threading import Thread, Event


import time
import csv

import os
from Whole_Graph import *



def optimal_value(type_inst):
    
    with open("../OPT/"+type_inst+".txt","r") as f:
        lines = f.readlines()
    return [int(i) for i in lines]
            


def thread_timeout(target, args, timeLimit):
    stop_event = Event()
    args.append(stop_event)
    try:
        thread = Thread(target=target,args=tuple(args))
        thread.start()
        thread.join(timeout=timeLimit)
        stop_event.set()
    except Exception:
        pass    
    
############################################################################################################################################################################
############################################################################################################################################################################

def appliquer_algo_genetique_aleatoire(filename_stp, opt_value, taillePopulation,probaMutation,probaMinGene=0.2,probaMaxGene=0.5,probaCroisement=0.2,timeLimit=300): #RETIRER LES VALEURS PAR DEFAUT

    best_individus = list()
    G = Whole_Graph(filename_stp)
    t = time.time()
    G.generer_N_individus_aleatoire(taillePopulation,probaMinGene,probaMaxGene,Event())
    timeLimit = timeLimit - (time.time() - t)

    
    thread_timeout(G.remplacement_generationnel, [opt_value, probaMutation,probaCroisement],timeLimit)
    best_individus.append(G.best_individu.get_fitness())
    print("GEN", str(G.ListeIndividus[G.calculer_meilleur_fitness()]))
  
    best_individus.append(G.get_time_max_fitness())

    G.restaurer_population_de_base()

    thread_timeout(G.remplacement_elitiste, [opt_value, probaMutation,probaCroisement],timeLimit)
    best_individus.append(G.best_individu.get_fitness())
    print("ELI", str(G.ListeIndividus[G.calculer_meilleur_fitness()]))

    best_individus.append(G.get_time_max_fitness())

    return tuple(best_individus)


def appliquer_algo_genetique_groupe_instances_aleatoire(dirname, Liste_opt, taillePopulation,probaMutationMin=0.01,probaMutationMax=0.04,probaMinGene=0.2,probaMaxGene=0.5,probaCroisement=0.2,timeLimit=300):
   
    probaMutation = random.uniform(pMutationMin, pMutationMax)
    
    nb_instances = len(os.listdir(dirname))

    Liste_opt_gene = [(-1,-1) for i in range(nb_instances)]
    Liste_opt_eli = [(-1,-1) for i in range(nb_instances)]
    
    type_inst = dirname.split("/")[1]
    with open("analyse/eval_"+type_inst+"_algo_gen_alea_N_%d"%taillePopulation+".csv",'w') as f:
        entete_csv = ["NUM_INST"]+["GEN","TIME_GEN"] + ["ELI","TIME_ELI"] + ["OPT"]
        
        writer = csv.DictWriter(f, fieldnames=entete_csv)
        writer.writeheader()     
        csvLine = dict()
        
    
        for filename_stp in os.listdir(dirname):
            number = int(filename_stp[1:3])
            print"_______________ FILE {} _______________".format(filename_stp)
            gen,time_gen, eli,time_eli = appliquer_algo_genetique_aleatoire(dirname+"/"+filename_stp, Liste_opt[number-1], taillePopulation, probaMutation, probaMinGene, probaMaxGene, probaCroisement, timeLimit)
            Liste_opt_gene.insert(number,(gen,time_gen))
            Liste_opt_eli.insert(number,(eli,time_eli))
    
            csvLine["NUM_INST"] = number
            csvLine["GEN"] = gen
            csvLine["TIME_GEN"] = time_gen
            csvLine["ELI"] = eli
            csvLine["TIME_ELI"] = time_eli
            csvLine["OPT"] = Liste_opt[number-1]

            writer.writerow(csvLine)
            f.flush()
            csvLine=dict()   
   


############################################################################################################################################################################
############################################################################################################################################################################


def appliquer_algo_genetique_randomise(filename_stp, opt_value, taillePopulation, heuristique_name,probaMutation,probaMinRandomisaton=0.05,probaMaxRandomisation=0.2,probaCroisement=0.2,timeLimit=300):    #RETIRER LES VALEURS PAR DEFAUT

    best_individus = list()
    
    G = Whole_Graph(filename_stp)
    
    if heuristique_name == "PCM":
        heuristique = G.heuristique_PCM
    else:
        heuristique = G.heuristique_ACPM
   
    t = time.time()
    G.generer_N_individus_heuristique(taillePopulation,heuristique,probaMinRandomisaton,probaMaxRandomisation)
    timeLimit = timeLimit - (time.time() - t)
    print " ___GENERATION INDIVIDU HEURISTIQUE "+heuristique_name+"  en %.2f"%(time.time() - t)+" sec\n"

    print"_____________________GENERATIONNEL_________________________\n\n"
    thread_timeout(G.remplacement_generationnel, [opt_value,probaMutation,probaCroisement],timeLimit)
    best_individus.append(G.best_individu.get_fitness())
    
    best_individus.append(G.get_time_max_fitness())

    G.restaurer_population_de_base()
    print"_____________________ELITISTE_________________________\n\n"

    thread_timeout(G.remplacement_elitiste, [opt_value,probaMutation,probaCroisement],timeLimit)
    best_individus.append(G.best_individu.get_fitness())
    
    
    best_individus.append(G.get_time_max_fitness())

    return tuple(best_individus)    



def appliquer_algo_genetique_groupe_instances_randomise(dirname, Liste_opt, taillePopulation, probaMutationMin=0.01,probaMutationMax=0.04,probaMinRandomisation=0.05,probaMaxRandomisation=0.2,probaCroisement=0.2,timeLimit=300):
    probaMutation = random.uniform(pMutationMin, pMutationMax)

    nb_instances = len(os.listdir(dirname))
    Liste_opt_gene = [(-1,-1) for i in range(nb_instances)]
    Liste_opt_eli = [(-1,-1) for i in range(nb_instances)]
    
    heuristique = "ACPM"
    
    type_inst = dirname.split("/")[1]
    with open("analyse/eval_"+type_inst+"_algo_gen_"+heuristique+"_N_%d"%taillePopulation+".csv",'w') as f:
        
        entete_csv = ["NUM_INST"] + ["GEN","TIME_GEN"] + ["ELI","TIME_ELI"] + ["OPT"]
        writer = csv.DictWriter(f, fieldnames=entete_csv)
        writer.writeheader()     
        csvLine = dict()
        
        
        for filename_stp in os.listdir(dirname):
            number = int(filename_stp[1:3])
            print "\n\n___________FILE {}  OPT {}________________\n".format(filename_stp,Liste_opt[number - 1])
            gen,time_gen, eli,time_eli = appliquer_algo_genetique_randomise(dirname+"/"+filename_stp, Liste_opt[number - 1], taillePopulation, heuristique,  probaMutation, probaMinRandomisation, probaMaxRandomisation, probaCroisement, timeLimit)
            Liste_opt_gene.insert(number,(gen,time_gen))
            Liste_opt_eli.insert(number,(eli,time_eli))
            print " GENERATIONNEL {} ({} sec) \t\t ELITISTE {} ({} sec)\n\n".format(gen,time_gen,eli,time_eli)
            csvLine["NUM_INST"] = number
            csvLine["GEN"] = gen
            csvLine["TIME_GEN"] = time_gen
            csvLine["ELI"] = eli
            csvLine["TIME_ELI"] = time_eli 
            csvLine["OPT"] = Liste_opt[number]

            writer.writerow(csvLine)
            f.flush()
            csvLine=dict()   


############################################################################################################################################################################
############################################################################################################################################################################


def compare_population_aleatoire_et_r_heuristiques(filename_stp,taillePopulation,probaMinRandomisaton=0.05,probaMaxRandomisaton=0.2,probaMinGene=0.2,probaMaxGene=0.5,timeLimit=300):
    
    parameters = list()

    G = Whole_Graph(filename_stp)
    print " INDIVIDU AVEC {} NOEUDS {} EDGE et  {} TERMINAUX \n".format(G.NumNodes,G.NumEdges,G.NumTerminals)
    t = time.time()
    # args = [taillePopulation,G.heuristique_PCM,probaMinRandomisaton,probaMaxRandomisaton]
    # thread_timeout(G.generer_N_individus_heuristique,args,timeLimit)
    G.generer_N_individus_heuristique(taillePopulation,G.heuristique_PCM,probaMinRandomisaton,probaMaxRandomisaton)
    t_pcm = time.time() - t
    # if len(G.ListeIndividus) == 0:
    #     parameters.append(float("inf"))
    # else:
    parameters.append(G.ListeIndividus[G.calculer_meilleur_fitness()].get_fitness())
    print "________________PCM DONE en {} sec_______________\n".format(t_pcm)
    parameters.append(t_pcm)
    
    G.reset_listIndividus()

    G.reinitialiser_dictValuations()

    t = time.time()
    # args = [taillePopulation,probaMinGene,probaMaxGene]
    # thread_timeout(G.generer_N_individus_aleatoire,args,timeLimit)
    G.generer_N_individus_aleatoire(taillePopulation,probaMinGene,probaMaxGene)
    t_alea = time.time() - t
    # if len(G.ListeIndividus) == 0:
    #     parameters.append(float("inf"))
    # else:
    parameters.append(G.ListeIndividus[G.calculer_meilleur_fitness()].get_fitness())
    parameters.append(t_alea)
    print "________________ALEA DONE en {} sec_______________\n".format(t_alea)

    G.reset_listIndividus()
    
    t = time.time()
    # args = [taillePopulation,G.heuristique_ACPM,probaMinRandomisaton,probaMaxRandomisaton]
    # thread_timeout(G.generer_N_individus_heuristique,args,timeLimit)
    G.generer_N_individus_heuristique(taillePopulation,G.heuristique_ACPM,probaMinRandomisaton,probaMaxRandomisaton)
    t_acpm = time.time() - t
    # if len(G.ListeIndividus) == 0:
    #     parameters.append(float("inf"))
    # else:
    parameters.append(G.ListeIndividus[G.calculer_meilleur_fitness()].get_fitness())
    parameters.append(t_acpm)
    print "________________ACPM DONE en {} sec_______________\n".format(t_acpm)

    return tuple(parameters)
            
            
def appliquer_comparaison_population_initiale_groupe_instances(dirname,Liste_opt,taillePopulation,probaMinRandomisaton=0.05,probaMaxRandomisaton=0.2,probaMinGene=0.2,probaMaxGene=0.5):

    nb_instances = len(os.listdir(dirname))
    Liste_opt_pcm = [(-1,-1) for i in range(nb_instances)]
    Liste_opt_alea = [(-1,-1) for i in range(nb_instances)]
    Liste_opt_acpm = [(-1,-1) for i in range(nb_instances)]
    
    type_inst = dirname.split("/")[1]
    with open("analyse/compare_alea_randh_"+type_inst+"_N_%d"%taillePopulation+".csv",'w') as f:
        print "___POPULATION TAILLE %d\n"%taillePopulation
        entete_csv = ["NUM_INST"] + ["ALEA","TIME_ALEA"] + ["PCM","TIME_PCM"] + ["ACPM","TIME_ACPM"] + ["OPT"]
        writer = csv.DictWriter(f, fieldnames=entete_csv)
        writer.writeheader()     
        csvLine = dict()
        

        for filename_stp in os.listdir(dirname):
            number = int(filename_stp[1:3])
            # if number in [20,5,2]:
            #     continue
            print "\n\n___________FILE {}  OPT {}________________\n".format(filename_stp,Liste_opt[number - 1])
            pcm,t_pcm,alea,t_alea,acpm,t_acpm = compare_population_aleatoire_et_r_heuristiques(dirname+"/"+filename_stp,taillePopulation,probaMinRandomisaton,probaMaxRandomisaton,probaMinGene,probaMaxGene)
            Liste_opt_pcm.insert(number,(pcm,t_pcm))
            Liste_opt_alea.insert(number,(alea,t_alea))
            Liste_opt_acpm.insert(number,(acpm,t_acpm))

            csvLine["NUM_INST"] = number
            csvLine["ALEA"] = alea
            csvLine["TIME_ALEA"] = t_alea 
            csvLine["PCM"] = pcm
            csvLine["TIME_PCM"] = t_pcm 
            csvLine["ACPM"] = acpm
            csvLine["TIME_ACPM"] = t_acpm 
            csvLine["OPT"] = Liste_opt[number - 1]
            
            
            writer.writerow(csvLine)
            f.flush()
            csvLine=dict()   
#        for i in range(nb_instances):
#            csvLine["ALEA"] = Liste_opt_alea[i][0]
#            csvLine["TIME_ALEA"] = Liste_opt_alea[i][1]
#            csvLine["PCM"] = Liste_opt_pcm[i][0]
#            csvLine["TIME_PCM"] = Liste_opt_pcm[i][1]
#            csvLine["ACPM"] = Liste_opt_acpm[i][0]
#            csvLine["TIME_ACPM"] = Liste_opt_acpm[i][1]
#            csvLine["OPT"] = Liste_opt[i]



############################################################################################################################################################################
############################################################################################################################################################################


def recherche_locale_rand_heuristique_population(filename_stp,opt_value,taillePopulation,heuristique,probaMinRandomisaton=0.05,probaMaxRandomisation=0.2,timeLimit=300):
  
    G = Whole_Graph(filename_stp)
    
    if heuristique == "PCM":
        heuristique = G.heuristique_PCM
    else:
        heuristique = G.heuristique_ACPM
        
    t = time.time()
    G.generer_N_individus_heuristique(taillePopulation,heuristique,probaMinRandomisaton,probaMaxRandomisation)
    timeLimit = timeLimit - (time.time() - t)
    try:
        thread_timeout(G.recherche_locale, [opt_value],timeLimit)
    except:
        pass
    return (G.best_individu.get_fitness(),G.get_time_max_fitness())



def appliquer_recherche_locale_groupe_instances_population_rh(dirname,Liste_opt,taillePopulation,probaMinRandomisaton=0.05,probaMaxRandomisation=0.2,timeLimit=300):
    
    nb_instances = len(os.listdir(dirname))
    Liste_opt_rl = [(-1,-1) for i in range(nb_instances)]
    
    
    type_inst = dirname.split("/")[1]
    with open("analyse/eval_recherche_locale_ACPM_"+type_inst+"_N_%d"%taillePopulation+".csv",'w') as f:

        entete_csv = ["NUM_INST"] + ["RL","TIME_RL"] + ["OPT"]
        writer = csv.DictWriter(f, fieldnames=entete_csv)
        writer.writeheader()     
        csvLine = dict()
        
        
        for filename_stp in os.listdir(dirname):
            number = int(filename_stp[1:3])
            fitness_rl,t_rl = recherche_locale_rand_heuristique_population(dirname+"/"+filename_stp,Liste_opt[number],taillePopulation,"ACPM",probaMinRandomisaton,probaMaxRandomisation)
            Liste_opt_rl.insert(number,(fitness_rl,t_rl))
            
#        for i in range(nb_instances):
            csvLine["NUM_INST"] = number
            csvLine["RL"] = fitness_rl
            csvLine["TIME_RL"] = t_rl
            csvLine["OPT"] = Liste_opt[number - 1]
            writer.writerow(csvLine)
            f.flush()
            csvLine=dict()   

############################################################################################################################################################################
############################################################################################################################################################################







pMutationMin = 0.01
pMutationMax = 0.04
probaMutation = random.uniform(pMutationMin, pMutationMax)
taillePopulation = 10
probaMinRandomisaton = 0.05
probaMaxRandomisation = 0.2
probaMutationMin=0.01
probaMutationMax=0.04
probaCroisement = 0.2

# type_inst = "B"
# num = '01'
# filename_stp = "../B/b"+num+".stp"


# G = Whole_Graph(filename_stp)
# print "_______________V = {}, E = {} , T = {}   OPT VALUE {} ____________________\n".format(G.NumNodes,G.NumEdges,G.NumTerminals,optimal_value(type_inst)[19])
# G.generer_N_individus_aleatoire(taillePopulation,0.2,0.5)
# G.generer_N_individus_heuristique(taillePopulation,G.heuristique_PCM,probaMinRandomisaton,probaMaxRandomisation)
# compare_population_aleatoire_et_r_heuristiques(filename_stp,taillePopulation)
# G.remplacement_generationnel(optimal_value(type_inst)[1],random.uniform(probaMinRandomisaton,probaMaxRandomisation),probaCroisement,stop_event)
# print " ____best fitness %d"%G.best_individu.get_fitness()
# appliquer_comparaison_population_initiale_groupe_instances("../"+type_inst,optimal_value(type_inst),taillePopulation)
# t = time.time()
# dic = G.heuristique_PCM()
# print "time : ",time.time() - t
# indiv = Graphe_Individu(G,dic)
# print "fitness ",indiv.get_fitness()

# appliquer_algo_genetique_aleatoire(filename_stp, optimal_value(type_inst)[int(num)-1], taillePopulation,probaMutation)

#
for t in ["B","C","D","E"]:
   dirname = "../"+t

   timeLimit=300

   # try:
   #     print "ALGO GENETIQUE ALEATOIRE GROUPE INSTANCE "+t
   #     appliquer_algo_genetique_groupe_instances_aleatoire(dirname,optimal_value(t), taillePopulation,timeLimit=timeLimit)
   # except:
   #     print "error in type "+t
   #     pass
    
   # try:
   #     print "ALGO GENETIQUE ALEATOIRE GROUPE INSTANCE "+type_inst
   #     appliquer_recherche_locale_groupe_instances_population_rh(dirname,optimal_value(type_inst),taillePopulation,timeLimit=timeLimit)
   # except:
   #     pass
# #
#    try:
#        appliquer_algo_genetique_groupe_instances_randomise(dirname,optimal_value(type_inst), taillePopulation,timeLimit=timeLimit)
#    except:
#        pass
# #
   try:
    print "COMPARAISON POPULATION ALEA ET RANDOM HEURISTIC    GROUPE INSTANCE :"+t
    appliquer_comparaison_population_initiale_groupe_instances(dirname,optimal_value(t),taillePopulation)
   except:
       print "error type "+t
       pass
    
    
