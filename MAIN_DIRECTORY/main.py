
from analyses import *



def optimal_value(type_inst):
    
    with open("../OPT/"+type_inst+".txt","r") as f:
        lines = f.readlines()
    return [int(i) for i in lines]
            


    
if __name__ == '__main__' :
    
    probaMinGene = 0.2
    probaMaxGene = 0.5
    taillePopulation = 1000
    nombreDeGenerations = 120
    pMutationMin = 0.01
    pMutationMax = 0.04
    probaMutation = random.uniform(pMutationMin, pMutationMax)
    probaCroisement = 0.20
    probaMinRandomisaton = 0.05
    probaMaxRandomisation = 0.2
    OPT_VALUES = optimal_value("B")
    filename_stp = "../B/b01.stp"


#################################################################################################################################################################
######################################_____COMPARAISON ENTRE REMPLACEMENT GENERATIONNEL ET ELITISTE (POPULATION INITIALE ALEATOIRE)_____#########################################################
#################################################################################################################################################################

#    for t in ['B/b']:#,'C/c','D/d','E/e'] :
#        for i in range(1,2):#len(OPT_VALUES)):
#            filename_stp = "../"+t+str(i).zfill(2)+".stp"
#            compare_remplacement_generationnel_et_elitiste_population_aleatoire(OPT_VALUES[i-1],filename_stp,taillePopulation,nombreDeGenerations,probaMinGene,probaMaxGene,probaMutation,probaCroisement)
##
#
##########################################################################################################################################################################
#############################_____COMPARAISON ENTRE GENERATION POPULATION INITIALE ALEATOIRE ET AVEC RANDOMISATION HEURISTIQUE_____#######################################
##########################################################################################################################################################################


#    for t in ['B/b']:#,'C/c','D/d','E/e'] :
#        for i in range(1,2):#len(OPT_VALUES)):
#            filename_stp = "../"+t+str(i).zfill(2)+".stp"    
#            compare_population_aleatoire_et_r_heuristiques(filename_stp,taillePopulation,probaMinRandomisaton,probaMaxRandomisation,probaMinGene,probaMaxGene)
#    
#################################################################################################################################################################
###############################_____EVALUATION RECHERCHE LOCALE AVEC RANDOMISATION HEURISTIQUE DE LA POPULATION INITIALE _____#########################################################
#################################################################################################################################################################
        
    
#    for t in ['B/b']:#,'C/c','D/d','E/e'] :
#        for i in range(2,3):#len(OPT_VALUES)):
#            filename_stp = "../"+t+str(i).zfill(2)+".stp"
#            evaluation_recherche_locale_randomisation_h(OPT_VALUES[i-1],filename_stp,taillePopulation,probaMinRandomisaton,probaMaxRandomisation)
            
            
            
            
            
            
            
            
            
            
            
            