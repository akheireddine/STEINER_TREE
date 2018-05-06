def evaluation_recherche_locale_randomisation_h(opt_value_instance,filename_stp,taillePopulation,probaMinRandomisaton=0.05,probaMaxRandomisation=0.2):
  
    G = Whole_Graph(filename_stp)
    
    G.generer_N_individus_heuristique(taillePopulation,G.heuristique_ACPM,probaMinRandomisaton,probaMaxRandomisation)
    try:
        th = Process(target=G.recherche_locale)
        th.start()
        th.join(timeout=3)
        th.terminate()
    except Exception:
        pass
#    print G.ListeIndividusRLocale
    y_fitness_RL = G.ListeIndividusRLocale
    x = list(range(1,len(y_fitness_RL)+1))
    
    plt.figure(1,figsize=(30,20))
    plt.plot(x,[opt_value_instance for i in x],c="red")
    plt.scatter(x,y_fitness_RL, c ="green", alpha=0.5)
    plt.title(" Recherche Locale sur l'instance "+G.get_name()+" Randomisation(PCM) (population taille%d"%taillePopulation)
    plt.xlabel(" i-eme iteration apres comvergence")
    plt.ylabel(" Valeur de la fitness optimale RL")
    
    plt.show()
    namef = "analyse/eval_RLR_"+G.get_name()+"_N%d"%taillePopulation+".png"
    plt.savefig(namef,dpi=800)


def compare_population_aleatoire_et_r_heuristiques(filename_stp,N,probaMinRandomisaton,probaMaxRandomisaton,probaMinGene,probaMaxGene):
    
    x = list(range(1,N+1))
    y_alea = list()
    y_heuristic_pcm = list()
    y_heuristic_acpm = list()
    G = Whole_Graph(filename_stp)

    G.generer_N_individus_heuristique(N,G.heuristique_PCM,probaMinRandomisaton,probaMaxRandomisaton)
    y_heuristic_pcm = [individu.get_fitness() for individu in G.get_listIndividus()]
#    x_pcm = [ i for i in range(1,N+1) if y_heuristic_pcm[i-1]<1e+3]
#    y_heuristic_pcm = [ i for i in y_heuristic_pcm if i<1e+3]

    G.reset_listIndividus()

    G.generer_N_individus_aleatoire(N,probaMinGene,probaMaxGene)
    y_alea = [ individu.get_fitness() for individu in G.get_listIndividus()]
#    x_alea = [ i for i in range(1,N+1) if y_alea[i-1]<1e+3]
#    y_alea = [ i for i in y_alea if i<1e+3]

    G.reset_listIndividus()

    G.generer_N_individus_heuristique(N,G.heuristique_ACPM,probaMinRandomisaton,probaMaxRandomisaton)
    y_heuristic_acpm = [individu.get_fitness() for individu in G.get_listIndividus()]
#    x_acpm = [ i for i in range(1,N+1) if y_heuristic_acpm[i-1]<1e+3]
#    y_heuristic_acpm = [ i for i in y_heuristic_acpm if i<1e+3]


    plt.figure(1,figsize=(30,30))
    plt.title(" Difference entre generation de population initiale avec randomisation heuristique ET aleatoire")
    plt.xlabel(" Num individu")
    plt.ylabel(" Valeur retourne")

    h1 = plt.scatter(x,y_heuristic_pcm, c="blue", alpha=0.5)
    for i in range(len(x)):
        value = str(y_heuristic_pcm[i])
        plt.annotate(value,(x[i],y_heuristic_pcm[i]))

    h2 = plt.scatter(x,y_alea, c ="green", alpha=0.5)
    for i in range(len(x)):
        if y_heuristic_pcm[i] != y_alea[i]:
            value = str(y_alea[i])
            plt.annotate(value,(x[i],y_alea[i]))

    h3 = plt.scatter(x,y_heuristic_acpm, c ="pink", alpha=0.5)
    for i in range(len(x)):
        if y_heuristic_acpm[i] not in [y_heuristic_pcm[i],y_alea[i]]:
            value = str(y_heuristic_acpm[i])
            plt.annotate(value,(x[i],y_heuristic_acpm[i]))

    
    plt.legend((h1,h2,h3),("heuristique PCM","Aleatoire","heuristique ACPM"))
    plt.show()
    namef = "analyse/"+filename_stp+".png"
    plt.savefig(namef,dpi=800)

def whiletrue(f):
    while True:
        f()
    
    
def compare_remplacement_generationnel_et_elitiste_population_aleatoire(opt_value_instance,filename_stp,taillePopulation,nombreDeGenerations,probaMinGene,probaMaxGene, probaMutation, probaCroisement):
    
        
    
    step = 3 #echantillo
    x = list(range(1,nombreDeGenerations+1,step))
    
    G = Whole_Graph(filename_stp)

    G.generer_N_individus_aleatoire(taillePopulation,probaMinGene,probaMaxGene)
    
    y_generationnel = G.remplacement_generationnel(taillePopulation,nombreDeGenerations,probaMinGene,probaMaxGene,probaMutation,probaCroisement)[:-1]
    
    G.reset_listIndividus()
    
    
    y_elitiste = G.remplacement_elitiste(taillePopulation,nombreDeGenerations,probaMinGene,probaMaxGene,probaMutation,probaCroisement)[:-1]
    
    y_generationnel = [ y_generationnel[i] for i in range(0,len(y_generationnel),step)]
    y_elitiste = [y_elitiste[i] for i in range(0,len(y_elitiste),step)]
    
    plt.figure(1,figsize=(30,20))
    plt.title(" Comparaison entre remplacement generationnel et elitiste A (population %d"%taillePopulation+",#generation %d"%nombreDeGenerations+")")
    plt.xlabel(" i-eme generation")
    plt.ylabel(" Valeur retourne par les deux methodes de remplacement")
    
    h1, = plt.plot(x,y_generationnel,c="blue")
    h2, = plt.plot(x,y_elitiste,c="green")
    plt.legend((h1,h2),("remplacement generationnel","remplacement elitiste"))



    val_generationnel = list()
    val_elitiste= list()
    
    s1 = plt.scatter(x,y_generationnel, c="blue", alpha=0.5)
    for i in range(len(x)):
        if y_generationnel[i] < opt_value_instance *1e+3:
            value = str(y_generationnel[i])
            plt.annotate(value,(x[i],y_generationnel[i]))
            if y_generationnel[i] == opt_value_instance:
                val_generationnel.append(x[i])
#                plt.scatter([i+1],opt_value_instance,s=50,c="red",marker="X")
        
    s2 = plt.scatter(x,y_elitiste, c ="green", alpha=0.5)
    for i in range(len(x)):
        if y_elitiste[i] < opt_value_instance *1e+3:
            value = str(y_elitiste[i])
            plt.annotate(value,(x[i],y_elitiste[i]))
            if y_elitiste[i] == opt_value_instance:
#                plt.scatter([i+1],opt_value_instance,s=50,c="cyan",marker="X")    
                val_elitiste.append(x[i])
                
    plt.scatter(val_generationnel,[opt_value_instance for i in range(len(val_generationnel))],s=50,c="pink",marker="*")
    plt.scatter(val_elitiste,[opt_value_instance for i in range(len(val_elitiste))],s=50,c="red",marker="*")
    
    plt.show()
    namef = "analyse/gen_eli_alea_"+G.get_name()+"_N%d"%taillePopulation+"_G%d"%nombreDeGenerations+".png"
    plt.savefig(namef,dpi=1000)
    
    
    
    
def compare_remplacement_generationnel_et_elitiste_population_RH(opt_value_instance,filename_stp,taillePopulation,nombreDeGenerations,probaMinGene,probaMaxGene, probaMutation, probaCroisement,probaMinRandomisation,probaMaxRandomisation):
    step = 3 #echantillo
    x = list(range(1,nombreDeGenerations+1,step))
    
    G = Whole_Graph(filename_stp)

    G.generer_N_individus_heuristique(taillePopulation,G.heuristique_PCM(),probaMinRandomisation,probaMaxRandomisation)
   
    y_generationnel = G.remplacement_generationnel(taillePopulation,nombreDeGenerations,probaMinGene,probaMaxGene,probaMutation,probaCroisement)[:-1]
    y_elitiste = G.remplacement_elitiste(taillePopulation,nombreDeGenerations,probaMinGene,probaMaxGene,probaMutation,probaCroisement)[:-1]
    
    y_generationnel = [ y_generationnel[i] for i in range(0,len(y_generationnel),step)]
    y_elitiste = [y_elitiste[i] for i in range(0,len(y_elitiste),step)]
    
    plt.figure(1,figsize=(30,20))
    plt.title(" Comparaison entre remplacement generationnel et elitiste RH (population %d"%taillePopulation+",#generation %d"%nombreDeGenerations+")")
    plt.xlabel(" i-eme generation")
    plt.ylabel(" Valeur retourne par les deux methodes de remplacement")
    
    h1, = plt.plot(x,y_generationnel,c="blue")
    h2, = plt.plot(x,y_elitiste,c="green")
    plt.legend((h1,h2),("remplacement generationnel","remplacement elitiste"))



    val_generationnel = list()
    val_elitiste= list()
    
    s1 = plt.scatter(x,y_generationnel, c="blue", alpha=0.5)
    for i in range(len(x)):
        if y_generationnel[i] < opt_value_instance *1e+3:
            value = str(y_generationnel[i])
            plt.annotate(value,(x[i],y_generationnel[i]))
            if y_generationnel[i] == opt_value_instance:
                val_generationnel.append(x[i])
#                plt.scatter([i+1],opt_value_instance,s=50,c="red",marker="X")
        
    s2 = plt.scatter(x,y_elitiste, c ="green", alpha=0.5)
    for i in range(len(x)):
        if y_elitiste[i] < opt_value_instance *1e+3:
            value = str(y_elitiste[i])
            plt.annotate(value,(x[i],y_elitiste[i]))
            if y_elitiste[i] == opt_value_instance:
#                plt.scatter([i+1],opt_value_instance,s=50,c="cyan",marker="X")    
                val_elitiste.append(x[i])
                
    plt.scatter(val_generationnel,[opt_value_instance for i in range(len(val_generationnel))],s=50,c="pink",marker="*")
    plt.scatter(val_elitiste,[opt_value_instance for i in range(len(val_elitiste))],s=50,c="red",marker="*")
    
    plt.show()
    namef = "analyse/gen_eli_"+G.get_name()+"_N%d"%taillePopulation+"_G%d"%nombreDeGenerations+".png"
    plt.savefig(namef,dpi=1000)    
    
    
