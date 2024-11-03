""""
-Pedro Paiva , nº 96900
-João Gonçalves, nº 96247

**Applied Computational Intelligence 2024/2025**

Project 2 : Travelling Salesman Problem using EAs

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from deap import creator,tools,base,algorithms
from deap.benchmarks.tools import hypervolume
import random
from evaluate import * #contains evaluation functions for the many optimization problems
import argparse #for command line arguments
import functools #for partials
from pymoo.indicators.hv import HV
import statistics


"""
Plots the best solution given by the EA

Inputs:
-points : path(solution) and coordinates fo all cities(xy.csv)
-path : sequence of visited cities (coordinates of city i can be obtained by points[i])

Output:
Plots the TSP sequence

"""

def TSPmap(path,all=False):
    
    #read xy.csv
    coordinates=pd.read_csv("xy.csv",index_col="City")
    
    #path has the following structure:
    #path=[[city1,transport1],[city2,transport2],...]
    
    if(all==False):
        colors={"plane":"g","bus":"g","train":"g"} 
    else:
        colors={"plane":"r","bus":"g","train":"b"}    
    
    #extract coordinates from the cities in the path
    x=[]
    y=[]
    for individual in path:
        city=individual[0]
        x.append(coordinates.loc[city,"Longitude"])
        y.append(coordinates.loc[city,"Latitude"])
    
    plt.plot(x, y, 'co')
    
    #set scale for arrows
    scale = 0.5*float(max(max(x),max(y)))/float(100)

    for i, city in enumerate(path):
        plt.text(x[i], y[i],i+1, fontsize=10, ha='right', va='bottom', color="black")
        
    # Draw the path for the TSP problem
    for i,city in zip(range(len(x)-1),path):
       plt.arrow(x[i], y[i], (x[i+1] - x[i]), (y[i+1] - y[i]), head_width = scale,
                color=colors[city[1]], length_includes_head = True)
        
    #connect first and last visited city (obtain initial city)
    plt.arrow(x[-1], y[-1], (x[0] - x[-1]), (y[0] - y[-1]), head_width = scale, 
            color=colors[path[-1][1]], length_includes_head=True)

    #Set axis too slitghtly larger than the set of x and y
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Solution of TSP for {len(path)} cities")
    plt.tight_layout()
    plt.grid()
    plt.show()
    return


# Generate a random city
# Ensures each individual has each city only once
def random_city(cities):
    return random.sample(cities, 1)[0]

# Generates a random transport 
def random_transport(transports):
    return random.choice(transports)

#Creates an individual
#random city order, with random transport
def random_individual(cities,transports):
    city_list = random.sample(cities, len(cities))  # Random city order
    individual = [[city_list[i], random_transport(transports)] for i in range(len(city_list))]
    return creator.Individual(individual)  # wrap as creator


#auxiliary function for Order Crossover (OX)
#Fill remaining cities in the other parent
def fill_individual(offspring, parent, start):
    size = len(parent)
    idx = start
    for city in parent:
        if city not in offspring:
            if idx >= size:
                idx = 0
            offspring[idx] = city
            idx += 1

#OX : order crossover -> good in TSP, maintains validity of a solution
#   -In-> 2 individals
#   -Out-> Their offsprings
def cxOrder(ind1, ind2):
    """
    Apply order crossover (OX) to ensure cities do not repeat in the offspring.
    """
    size = len(ind1)
    
    # Choose two crossover points
    cxpoint1 = random.randint(0, size - 1)
    cxpoint2 = random.randint(0, size - 1)
    
    if cxpoint1 > cxpoint2:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    
    # Create offspring by taking the cities between the crossover points from one parent
    offspring1 = [None] * size
    offspring2 = [None] * size
    offspring1[cxpoint1:cxpoint2] = ind1[cxpoint1:cxpoint2]
    offspring2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2]
    
    # Fill in the remaining cities in the order they appear in the other parent
    fill_individual(offspring1, ind2, cxpoint2)
    fill_individual(offspring2, ind1, cxpoint2)
    
    return offspring1, offspring2

def inverse_mutation(individual):
    """
    Performs inverse mutation on a TSP path
    
    Parameters:
    - individual : list of pairs (city,transport) representing the TSP solution
    
    Returns:
    - A new path with a portion of the cities reversed
    """
    
    # Step 1: Select two random points in the tour
    point1 = random.randint(0, len(individual) - 1)
    point2 = random.randint(0, len(individual) - 1)
    
    # Ensure point1 is less than point2
    if point1 > point2:
        point1, point2 = point2, point1
    
    # Step 2: Reverse the sublist between the two points
    new_path = individual[:point1] + individual[point1:point2 + 1][::-1] + individual[point2 + 1:]
    return new_path

# Mutation: Swap cities or change transport mode, the last only if transport="all"
def mutate_path_and_or_transport(individual, all=False):
    """
    Perform mutation on an individual.
    
    - If all is True, mutate the transport mode
    - Otherwise, only swap cities (keep transport mode constant)
    
    Returns:
    - Mutated individual
    
    """
    transports=["bus","train","plane"]
    
    if all==True:
        # Case 1: All transport modes are allowed, mutate transport
        # Change the transport mode for one leg of the journey
        idx = random.randint(0, len(individual) - 1)
        new_transport = random.choice(transports)  # Pick a new transport
        individual[idx] = [individual[idx][0], new_transport]
        #swap two cities
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = [individual[idx2][0], individual[idx1][1]], [individual[idx1][0], individual[idx2][1]]
    else:
        # Case 2: Only one transport mode is allowed, mutate cities ONLY
        # Swap two cities, keeping transport modes the same  --> Invert 2 cities in the path
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = [individual[idx2][0], individual[idx1][1]], [individual[idx1][0], individual[idx2][1]]
    
    return individual,


"""
Heuristic Function 

Same as the slides
But consider mean of transporattion with the least cost/time
Considers the minimization of both cost and time
"""
def heuristic(Ncities,timebus,timetrain,timeplane,costbus,costtrain,costplane,mode,MOO=False):
    df=pd.read_csv("xy.csv").iloc[0:Ncities]
    
    # Step 1: Determine the median longitude to split the cities into left and right halves
    median_longitude = df['Longitude'].median()
    # Step 2: Split cities into left and right halves based on their longitude
    left_half = df[df['Longitude'] <= median_longitude].copy()  # Left half (Longitude <= median)
    right_half = df[df['Longitude'] > median_longitude].copy()  # Right half (Longitude > median)
    # Step 3: Sort cities in the left half by higher Latitude and larger Longitude if Latitude is the same
    left_half_sorted = left_half.sort_values(by=['Latitude', 'Longitude'], ascending=[False, False])
    # Step 4: Sort cities in the right half by smaller Latitude and smaller Longitude if Latitude is the same
    right_half_sorted = right_half.sort_values(by=['Latitude', 'Longitude'], ascending=[True, True])
    # Step 5: Combine the sorted cities, first from the left half, then from the right half
    sorted_cities = pd.concat([left_half_sorted, right_half_sorted])
    # Extract the sequence of cities
    city_sequence = sorted_cities['City'].tolist()
    
    #now consider, depending on the mode , if transport should have min cost or time
    
    heuristic_path=[]
    for i in range(len(city_sequence)-1):
        if(mode=="time"):
            transport_modes_time = {
            "bus": timebus.loc[city_sequence[i], city_sequence[i+1]],
            "train": timetrain.loc[city_sequence[i], city_sequence[i+1]],
            "plane": timeplane.loc[city_sequence[i], city_sequence[i+1]]
            }
            transport=min(transport_modes_time,key=transport_modes_time.get)
            heuristic_path.append([city_sequence[i],transport])
        elif(mode=="cost"):
            transport_modes_cost= {
            "bus": costbus.loc[city_sequence[i], city_sequence[i+1]],
            "train": costtrain.loc[city_sequence[i], city_sequence[i+1]],
            "plane": costplane.loc[city_sequence[i], city_sequence[i+1]]
            }
            transport=min(transport_modes_cost,key=transport_modes_cost.get)
            heuristic_path.append([city_sequence[i],transport])
    
    if(mode=="time"):
        transport_modes_time= {
        "bus": timebus.loc[city_sequence[-1], city_sequence[0]],
        "train": timetrain.loc[city_sequence[-1], city_sequence[0]],
        "plane": timeplane.loc[city_sequence[-1], city_sequence[0]]
        }
        transport_last_to_first=min(transport_modes_time,key=transport_modes_time.get)
        heuristic_path.append([city_sequence[-1],transport_last_to_first])
    elif(mode=="cost"):
        transport_modes_cost={
        "bus": costbus.loc[city_sequence[-1], city_sequence[0]],
        "train": costtrain.loc[city_sequence[-1], city_sequence[0]],
        "plane": costplane.loc[city_sequence[-1], city_sequence[0]]
        }
        transport_last_to_first=min(transport_modes_cost,key=transport_modes_cost.get)
        heuristic_path.append([city_sequence[-1],transport_last_to_first])
    
    if(not MOO):
        return heuristic_path
    else:
        return city_sequence
#Function that uses an EA approach and returns the statistics
"""
Genetic Algorithm for the variant of the TSP for the SOO

    Genetic Operators:
    -Order Crossover
    -Mutation -> swaping cities in the path (1 transport) or change transport (all transports)
    -selection -> tournament selection of size 3
    
type: type of transportation (default="bus")
mode: "cost" or "time" 
transport : "bus","train" or "plane"

Individuals:List --> [city,type of trasnport] : ["Lisbon","bus"] -> starting point Lisbon, chooses bus as the mean of transportation to the next city
Population : list of lists -> [["Lisbon","bus"],["Sofia","train"],...]

"""
def EvolutionaryAlgorithm(mode,transport,timebus,timetrain,timeplane,costbus,costtrain,costplane,Ncities,use_heuristic):
   
    #read cities file
    cities_xy=pd.read_csv("xy.csv")["City"].iloc[0:Ncities]
    #list of cities
    cities=list(cities_xy)
    print(cities)
    #list fo possible transports
    transports=["bus","train","plane"]
    
    #Create the Fitness and Individual Classes
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimization problem
    creator.create("Individual", list, fitness=creator.FitnessMin)  # Individual=list of 2 strings [city,transport]
    toolbox = base.Toolbox()
    # Register individual and population
    toolbox.register("individual", functools.partial(random_individual, cities, transports))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    #define evaluation function based on mode and transport
    if(mode=="cost"):
        if(transport=="bus"):
            toolbox.register("evaluate",functools.partial(evaluate,df=costbus))
        elif(transport=="train"):
            toolbox.register("evaluate",functools.partial(evaluate,df=costtrain))
        elif(transport=="plane"):
            toolbox.register("evaluate",functools.partial(evaluate,df=costplane))
        elif(transport=="all"):
            toolbox.register("evaluate",functools.partial(evaluate_all,bus=costbus,train=costtrain,plane=costplane))
                
    elif(mode=="time"):
        if(transport=="bus"):
            toolbox.register("evaluate",functools.partial(evaluate,df=timebus))
        elif(transport=="train"):
            toolbox.register("evaluate",functools.partial(evaluate,df=timetrain))
        elif(transport=="plane"):
            toolbox.register("evaluate",functools.partial(evaluate,df=timeplane))
        elif(transport=="all"):
            toolbox.register("evaluate",functools.partial(evaluate_all,bus=timebus,train=timetrain,plane=timeplane))
            
    #define crossover,mutation and selection operators : Genetic Operators
    #crossover
    toolbox.register("mate",cxOrder)
        
    #mutation
    if(transport=="all"):
        toolbox.register("mutate",mutate_path_and_or_transport,all=True)
    else:
        toolbox.register("mutate",mutate_path_and_or_transport,all=False)
        #toolbox.register("mutate",inverse_mutation)
            
    #selection : tournament selection, size=2
    toolbox.register("select", tools.selTournament, tournsize=2)
    
    #operators for Multi-Objective problem

    #Proceed with evolution
    pop=toolbox.population(n=40) #population of 40 individuals 
    elite_size=1 # for elitism
    
    #Use heuristic?
    if(use_heuristic):
        heuristic_path=creator.Individual(heuristic(Ncities,timebus,timetrain,timeplane,costbus,costtrain,costplane,mode))
        pop[-1]=heuristic_path
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
       
    #Probability that 2 individulas are crossed, mutation probability  default=(0.5,0.2)
    CXPB, MUTPB = 0.7, 0.4
    
    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]
    
    # Variable to keep track of the number of generations
    gen = 0
    #keep track of the best scores in each generation
    best_scores={}
    # Begin the evolution
    while gen < 250:
        # A new generation
        gen = gen + 1

        
        elite_individuals = tools.selBest(pop, elite_size)  # Select top elite individuals
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop) - elite_size)  # Reduce size by elite_size
        
        # Select the next generation individuals, was being used before
        #offspring = toolbox.select(pop, len(pop)) 
        
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring)) # copy of the selected individuals,genetic operators will make changes-in-place
        
        # Apply crossover (mating) and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values # invalidate fitnesses of the generated individuals

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                
        # Evaluate the individuals with an invalid fitness --> offsprings
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        #replace old population with offsprings
        pop[:] = offspring + elite_individuals
        
        best_scores[gen]=min(fits) #minimization problem
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5


    #selects best path
    elite = tools.selBest(pop, 1)
    
    #print(elite) --> best_path
    #print(elite[0])
    
    return best_scores,elite[0]

def check_and_fix_transport_modes(path, costbus, costtrain, costplane, timebus, timetrain, timeplane, mode, current_score):
    all_transports = ['bus', 'train', 'plane']
    used_transports = {city[1] for city in path}  # Collect transports used in the path
    missing_transports = set(all_transports) - used_transports  # Find the missing transports
    
    if len(missing_transports) == 0:
        # If all transports are used, return the original path and score
        return path, current_score
    
    # Choose the relevant matrix based on the mode
    if mode == 'cost':
        data_transport = {
            'bus': costbus,
            'train': costtrain,
            'plane': costplane
        }
    elif mode == 'time':
        data_transport = {
            'bus': timebus,
            'train': timetrain,
            'plane': timeplane
        }

    # Identify the missing transports and fix the path by replacing the shortest connections
    for missing_transport in missing_transports:
        # Find the shortest connection in the path
        min_distance = float('inf')
        min_index = -1
        current_city = None
        next_city = None
        
        for i in range(len(path) - 1):
            current_city = path[i][0]
            next_city = path[i + 1][0]
            current_transport = path[i][1]
            
            # Get the distance based on current transport
            distance = data_transport[current_transport].loc[current_city, next_city]
            if distance < min_distance:
                min_distance = distance
                min_index = i
                
        
        # Replace the shortest connection with the missing transport
        if min_index != -1:
            # Calculate the cost of substitution
            current_score -= min_distance  # Subtract the cost of the old transport
            path[min_index][1] = missing_transport  # Change the transport to the missing one
            #print('added', missing_transport)
            current_score += data_transport[missing_transport].loc[current_city, next_city]  # Add the cost of the new transport

    return path, current_score

"""
Function for multiobjective TSP problem

"""
def MOO(timebus,timetrain,timeplane,costbus,costtrain,costplane,Ncities,use_heuristic):
    
    #read cities file
    cities_xy=pd.read_csv("xy.csv")["City"].iloc[0:Ncities]
    #list of cities
    cities=list(cities_xy)
    #list fo possible transports
    transports=["bus","train","plane"]
    add=1 # for hypervolume without heuristic
     
    #Create the Fitness and Individual Classes
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))  # Minimization problem
    creator.create("Individual", list, fitness=creator.FitnessMin)  # Individual=list of 2 strings [city,transport]
    toolbox = base.Toolbox()
    # Register individual and population
    toolbox.register("individual", functools.partial(random_individual, cities, transports))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    #genetic operators
    toolbox.register("mate",cxOrder)
    toolbox.register("mutate",mutate_path_and_or_transport,all=True)
    toolbox.register("evaluate",evaluate_all_moo,timebus=timebus,timetrain=timetrain,timeplane=timeplane,
                     costbus=costbus,costtrain=costtrain,costplane=costplane)
    toolbox.register("select", tools.selNSGA2)
    
    
    #for hypervolume evolution
    hypervolume_evolution=[]

    #Proceed with evolution
    pop=toolbox.population(n=100) #population of 40 individuals 
    
    #heuristic?
    if(use_heuristic):
        heuristic_path=heuristic(Ncities,timebus,timetrain,timeplane,costbus,costtrain,costplane,"time",True)
        heuristic_solution=[[heuristic_path[i],pop[-1][i][1]] for i in range(len(pop[-1]))]
        #print(heuristic_solution)
        pop[-1]=creator.Individual(heuristic_solution)
        add=10
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
       
    #Probability that 2 individulas are crossed, mutation probability  default=(0.5,0.2)
    CXPB, MUTPB = 0.6, 0.3
    
    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]
    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))
    
    # Variable to keep track of the number of generations
    gen = 0
    while(gen <100):
        gen = gen + 1
        print("-- Generation %i --" % gen)
        
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover (mating) and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values # invalidate fitnesses of the generated individuals

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                
        # Evaluate the individuals with an invalid fitness --> offsprings
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        #replace old population with offsprings
        # Select the next generation individuals
        pop = toolbox.select(pop+offspring, len(pop)) 
        
        # Calculate Pareto front and hypervolume for this generation
        pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        # Extract the cost and time objectives for Pareto front
        pareto_times = [ind.fitness.values[1] for ind in pareto_front]
        pareto_costs = [ind.fitness.values[0] for ind in pareto_front]

        # Hypervolume calculation
        ref_point = [max(pareto_costs) + add, max(pareto_times) + add]  # Adjust this (far from optimal pareto front)
        hv = hypervolume(pareto_front)
        hypervolume_evolution.append(hv)

        pareto_front_paths = [ind for ind in pareto_front]  # Store individuals directly for access to paths
        
    
    min_cost_index = pareto_costs.index(min(pareto_costs))
    min_time_index = pareto_times.index(min(pareto_times))

    # Paths corresponding to minimum cost and minimum time
    min_cost_path = pareto_front_paths[min_cost_index]
    min_time_path = pareto_front_paths[min_time_index]

    TSPmap(min_time_path,True)
    TSPmap(min_cost_path,True)

    pareto=[pareto_costs,pareto_times]
    print(f'POF values (cost and time):\n')
    print(pareto)   #In this print we see the cost/time pairs for POF and retrieve them.
    
    return pareto,hypervolume_evolution

#define here what to do
def main():
    
    #Create the argument parser
    parser = argparse.ArgumentParser(description="Parse type of problem (cost,time),transportation (bus,train,plane,all),MOO for multi-objective problem")
    
    #Define commandline arguments
    #TO DO : add multi-objective problem
    parser.add_argument('--mode', choices=['cost', 'time'], required=False, help="Mode of evaluation: 'cost' or 'time'.",default="time")
    parser.add_argument('--transport', choices=['bus','train','plane','all'], required=False, help="Transportation : bus,plane,train or all",default="bus")
    parser.add_argument("--heuristic",type=bool,required=False,default=False,help="Heuristic? yes-True , no-False")
    parser.add_argument("--MOO",type=bool,required=False,default=False,help="Multi-objective problem")
    parser.add_argument("--Ncities",type=int,required=False,default=30,help="Select the number of cities")
    
    args = parser.parse_args()
    
    #read all data files
    costbus=pd.read_csv("costbus.csv",index_col="City")
    costtrain=pd.read_csv("costtrain.csv",index_col="City")
    costplane=pd.read_csv("costplane.csv",index_col="City")
    timebus=pd.read_csv("timebus.csv",index_col="City")
    timetrain=pd.read_csv("timetrain.csv",index_col="City")
    timeplane=pd.read_csv("timeplane.csv",index_col="City")
    
    
    best_scores=[] # to retain best score for each run
    all_paths=[] # keeps track of all paths
    convergence_curves=[]
    stored_convergence_curves = {}  # To store specific convergence curves for 8 test cases
    
    if(not args.MOO):
        #call EA for a given number of generations, obtain convergence curve and the best path
        for runs in range(1):
            
            scores,best_path=EvolutionaryAlgorithm(args.mode,args.transport,timebus,timetrain,timeplane,costbus,costtrain,costplane,args.Ncities,args.heuristic)
            current_best_score = list(scores.values())[-1]
            #retain list of best_scores in all 30 runs
            best_scores.append(current_best_score)
            #retain list of all best_paths from each run
            all_paths.append(best_path)
            convergence_curves.append(scores)
            
        best_solution_index=best_scores.index(min(best_scores)) # get index of the best solution
        #obtain best path from all runs
        best_path=all_paths[best_solution_index]
        #obtain convergence curve
        convergence_curve=convergence_curves[best_solution_index]
        ##std=statistics.stdev(best_scores)
        #print(f"Mean:{mean} , sdt_dev :{std}")  

        if args.transport == 'all':
           
            for runs in range(1):
                scores,best_path=EvolutionaryAlgorithm(args.mode,args.transport,timebus,timetrain,timeplane,costbus,costtrain,costplane,args.Ncities,args.heuristic)
                current_best_score = list(scores.values())[-1]
                #retain list of best_scores in all 30 runs
                best_scores.append(current_best_score)
                #retain list of all best_paths from each run
                all_paths.append(best_path)
                convergence_curves.append(scores)
            
            best_solution_index=best_scores.index(min(best_scores)) # get index of the best solution
            #obtain best path from all runs
            best_path=all_paths[best_solution_index]
            #atualiza o best_path e custo se faltar transporte
            best_path, newscore = check_and_fix_transport_modes(best_path, costbus, costtrain, costplane, timebus, timetrain, timeplane, args.mode, current_best_score)
            #obtain convergence curve
            convergence_curve=convergence_curves[best_solution_index]
            #mean=np.array(best_scores).mean()
            #std=statistics.stdev(best_scores)
            #print(f"Mean:{mean} , sdt_dev :{std}")

            TSPmap(best_path,True)
        else:
            TSPmap(best_path)
       
        sns.set(style="whitegrid") 
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=convergence_curve.keys(), y=convergence_curve.values(), marker='o', color='blue', label="Best Score")
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Best Fitness Score', fontsize=12)
        plt.title(f'Evolution of Best Fitness Score for mode={args.mode},transport={args.transport},heuristic={args.heuristic}', fontsize=14)
        plt.legend()
        plt.show()
    else:
        pareto,hypervolume_evolution=MOO(timebus,timetrain,timeplane,costbus,costtrain,costplane,args.Ncities,args.heuristic)
        # Final Pareto curve plot
        sns.set(style="whitegrid")
        plt.figure(figsize=(10,6))
        sns.lineplot(x=pareto[0],y=pareto[1],marker="o",color="blue",label="Pareto Curve")
        plt.title("Pareto Curve: Cost vs Time")
        plt.xlabel("Cost")
        plt.ylabel("Time")
        plt.grid(True)
        plt.show()

        # Plot Hypervolume Evolution
        sns.set(style="whitegrid")
        plt.figure(figsize=(10,6))
        plt.plot(range(len(hypervolume_evolution)), hypervolume_evolution)
        plt.title("Hypervolume Evolution Across Generations")
        plt.xlabel("Generation")
        plt.ylabel("Hypervolume")
        plt.grid(True)
        plt.show()
    

    return



if __name__ == "__main__":
    main()
    
    