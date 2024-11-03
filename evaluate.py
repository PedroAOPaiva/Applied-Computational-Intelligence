""""
This file contains all evaluation functions used for the single objective problem

Time: bus,train,plane,all
Cost: bus,train,plane,all

Individual is given by a list containing a city and the transport
to the next city in the path -> ["Lisbon","bus"] : goes from Lisbon to the next city by bus

"""

import pandas as pd

#Define all evaluation fucntions to be used (for each case)


#for single transports, pass respective dataframe (depends on transport and mode)
def evaluate(individual,df):
    
    fitness=0
    
    for i in range(len(individual)-1):
        fitness=fitness+df.loc[individual[i][0],individual[i+1][0]]
        
    #add last->first
    fitness=fitness+df.loc[individual[-1][0],individual[0][0]]
    
    return fitness,


#for multiple_transports , pass respective dataframes (depends on mode)
def evaluate_all(individual,bus,train,plane):
    
    fitness=0
    
    for i in range(len(individual)-1):
        if(individual[i][1]=="bus"):
            fitness=fitness+bus.loc[individual[i][0],individual[i+1][0]]
        elif(individual[i][1]=="train"):
            fitness=fitness+train.loc[individual[i][0],individual[i+1][0]]   
        elif(individual[i][1]=="plane"):
            fitness=fitness+plane.loc[individual[i][0],individual[i+1][0]]
            
    if(individual[-1][1]=="bus"):
        fitness=fitness+bus.loc[individual[-1][0],individual[0][0]]
    elif(individual[-1][1]=="train"):
        fitness=fitness+train.loc[individual[-1][0],individual[0][0]]
    else:
        fitness=fitness+plane.loc[individual[-1][0],individual[0][0]]
    
    return fitness,



#evaluation function for multiobjective problem
#considers both cost ant time in a given solution/path
def evaluate_all_moo(individual,timebus,timetrain,timeplane,costbus,costtrain,costplane):
    
    #time
    fitness1=0
    #cost
    fitness2=0
    
    for i in range(len(individual)-1):
        if(individual[i][1]=="bus"):
            fitness1=fitness1+timebus.loc[individual[i][0],individual[i+1][0]]
            fitness2=fitness2+costbus.loc[individual[i][0],individual[i+1][0]]
        elif(individual[i][1]=="train"):
            fitness1=fitness1+timetrain.loc[individual[i][0],individual[i+1][0]]
            fitness2=fitness2+costtrain.loc[individual[i][0],individual[i+1][0]]   
        elif(individual[i][1]=="plane"):
            fitness1=fitness1+timeplane.loc[individual[i][0],individual[i+1][0]]
            fitness2=fitness2+costplane.loc[individual[i][0],individual[i+1][0]]
            
    if(individual[-1][1]=="bus"):
        fitness1=fitness1+timebus.loc[individual[-1][0],individual[0][0]]
        fitness2=fitness2+costbus.loc[individual[-1][0],individual[0][0]]
    elif(individual[-1][1]=="train"):
        fitness1=fitness1+timetrain.loc[individual[-1][0],individual[0][0]]
        fitness2=fitness2+costtrain.loc[individual[-1][0],individual[0][0]]
    else:
        fitness1=fitness1+timeplane.loc[individual[-1][0],individual[0][0]]
        fitness2=fitness2+costplane.loc[individual[-1][0],individual[0][0]]

    #cost,time
    return fitness2,fitness1
    
