'''
    Modelo de computação evolutiva para resolver o problema das 8 rainhas

    gerar populacao inicial
    loop ate achar solucao otima:
        rankear com base no fit
        escolher pares para crossover
        gerar filhos
        mutacoes
    
'''

'''
    Evolutionary Algorithm model to solve the 8 Queens problem

    Steps:
        1 - Generate Initial Population
        2 - Loop until achieve optimal solution:
            2.1 - get fitness of each individual
            2.2 - choose pair os parents for crossover (tournament)
            2.3 - reproduce and make children (1 point and 2 point crossover)
            2.4 - generate mutation on children (replace and swap)
            2.5 - update population
        3 - get optimal solution or not 
'''


import numpy as np

queens = np.array([1, 2, 3, 4, 5, 6, 7, 8])
tabletest = np.array([[1,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,8],
                      [0,2,0,0,0,0,0,0],
                      [0,0,0,0,0,0,7,0],
                      [0,0,3,0,0,0,0,0],
                      [0,0,0,0,0,6,0,0],
                      [0,0,0,4,0,0,0,0],
                      [0,0,0,0,5,0,0,0]]) #[1,3,5,7,8,6,4,2]
# positionstest = np.array([1,3,5,7,8,6,4,2])                      
# positionstest = np.array([4,2,5,8,5,1,3,6])
positionstest = np.array([7,3,8,2,5,1,6,4])
table = np.zeros((8,8))



def computeQueensCost(positions):
    positions = positions -1
    cost = 0
    table_aux = np.zeros((8,8))
    for i in range(0,8):
        table_aux[positions[i]][i] = 1
    # print(table_aux)
    cost_offset = 7
    for i in range(0,8):
        j = positions[i]
        d = i - j
        d_flip = 7 - (i + j)
        table_aux[j][i] = 0
        cost += cost_offset - (np.sum(table_aux[j,:])+np.sum(table_aux[:,i])+np.sum(np.diagonal(table_aux,d))+np.sum(np.fliplr(table_aux).diagonal(d_flip)))
        # print((np.sum(table_aux[j,:]), np.sum(table_aux[:,i]), np.sum(np.diagonal(table_aux,d)), np.sum(np.fliplr(table_aux).diagonal(d_flip))))
        cost_offset -= 1
        

    return cost

def tournament(population):
    indexes = np.random.choice(range(0,population.shape[0]),4,False)
    return population[indexes[np.argmax(getFitness(population[indexes]))]]

def getParent(population):
    return tournament(population)

def getFitness(population):
    return np.array([computeQueensCost(population[i]) for i in range(0,population.shape[0])])

def mutation(children, n):
    indexes = np.random.choice(range(0,children.shape[0]),n,False)
    # print("idxs: ", indexes)
    mode = 0
    for i in indexes:
        if mode % 2 < 1: #replace
            j = np.random.choice(8,1)
            val = np.random.choice(range(1,9),1)
            # print(i, "mut-replace: ",j,val)
            children[i][j] = val
        else: #swap
            j = np.random.choice(8,2,replace=False)
            # print(i, "mut-swap: ",j)
            val = children[i][j[0]]
            children[i][j[0]] = children[i][j[1]]
            children[i][j[1]] = val
        mode += 1
        
    return children

def findQueensSolution():
    population = np.random.randint(1,9, size=(32, 8))
    fitness = getFitness(population)
    count = 0
    print(count, fitness)
    while np.sum(fitness[fitness >= 28]) < 1 and count < 200:
        count += 1
        children = np.zeros((population.shape[0],population.shape[1]),dtype=int)
        mode = 0
        for i in range(0,int(population.shape[0]/2)):
            father = getParent(population)
            mother = getParent(population)
            while(np.array_equal(father, mother)):
                father = getParent(population)
            if mode % 2 < 1:
                children[2*i] = np.concatenate((father[:5], mother[5:]), axis=None)
                children[(2*i)+1] = np.concatenate((mother[:5], father[5:]), axis=None)
            else: 
                children[2*i] = np.concatenate((father[:3], mother[3:5], father[5:]), axis=None)
                children[(2*i)+1] = np.concatenate((mother[:3], father[3:5], mother[5:]), axis=None)
            mode += 1
            

        children = mutation(children,int(children.shape[0]*0.75))#mutation

        #elitism
        fittest_parent = population[np.argmax(getFitness(population))]
        children[np.argmin(getFitness(children))] = fittest_parent

        population = children.copy()
        print(children)

        fitness = getFitness(population)
        print(count, fitness)

    return population[np.argmax(getFitness(population))]
    

# print(computeQueensCost(positionstest))
solution = findQueensSolution()
print(solution)