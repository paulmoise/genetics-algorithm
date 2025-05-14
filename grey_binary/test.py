from platypus import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from Gray_binary.comp_bin_gray import Gray, BitFlipUpdated, HUXUpdated

# This simple example has an optimal value of 15 when picking items 1 and 4.
# items = 7
# capacity = 9
# weights = [2, 3, 6, 7, 5, 9, 4]
# profits = [6, 5, 8, 9, 6, 7, 3]

items = 15
capacity = 750
weights = [70, 73, 77, 80, 82, 87, 90, 94, 98, 106, 110, 113, 115, 118, 120]
profits = [135, 139, 149, 150, 156, 163, 173, 184, 192, 201, 210, 214, 221, 229, 240]
# sols = [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]


def knapsack(x):
    selection = x[0]
    total_weight = sum([weights[i] if selection[i] else 0 for i in range(items)])
    total_profit = sum([profits[i] if selection[i] else 0 for i in range(items)])

    return total_profit, total_weight

def printBestIndividual(self):
    """ Function called during each iteration of the algorithm"""
    # select one individual in population regarding its objective value as minimized 'larger=False'
    solutions = [solution for solution in unique(nondominated(self.result))]
    bestSolution = nondominated_truncate(solutions, 1)

    print(solutions)
    # for solution in unique(nondominated(self.result)):
    #     print(solution.variables, solution.objectives)

    # print(bestSolution.objectives)
    # print('{0} {1} {2}'.format(self.nfe, bestSolution.variables[0],
    #



def saveStatistics(self):
    """ Observer function for saving population statistics
    called during each iteration of the algorithm"""
    # check whether self has attribute 'statistics' ?
    if not hasattr(self, 'statistics'):
        self.statistics = {'nfe': [], 'avg': [], 'min': [], 'max': [], 'std': []}
    self.statistics['nfe'].append(self.nfe)
    fitness = [x.objectives[0] for x in self.population]
    self.statistics['avg'].append(np.average(fitness))
    self.statistics['min'].append(np.min(fitness))
    self.statistics['max'].append(np.max(fitness))
    self.statistics['std'].append(np.std(fitness))


def Observers(self):
    """ Defines a set of functions to be called"""
    if hasattr(self, 'observers'):
        for obs in self.observers:
            obs(self)
    else:
        raise NameError("Unknown attribute 'observers'. No method to call.")

nexec = 30
nfe = 15*1000
fig = plt.figure(figsize=(15, 10))  # a new figure
# for all combinations of Xover and Mutation
for type in ['BINARY', 'GRAY']:

    resultNfe, resultMax = [], [],  # empty results
    for seed in range(nexec):  # execute same algorithm several times
        random.seed(seed)  # modify current seed

        myProblem = Problem(1, 1, 1)
        myProblem.function = knapsack
        myProblem.types[0] = Binary(items) if type == 'BINARY' else Gray(items)
        myProblem.directions[0] = Problem.MAXIMIZE
        myProblem.constraints[0] = Constraint("<=", capacity)

        algorithm = GeneticAlgorithm(myProblem, variator=GAOperator(HUXUpdated(), BitFlipUpdated()))
        algorithm.observers = [printBestIndividual, saveStatistics]
        algorithm.run(nfe, callback=Observers)
        # create a pandas serie for Nfe & Min fitness
        resultNfe.append(pd.Series(algorithm.statistics['nfe']))
        resultMax.append(pd.Series(algorithm.statistics['max']))

        # execution may be long, so print where we are
        print('run {0} with {1}'.format(seed, type))
        # mean of all executions for same algorithm using pandas
        X = pd.concat(resultNfe, axis=1).mean(axis=1).tolist()
        Y = pd.concat(resultMax, axis=1).mean(axis=1).tolist()
    # TO BE COMPLETED
    plt.plot(X, Y, label=f'{type}')
plt.title(
    'Genetic Algorithm on KnapSack Problem on Binary and Gray encoding(' + str(items) + 'nbits) on ' + str(
        nexec) + ' executions')
plt.xlabel('Number of Function Evaluations')
plt.ylabel('Average max fitness')
plt.legend()
plt.show(block=True)


problem = Problem(1, 1, 1)
problem.types[0] = Binary(items)
problem.directions[0] = Problem.MAXIMIZE
problem.constraints[0] = Constraint("<=", capacity)
problem.function = knapsack

algorithm = GeneticAlgorithm(problem)
algorithm.run(10000)
print(algorithm.result)
for solution in unique(nondominated(algorithm.result)):
    print(solution.variables, solution.objectives)



def gray_vs_bin_max_best_fitness(nfe=15 * 1000, nexec=3):
    knap = KnapSack()
    knap.size = 15
    knap.capacity = 750
    knap.weights = [70, 73, 77, 80, 82, 87, 90, 94, 98, 106, 110, 113, 115, 118, 120]
    knap.profits = [135, 139, 149, 150, 156, 163, 173, 184, 192, 201, 210, 214, 221, 229, 240]
    knap.sols = [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]

    # s = 1458
    def knapsack_func(x):
        selection = x[0]
        total_weight = sum([knap.weights[i] if selection[i] else 0 for i in range(knap.size)])
        total_profit = sum([knap.profits[i] if selection[i] else 0 for i in range(knap.size)])
        return total_profit, total_weight

    fig = plt.figure(figsize=(15, 10))  # a new figure
    # for all combinations of Xover and Mutation
    for type in ['BINARY', 'GRAY']:

        resultNfe, resultMax = [], [],  # empty results
        for seed in range(nexec):  # execute same algorithm several times
            random.seed(seed)  # modify current seed

            myProblem = Problem(1, 1, 1)
            myProblem.function = knapsack_func
            myProblem.types[0] = Binary(knap.size) if type == 'BINARY' else Gray(knap.size)
            myProblem.directions[0] = Problem.MAXIMIZE
            myProblem.constraints[0] = Constraint("<=", knap.size)

            algorithm = GeneticAlgorithm(myProblem, variator=GAOperator(HUXUpdated(), BitFlipUpdated()))
            algorithm.observers = [printBestIndividual, saveStatistics]
            algorithm.run(nfe, callback=Observers)
            # create a pandas serie for Nfe & Min fitness
            resultNfe.append(pd.Series(algorithm.statistics['nfe']))
            resultMax.append(pd.Series(algorithm.statistics['max']))

            # execution may be long, so print where we are
            print('run {0} with {1}'.format(seed, type))
            # mean of all executions for same algorithm using pandas
            X = pd.concat(resultNfe, axis=1).mean(axis=1).tolist()
            Y = pd.concat(resultMax, axis=1).mean(axis=1).tolist()
        # TO BE COMPLETED
        plt.plot(X, Y, label=f'{type}')
    plt.title(
        'Genetic Algorithm on KnapSack Problem on Binary and Gray encoding(' + str(knap.size) + 'nbits) on ' + str(
            nexec) + ' executions')
    plt.xlabel('Number of Function Evaluations')
    plt.ylabel('Average max fitness')
    plt.legend()
    plt.show(block=True)