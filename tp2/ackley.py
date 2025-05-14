from math import exp, cos, sqrt, e, pi
import numpy as np
from matplotlib.ticker import LinearLocator
from platypus import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

"""
      SMPSO:	Speed-Constrained Multiobjective Particle Swarm Optimization
      MOEA/D:	Multiobjective Evolutionary Algorithm with Decomposition
      NSGA-II: Non-dominated Sorting Genetic Algorithm II
      NSGA-III	Reference-Point Based Non-dominated Sorting Genetic Algorithm
"""
'''Akcley function
   https://en.wikipedia.org/wiki/Ackley_function
'''


def ackley(individual):
    a = 20
    b = 0.2
    c = 2 * pi
    fitness = -a * exp(-b * sqrt(1.0 / len(individual)) * sum([xi ** 2 for xi in individual])) - \
              exp(1.0 / len(individual) * sum([cos(c * xi) for xi in individual])) + a + e
    return fitness


def printBestIndividual(self):
    """ Function called during each iteration of the algorithm"""
    # select one individual in population regarding its objective value as minimized 'larger=False'
    bestSolution = truncate_fitness(self.population, 1,
                                    larger_preferred=self.problem.directions[0] == self.problem.MAXIMIZE,
                                    getter=objective_key)[0]
    print(bestSolution.objectives)
    print('{0} {1:4f} {2:4f} {3:7f}'.format(self.nfe, bestSolution.variables[0],
                                            bestSolution.variables[1],
                                            bestSolution.objectives[0]))


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


def plotStatistics(self):
    if hasattr(self, 'statistics'):
        fig = plt.figure(0)
        plt.plot(self.statistics['nfe'], self.statistics['avg'], label='avg')
        plt.plot(self.statistics['nfe'], self.statistics['min'], label='min')
        plt.plot(self.statistics['nfe'], self.statistics['max'], label='max')
        plt.xlabel('nfe')
        plt.ylabel('fitness')
        plt.legend()
        # Should we wait action from user ? => block=True
        plt.show(block=True)
    else:
        raise NameError(
            "Unknown attribute 'statistics for plotting statistics. Method 'saveStatistics should be used as observer.")


def plotSearchSpace(self):
    """ Plot search space and population"""
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')  # create a 3D plot
    ax.view_init(30, 40)  # change 3D view
    x = np.arange(-5, 5, 0.1)  # set of float values between
    y = np.arange(-5, 5, 0.1)  # -0.5 and 0.5 step 0.1
    X, Y = np.meshgrid(x, y)  # dot product between x & y
    Z = [ackley([a, b]) for a, b in zip(np.ravel(X), np.ravel(Y))]
    Z = np.array(Z).reshape(X.shape)
    surf = ax.plot_surface(X, Y, Z, alpha=0.3)
    ...  # TO BE COMPLETED (plot solutions in red)
    # surf = ax.zaxis.set_major_locator(LinearLocator(10))
    solX = [s.variables[0] for s in self.population]
    solY = [s.variables[1] for s in self.population]
    solZ = [s.objectives[0] for s in self.population]
    surf = ax.scatter(solX, solY, solZ, color='red')
    plt.show()


# if __name__ == '__main__':
#     # define a new Problem with 2 variables and 1 objective
#     myProblem = Problem(2, 1, function=ackley)
#     # each variable of the myProblem is a float value in [-5,5]
#     myProblem.types[:] = Real(-5, 5)
#     # set the optimization direction (Min by default or Max) for each objective
#     myProblem.directions = [Problem.MINIMIZE]
#     # choose optimization algorithm
#     algorithm = GeneticAlgorithm(myProblem)
#     # for verification, comment the line above and uncomment the line below
#     # algorithm = GeneticAlgorithm(myProblem, variator=GAOperator(SBX(), PM()))
#
#     algorithm.observers = [printBestIndividual, saveStatistics]
#     # execute algorithmprintBestIndividual
#     algorithm.run(10000,
#                   callback=Observers)  # 1. 4000 is the maximum number of the evaluation function is called inside the code
#     # uncomment the line below to plot statistics
#     plotStatistics(algorithm)
#
#     # plot search space
#     plotSearchSpace(algorithm)

"""
7-What is interesting, now, is to launch as many
executions as there are possible combinations to then deduce which one is the most suitable for this optimization
problem.
"""


def find_suitable_mut_xover(nfe=1000, nexec=3, ackley_variable=2):
    # Xovers = [SBX(), DifferentialEvolution(), PCX(), UNDX(), SPX(), HUX(), PMX(), SSX()]  # all possible Xovers
    Xovers = [SBX(), PCX(), UNDX(), SPX()]  # all possible Xovers
    # Mutations = [PM(), CompoundMutation(), Replace(), Swap(), BitFlip(), UM(), UniformMutation(),
    Mutations = [PM(), UM(), UniformMutation(probability=0.01, perturbation=0.5)]  # all possible Mutations
    fig = plt.figure(figsize=(15, 10))  # a new figure
    # for all combinations of Xover and Mutation
    for Xover, Mutation in [(x, m) for x in Xovers for m in Mutations]:
        resultNfe, resultMin = [], [],  # empty results
        XoverName, MutName = type(Xover).__name__, type(Mutation).__name__
        for seed in range(nexec):  # execute same algorithm several times
            random.seed(seed)  # modify current seed
            # TO BE COMPLETED
            myProblem = Problem(ackley_variable, 1, function=ackley)
            # each variable of the myProblem is a float value in [-5,5]
            myProblem.types[:] = Real(-5, 5)
            # set the optimization direction (Min by default or Max) for each objective
            myProblem.directions = [Problem.MINIMIZE]
            algorithm = GeneticAlgorithm(myProblem, variator=GAOperator(Xover, Mutation))
            algorithm.observers = [printBestIndividual, saveStatistics]
            algorithm.run(nfe, callback=Observers)
            # create a pandas serie for Nfe & Min fitness
            resultNfe.append(pd.Series(algorithm.statistics['nfe']))
            resultMin.append(pd.Series(algorithm.statistics['min']))

            # execution may be long, so print where we are
            print('run {0} with {1} {2}'.format(seed, XoverName, MutName))
            # mean of all executions for same algorithm using pandas
            X = pd.concat(resultNfe, axis=1).mean(axis=1).tolist()
            Y = pd.concat(resultMin, axis=1).mean(axis=1).tolist()
        # TO BE COMPLETED
        plt.plot(X, Y, label=f'{XoverName} {MutName}')
    plt.title('Genetic Algorithm XOver & Mutation comparisons on Ackley(' + str(myProblem.nvars) + 'D) on ' + str(
        nexec) + ' executions')
    plt.xlabel('Number of Function Evaluations')
    plt.ylabel('Average min fitness')
    plt.legend()
    plt.show(block=True)

if __name__ == '__main__':
    find_suitable_mut_xover()
    # find_suitable_mut_xover(nfe=10000, nexec=30, ackley_variable=2)
    # find_suitable_mut_xover(nfe=10000, nexec=30, ackley_variable=10)
