from math import exp, cos, sqrt, e, pi
import random as prng
import numpy as np
from matplotlib.ticker import LinearLocator
from platypus import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from platypus.core import _EvaluateJob
import itertoograyls

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


class GeneticAlgorithmFitness(GeneticAlgorithm):
    def __init__(self, problem,
                 population_size=100,
                 offspring_size=100,
                 generator=RandomGenerator(),
                 selector=TournamentSelector(2),
                 comparator=ParetoDominance(),
                 variator=None,
                 psim=0.6,
                 **kwargs):
        super().__init__(problem, population_size, generator, **kwargs)
        self.offspring_size = offspring_size
        self.selector = selector
        self.comparator = comparator
        self.variator = variator
        self.psim = psim

    def initialize(self):
        super().initialize()

        if self.variator is None:
            self.variator = default_variator(self.problem)

        self.population = sorted(self.population, key=functools.cmp_to_key(self.comparator))
        self.fittest = self.population[0]

    def iterate(self):
        offspring = []
        # lambda is population size
        # copy two best individuals from population
        chosen_parents = self.selector.select(self.variator.arity, self.population)

        # create number_of_pa
        while len(offspring) < self.population_size - 2:
            parents = self.selector.select(self.variator.arity, self.population)
            p1, p2 = parents

            # return offspring of two individuals
            ind = self.variator.evolve(parents)

            # for every individuals in the offspring compute the average of every objectives
            for i in range(2):
                for j in range(len(p1.objectives)):
                    ind[i].objectives[j] = (p1.objectives[j] + p2.objectives[j]) / 2

                # mark individual as unevaluated and when it will be selected to be evaluated normally, his fitness will be recompute
                ind[i].evaluated = False
            offspring.extend(ind)  # extends the offspring
        offspring.extend(chosen_parents)  # add chosen parents

        # number of individual normally evaluated
        n_normal_evaluated = int(self.population_size * self.psim)

        # get index of individuals normally evaluated randomly
        normally_evaluated_indexes = prng.sample(range(len(offspring)), n_normal_evaluated)

        normally_evaluated = [offspring[i] for i in normally_evaluated_indexes]
        remain_indexes = list(set(range(len(offspring))).difference(set(normally_evaluated_indexes)))
        fi_evaluated = [offspring[i] for i in remain_indexes]
        self.evaluate_all(normally_evaluated)

        offspring.append(self.fittest)

        offspring = sorted(offspring, key=functools.cmp_to_key(self.comparator))

        self.population = offspring[:self.population_size]
        self.fittest = self.population[0]

    def evaluate_all(self, solutions):
        unevaluated = [s for s in solutions if not s.evaluated]
        print(f'len non evalue = {len(unevaluated)}')
        print(f'len evaluated = {len(solutions) - len(unevaluated)}')
        jobs = [_EvaluateJob(s) for s in unevaluated]
        results = self.evaluator.evaluate_all(jobs)

        # if needed, update the original solution with the results
        # only update element that changes after evaluation
        for i, result in enumerate(results):
            if unevaluated[i] != result.solution:
                unevaluated[i].variables[:] = result.solution.variables[:]
                unevaluated[i].objectives[:] = result.solution.objectives[:]
                unevaluated[i].constraints[:] = result.solution.constraints[:]
                unevaluated[i].constraint_violation = result.solution.constraint_violation
                unevaluated[i].feasible = result.solution.feasible
                unevaluated[i].evaluated = result.solution.evaluated

        # self.nfe += len(unevaluated)
        # using fitness inheritance
        self.nfe += len(solutions) * self.psim

    def run(self, condition, callback=None):
        if isinstance(condition, int):
            condition = MaxEvaluations(condition)

        if isinstance(condition, TerminationCondition):
            condition.initialize(self)

        last_log = self.nfe
        start_time = time.time()

        LOGGER.log(logging.INFO, "%s starting", type(self).__name__)

        while not condition(self):
            self.step()

            if self.log_frequency is not None and self.nfe >= last_log + self.log_frequency:
                LOGGER.log(logging.INFO,
                           "%s running; NFE Complete: %d, Elapsed Time: %s",
                           type(self).__name__,
                           self.nfe,
                           datetime.timedelta(seconds=time.time() - start_time))

            if callback is not None:
                callback(self)

        LOGGER.log(logging.INFO,
                   "%s finished; Total NFE: %d, Elapsed Time: %s",
                   type(self).__name__,
                   self.nfe,
                   datetime.timedelta(seconds=time.time() - start_time))


if __name__ == '__main__':
    # define a new Problem with 2 variables and 1 objective
    myProblem = Problem(2, 1, function=ackley)
    # each variable of the myProblem is a float value in [-5,5]
    myProblem.types[:] = Real(-5, 5)
    # set the optimization direction (Min by default or Max) for each objective
    myProblem.directions = [Problem.MINIMIZE]
    # choose optimization algorithm
    algorithm = GeneticAlgorithmFitness(myProblem)
    # for verification, comment the line above and uncomment the line below
    # algorithm = GeneticAlgorithm(myProblem, variator=GAOperator(SBX(), PM()))

    algorithm.observers = [printBestIndividual, saveStatistics]
    # execute algorithmprintBestIndividual
    algorithm.run(10000,
                  callback=Observers)  # 1. 4000 is the maximum number of the evaluation function is called inside the code
    # uncomment the line below to plot statistics
    plotStatistics(algorithm)

    # plot search space
    plotSearchSpace(algorithm)
