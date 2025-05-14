from copy import deepcopy

from platypus import *
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def printBestIndividual(self):
    """ Function called during each iteration of the algorithm"""
    # select one individual in population regarding its objective value as minimized 'larger=False'
    bestSolution = [solution for solution in unique(nondominated(self.result))][0]
    # print('{0} {1} {2}'.format(self.nfe, bestSolution.variables[0],
    #                            bestSolution.objectives[0]))


def saveStatistics(self):
    """ Observer function for saving population statistics
    called during each iteration of the algorithm"""
    # check whether self has attribute 'statistics' ?
    if not hasattr(self, 'statistics'):
        self.statistics = {'nfe': [], 'avg': [], 'min': [], 'max': [], 'std': []}
    self.statistics['nfe'].append(self.nfe)
    fitness = [x.objectives[0] for x in nondominated(self.result)]
    print(fitness)
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


class bestFitness(Indicator):
    """find best fitness in population"""

    def __init__(self):
        super(bestFitness, self).__init__()

    def calculate(self, set):
        feasible = [s for s in set if s.constraint_violation == 0.0]
        if len(feasible) == 0:
            return 0.0
        elif feasible[0].problem.nobjs != 1:
            raise ValueError("bestFitness indicator can only be used for single-objective problems")
        best = None
        optimum = np.min if feasible[0].problem.directions[0] == Problem.MINIMIZE else np.max
        best = optimum([x.objectives[0] for x in feasible])
        return best


class HUXUpdated(Variator):
    def __init__(self, probability=1.0):
        super().__init__(2)
        self.probability = probability

    def evolve(self, parents):
        result1 = copy.deepcopy(parents[0])
        result2 = copy.deepcopy(parents[1])
        problem = result1.problem

        if random.uniform(0.0, 1.0) <= self.probability:
            for i in range(problem.nvars):
                if isinstance(problem.types[i], Binary) or isinstance(problem.types[i], Gray):
                    for j in range(problem.types[i].nbits):
                        if result1.variables[i][j] != result2.variables[i][j]:
                            if bool(random.getrandbits(1)):
                                result1.variables[i][j] = not result1.variables[i][j]
                                result2.variables[i][j] = not result2.variables[i][j]
                                result1.evaluated = False
                                result2.evaluated = False

        return [result1, result2]


class BitFlipUpdated(Mutation):

    def __init__(self, probability=1):
        """Bit Flip Mutation for Binary Strings.

        Parameters
        ----------
        probability : int or float
            The probability of flipping an individual bit.  If the value is
            an int, then the probability is divided by the number of bits.
        """
        super().__init__()
        self.probability = probability

    def mutate(self, parent):
        result = copy.deepcopy(parent)
        problem = result.problem
        probability = self.probability

        if isinstance(probability, int):
            probability /= sum([t.nbits for t in problem.types if isinstance(t, Binary) or isinstance(t, Gray)])

        for i in range(problem.nvars):
            type = problem.types[i]

            if isinstance(type, Binary) or isinstance(type, Gray):
                for j in range(type.nbits):
                    if random.uniform(0.0, 1.0) <= probability:
                        result.variables[i][j] = not result.variables[i][j]
                        result.evaluated = False

        return result


class Gray(Type):
    def __init__(self, nbits):
        super().__init__()
        self.nbits = nbits

    def rand(self):
        return [random.choice([False, True]) for _ in range(self.nbits)]

    def encode(self, value):
        return bin2gray(value)

    def __str__(self):
        return "Gray(%d)" % self.nbits


class KnapSack:
    def __int__(self, size, weights, profits, capacity, sols):
        self.size = size
        self.weights = weights
        self.profits = profits
        self.capacity = capacity
        self.sols = sols


def gray_vs_binary(knap):
    def knapsack_func(x):
        selection = x[0]
        total_weight = sum([knap.weights[i] if selection[i] else 0 for i in range(knap.size)])
        total_profit = sum([knap.profits[i] if selection[i] else 0 for i in range(knap.size)])
        return total_profit, total_weight

    nexec = 30
    problem1 = Problem(1, 1, 1)
    problem2 = Problem(1, 1, 1)

    problem1.directions[0] = Problem.MAXIMIZE
    problem2.directions[0] = Problem.MAXIMIZE

    problem1.constraints[0] = Constraint("<=", knap.capacity)
    problem2.constraints[0] = Constraint("<=", knap.capacity)

    problem1.function = knapsack_func
    problem2.function = knapsack_func

    problem1.types[0] = Binary(knap.size)
    problem2.types[0] = Gray(knap.size)

    problems = [(problem1, 'Binary'), (problem2, 'Gray')]

    algorithms = [(GeneticAlgorithm, dict(variator=GAOperator(HUXUpdated(), BitFlipUpdated())), 'GA')]
    nfe = knap.size * 1000
    results = experiment(algorithms=algorithms, problems=problems, nfe=nfe, seeds=nexec, display_stats=True)
    print(results)

    indicators = [bestFitness()]
    indicators_result = calculate(results, indicators)
    display(indicators_result, ndigits=3)

    data = dict()
    for key_algorithm, algorithm in indicators_result.items():
        print(key_algorithm, algorithm)
        for key_problem, problem in algorithm.items():
            data[(key_algorithm, key_problem)] = indicators_result[key_algorithm][key_problem]['bestFitness']
    df_bestFitness = pd.DataFrame(data=data)
    df_bestFitness.to_csv(f'experiment_{nexec}_runs_nbits_{knap.size}.csv')


def comparison():
    knapsack1 = KnapSack()
    knapsack1.size = 15
    knapsack1.capacity = 750
    knapsack1.weights = [70, 73, 77, 80, 82, 87, 90, 94, 98, 106, 110, 113, 115, 118, 120]
    knapsack1.profits = [135, 139, 149, 150, 156, 163, 173, 184, 192, 201, 210, 214, 221, 229, 240]
    knapsack1.sols = [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]

    knapsack3 = KnapSack()
    knapsack3.size = 100
    knapsack3.profits = [
        360, 83, 59, 130, 431, 67, 230, 52, 93, 125, 670, 892, 600, 38, 48, 147,
        78, 256, 63, 17, 120, 164, 432, 35, 92, 110, 22, 42, 50, 323, 514, 28,
        87, 73, 78, 15, 26, 78, 210, 36, 85, 189, 274, 43, 33, 10, 19, 389, 276,
        312, 360, 83, 59, 130, 431, 67, 230, 52, 93, 125, 670, 892, 600, 38, 48, 147,
        78, 256, 63, 17, 120, 164, 432, 35, 92, 110, 22, 42, 50, 323, 514, 28,
        87, 73, 78, 15, 26, 78, 210, 36, 85, 189, 274, 43, 33, 10, 19, 389, 276,
        312
    ]
    knapsack3.weights = [
        7, 2, 30, 22, 80, 94, 11, 81, 70, 64, 59, 18, 17, 36, 3, 8, 15, 42, 9, 4,
        42, 47, 52, 32, 26, 48, 55, 6, 29, 84, 2, 4, 18, 56, 7, 29, 93, 44, 71,
        3, 86, 66, 31, 65, 9, 79, 20, 65, 52, 13, 7, 2, 30, 22, 80, 94, 11, 81, 70, 64, 59, 18, 17, 36, 3, 8, 15, 42, 9,
        4,
        42, 47, 52, 32, 26, 48, 55, 6, 29, 84, 2, 4, 18, 56, 7, 29, 93, 44, 71,
        3, 86, 66, 31, 65, 9, 79, 20, 65, 52, 13
    ]
    knapsack3.capacity = 1000
    knapsack3.sols = [0, 1, 3, 4, 6, 10, 11, 12, 14, 15, 16, 17, 18, 22, 27, 30, 31, 34, 39, 42, 47, 49, 50, 51, 53, 54,
                      56,
                      60, 61,
                      62, 64, 65, 66, 67, 68, 72, 77, 80, 84, 89, 92, 97, 98, 99]

    knapsack2 = KnapSack()
    knapsack2.size = 50
    knapsack2.profits = [
        360, 83, 59, 130, 431, 67, 230, 52, 93, 125, 670, 892, 600, 38, 48, 147,
        78, 256, 63, 17, 120, 164, 432, 35, 92, 110, 22, 42, 50, 323, 514, 28,
        87, 73, 78, 15, 26, 78, 210, 36, 85, 189, 274, 43, 33, 10, 19, 389, 276,
        312
    ]
    knapsack2.weights = [
        7, 2, 30, 22, 80, 94, 11, 81, 70, 64, 59, 18, 17, 36, 3, 8, 15, 42, 9, 4,
        42, 47, 52, 32, 26, 48, 55, 6, 29, 84, 2, 4, 18, 56, 7, 29, 93, 44, 71,
        3, 86, 66, 31, 65, 9, 79, 20, 65, 52, 13
    ]
    knapsack2.capacity = 850
    knapsack2.sols = [0, 1, 3, 4, 6, 10, 11, 12, 14, 15, 16, 17, 18, 21, 22, 24, 27, 29, 30, 31, 32, 34, 38, 39, 41, 42,
                      44, 47, 48,
                      49]

    knapsacks = [knapsack1, knapsack2, knapsack3]
    # knapsacks = [knapsack1]

    for knap in knapsacks:
        gray_vs_binary(knap)


def gray_vs_bin_max_best_fitness(nfe=30 * 1000, nexec=30):
    knap = KnapSack()
    knap.size = 15
    knap.capacity = 750
    knap.weights = [70, 73, 77, 80, 82, 87, 90, 94, 98, 106, 110, 113, 115, 118, 120]
    knap.profits = [135, 139, 149, 150, 156, 163, 173, 184, 192, 201, 210, 214, 221, 229, 240]
    knap.sols = [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]

    # knap = KnapSack()
    # knap.size = 50
    # knap.profits = [
    #     360, 83, 59, 130, 431, 67, 230, 52, 93, 125, 670, 892, 600, 38, 48, 147,
    #     78, 256, 63, 17, 120, 164, 432, 35, 92, 110, 22, 42, 50, 323, 514, 28,
    #     87, 73, 78, 15, 26, 78, 210, 36, 85, 189, 274, 43, 33, 10, 19, 389, 276,
    #     312
    # ]
    # knap.weights = [
    #     7, 2, 30, 22, 80, 94, 11, 81, 70, 64, 59, 18, 17, 36, 3, 8, 15, 42, 9, 4,
    #     42, 47, 52, 32, 26, 48, 55, 6, 29, 84, 2, 4, 18, 56, 7, 29, 93, 44, 71,
    #     3, 86, 66, 31, 65, 9, 79, 20, 65, 52, 13
    # ]
    # knap.capacity = 850
    # knap.sols = [0, 1, 3, 4, 6, 10, 11, 12, 14, 15, 16, 17, 18, 21, 22, 24, 27, 29, 30, 31, 32, 34, 38, 39, 41, 42,
    #                   44, 47, 48,
    #                   49]

    # knap = KnapSack()
    # knap.size = 100
    # knap.profits = [
    #     360, 83, 59, 130, 431, 67, 230, 52, 93, 125, 670, 892, 600, 38, 48, 147,
    #     78, 256, 63, 17, 120, 164, 432, 35, 92, 110, 22, 42, 50, 323, 514, 28,
    #     87, 73, 78, 15, 26, 78, 210, 36, 85, 189, 274, 43, 33, 10, 19, 389, 276,
    #     312, 360, 83, 59, 130, 431, 67, 230, 52, 93, 125, 670, 892, 600, 38, 48, 147,
    #     78, 256, 63, 17, 120, 164, 432, 35, 92, 110, 22, 42, 50, 323, 514, 28,
    #     87, 73, 78, 15, 26, 78, 210, 36, 85, 189, 274, 43, 33, 10, 19, 389, 276,
    #     312
    # ]
    # knap.weights = [
    #     7, 2, 30, 22, 80, 94, 11, 81, 70, 64, 59, 18, 17, 36, 3, 8, 15, 42, 9, 4,
    #     42, 47, 52, 32, 26, 48, 55, 6, 29, 84, 2, 4, 18, 56, 7, 29, 93, 44, 71,
    #     3, 86, 66, 31, 65, 9, 79, 20, 65, 52, 13, 7, 2, 30, 22, 80, 94, 11, 81, 70, 64, 59, 18, 17, 36, 3, 8, 15, 42, 9,
    #     4, 42, 47, 52, 32, 26, 48, 55, 6, 29, 84, 2, 4, 18, 56, 7, 29, 93, 44, 71,
    #     3, 86, 66, 31, 65, 9, 79, 20, 65, 52, 13
    # ]
    # knap.capacity = 1000
    # knap.sols = [0, 1, 3, 4, 6, 10, 11, 12, 14, 15, 16, 17, 18, 22, 27, 30, 31, 34, 39, 42, 47, 49, 50, 51, 53, 54,
    #              56, 60, 61, 62, 64, 65, 66, 67, 68, 72, 77, 80, 84, 89, 92, 97, 98, 99]

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
            myProblem.constraints[0] = Constraint("<=", knap.capacity)

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


if __name__ == '__main__':
    # comparison()
    gray_vs_bin_max_best_fitness()
#
# # 12332
#
# problem = Problem(1, 1, 1)
# problem.types[0] = Gray(items)
# problem.directions[0] = Problem.MAXIMIZE
# problem.constraints[0] = Constraint("<=", capacity)
# problem.function = knapsack
#
# algorithm = GeneticAlgorithm(problem, variator=GAOperator(HUX(), BitFlipUpdated()))
# algorithm.run(100 * 1000)
#
# for solution in unique(nondominated(algorithm.result)):
#     print(solution.variables, solution.objectives)
#
# # print(sum([sol[i] * profits[i] for i in range(len(sol))]))
# print(sum([profits[i] for i in sol]))
