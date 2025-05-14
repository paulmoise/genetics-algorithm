import optproblems.cec2005
import numpy as np
from platypus import *
import pandas as pd
import matplotlib.pyplot as plt
from ackley import printBestIndividual, saveStatistics, Observers
from fitness_inheritance import GeneticAlgorithmFitnessInheritance


class interceptedFunction(object):
    """ Normalize returned evaluation types in CEC 2005 functions"""

    def __init__(self, initial_function):
        self.__initFunc = initial_function

    def __call__(self, variables):
        objs = self.__initFunc(variables)
        if isinstance(objs, np.floating):
            objs = [objs]
        return objs


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


# if __name__ == '__main__':
#     # # USE PLATYPUS EXPERIMENT AND DO STATISTICAL TESTS
#     # use single objective functions from optproblems.cec2005
#     nexec = 3
#     # for all wanted dimensions
#     dims = [2]  # 2,10,30,50 are common for all cec functions
#     Xovers = [SBX(), PCX(), UNDX(), SPX()]  # all possible Xovers
#     # Mutations = [PM(), CompoundMutation(), Replace(), Swap(), BitFlip(), UM(), UniformMutation(),
#     Mutations = [PM(), UM(), UniformMutation(probability=0.01, perturbation=0.5)]  # all
#     # build a list of problems for all dimensions
#     problems = []
#     results = OrderedDict()
#     for dim in dims:
#         nfe = 500
#         for cec_function in optproblems.cec2005.CEC2005(dim):
#             # Platypus problem based on CEC functions using our intercepted class
#             problem = Problem(dim, cec_function.num_objectives, function=interceptedFunction(cec_function))
#             problem.CECProblem = cec_function
#             problem.types[:] = Real(-50, 50) if cec_function.min_bounds is None else Real(cec_function.min_bounds[0],
#                                                                                           cec_function.max_bounds[0])
#             problem.directions = [Problem.MAXIMIZE if cec_function.do_maximize else Problem.MINIMIZE]
#             # a couple (problem_instance,problem_name) Mandatory because all functions are instance of Problem class
#
#             name = type(cec_function).__name__ + '_' + str(dim) + 'D'
#             problems.append((problem, name))
#         # a list of (type_algorithm, kwargs_algorithm, name_algorithm)
#         algorithms = [(GeneticAlgorithm, dict(variator=GAOperator(x, m)), 'GA_' + type(x).__name__ + '_' +
#                        type(m).__name__) for x in Xovers for m in Mutations]
#         results = results | experiment(algorithms=algorithms, problems=problems, nfe=nfe, seeds=nexec,
#                                        display_stats=True)
#         indicators = [bestFitness()]
#         indicators_result = calculate(results, indicators)
#         display(indicators_result, ndigits=3)


def func_F8():
    # # USE PLATYPUS EXPERIMENT AND DO STATISTICAL TESTS
    # use single objective functions from optproblems.cec2005
    nexec = 2

    dim = 2
    nfe = 5000
    cec_function = optproblems.cec2005.F8(dim)
    problem = Problem(dim, cec_function.num_objectives, function=interceptedFunction(cec_function))
    problem.CECProblem = cec_function
    problem.types[:] = Real(-32, 32) if cec_function.min_bounds is None else Real(cec_function.min_bounds[0],
                                                                                  cec_function.max_bounds[0])
    problem.directions = [Problem.MAXIMIZE if cec_function.do_maximize else Problem.MINIMIZE]
    algorithm = GeneticAlgorithm(problem, variator=GAOperator(SBX(), PM()))
    algorithm.observers = [printBestIndividual, saveStatistics]
    algorithm.run(nfe, callback=Observers)  # 1
    # indicators = [bestFitness()]
    # indicators_result = calculate(results, indicators)
    # display(indicators_result, ndigits=3)


def compare_f8_f10():
    # # USE PLATYPUS EXPERIMENT AND DO STATISTICAL TESTS
    # use single objective functions from optproblems.cec2005
    nexec = 3
    # for all wanted dimensions
    dims = [2]  # 2,10,30,50 are common for all cec functions
    Xovers = [SBX()]  # all possible Xovers
    # Mutations = [PM(), CompoundMutation(), Replace(), Swap(), BitFlip(), UM(), UniformMutation(),
    Mutations = [PM()]  # all
    # build a list of problems for all dimensions
    problems = []
    results = OrderedDict()
    cec_functions = [optproblems.cec2005.F8, optproblems.cec2005.F10]
    for dim in dims:
        nfe = 500
        for cec_function in cec_functions:
            cec_function = cec_function(dim)
            # Platypus problem based on CEC functions using our intercepted class
            problem = Problem(dim, cec_function.num_objectives, function=interceptedFunction(cec_function))
            problem.CECProblem = cec_function
            problem.types[:] = Real(cec_function.min_bounds[0], cec_function.max_bounds[0])
            problem.directions = [Problem.MAXIMIZE if cec_function.do_maximize else Problem.MINIMIZE]

            name = type(cec_function).__name__ + '_' + str(dim) + 'D'
            problems.append((problem, name))
        # a list of (type_algorithm, kwargs_algorithm, name_algorithm)
        algorithms = [
            (GeneticAlgorithmFitnessInheritance, dict(variator=GAOperator(x, m)), 'GA_' + type(x).__name__ + '_' +
             type(m).__name__) for x in Xovers for m in Mutations]
        results = results | experiment(algorithms=algorithms, problems=problems, nfe=nfe, seeds=nexec,
                                       display_stats=True)
        indicators = [bestFitness()]
        indicators_result = calculate(results, indicators)
        # display(indicators_result, ndigits=3)
        data = dict()
        for key_algorithm, algorithm in indicators_result.items():
            print(key_algorithm, algorithm)
            for key_problem, problem in algorithm.items():
                data[(key_algorithm, key_problem)] = indicators_result[key_algorithm][key_problem]['bestFitness']
        df_bestFitness = pd.DataFrame(data=data)
        df_bestFitness.to_csv('experiment_%d_runs.csv' % nexec)
        # print dataframe statistics
        print(df_bestFitness.describe())
        # reverse MultiIndex levels, bestFitness['Algortihm']['Problem'] -> bestFitness['Problem']['Algortihm']
        df_bestFitness = df_bestFitness.stack(level=0).unstack()
        # plot columns concerned by problem F1_2D
        df_bestFitness[name].plot()
        plt.show()


def test_compare_f8_f10(nexec=3, nfe=1000, ndim=2):
    # # USE PLATYPUS EXPERIMENT AND DO STATISTICAL TESTS
    # use single objective functions from optproblems.cec2005
    # for all wanted dimensions
    dims = [ndim]  # 2,10,30,50 are common for all cec functions
    Xovers = [SBX(), PCX()]  # all possible Xovers
    # Mutations = [PM(), CompoundMutation(), Replace(), Swap(), BitFlip(), UM(), UniformMutation(),
    Mutations = [PM()]  # all
    # build a list of problems for all dimensions
    problems = []
    results = OrderedDict()
    # cec_functions = [optproblems.cec2005.F8, optproblems.cec2005.F10]
    cec_functions = [optproblems.cec2005.F8]
    for dim in dims:
        for cec_function in cec_functions:
            cec_function = cec_function(dim)
            # Platypus problem based on CEC functions using our intercepted class
            problem = Problem(dim, cec_function.num_objectives, function=interceptedFunction(cec_function))
            problem.CECProblem = cec_function
            problem.types[:] = Real(cec_function.min_bounds[0], cec_function.max_bounds[0])
            problem.directions = [Problem.MAXIMIZE if cec_function.do_maximize else Problem.MINIMIZE]

            name = type(cec_function).__name__ + '_' + str(dim) + 'D'
            problems.append((problem, name))
        # a list of (type_algorithm, kwargs_algorithm, name_algorithm)
        algorithms = [(GeneticAlgorithm, dict(variator=GAOperator(x, m)), 'GA_' + type(x).__name__ + '_' +
                       type(m).__name__) for x in Xovers for m in Mutations]
        results = results.update(experiment(algorithms=algorithms, problems=problems, nfe=nfe, seeds=nexec,
                                            display_stats=True))
        indicators = [bestFitness()]
        indicators_result = calculate(results, indicators)
        # display(indicators_result, ndigits=3)
        data = dict()
        for key_algorithm, algorithm in indicators_result.items():
            print(key_algorithm, algorithm)
            for key_problem, problem in algorithm.items():
                data[(key_algorithm, key_problem)] = indicators_result[key_algorithm][key_problem]['bestFitness']
        df_bestFitness = pd.DataFrame(data=data)
        df_bestFitness.to_csv(f'experiment_{nexec}_runs_dim_{ndim}.csv')
        # print dataframe statistics
        print(df_bestFitness.describe())
        # reverse MultiIndex levels, bestFitness['Algortihm']['Problem'] -> bestFitness['Problem']['Algortihm']
        df_bestFitness = df_bestFitness.stack(level=0).unstack()
        # plot columns concerned by problem F1_2D
        df_bestFitness[name].plot()
        plt.show()


if __name__ == '__main__':
    # compare_f8_f10()
    test_compare_f8_f10(30, 10000, 2)
    test_compare_f8_f10(30, 10000, 10)
