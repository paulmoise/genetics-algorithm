from platypus import GeneticAlgorithm, truncate_fitness, objective_key, Problem, Real
from math import exp, cos, sqrt, e, pi


def Ackley(individual):
    a = 20
    b = 0.2
    c = 2 * pi
    fitness = -a * exp(-b * sqrt(1.0 / len(individual)) * sum([xi ** 2 for xi in individual])) - \
              exp(1.0 / len(individual) * sum([cos(c * xi) for xi in individual])) + a + e
    return fitness


