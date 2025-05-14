import math
from typing import List, Callable, Tuple
import random as prng
import numpy as np


def tournamentSelection(fitness: List[float], tournament_size: int = 2) -> int:
    """ Select one individual using
    Tournament selection of size size
    Return index of selected individual"""
    selected_index = prng.sample(range(len(fitness)), tournament_size)
    best_index = np.argmin([fitness[i] for i in selected_index])
    return selected_index[best_index]


def onePointXover(parent1: List[int], parent2: List[int]) -> List[List[int]]:
    child1, child2 = parent1.copy(), parent2.copy()
    min_length = min(len(parent1), len(parent2))  # if parents have not same length
    pointcut = prng.randint(1, min_length - 1)  # to be sure to be inside genome
    child1[pointcut:] = parent2[pointcut:]
    child2[pointcut:] = parent1[pointcut:]
    return [child1, child2]


def bitflipMutation(parent: List[int], probability: float = 0.05) -> List[int]:
    """Bit-flip mutation"""
    assert 0.0 <= probability < 1.0  # probability of mutation should be in [0,1]
    for indexGene in range(len(parent)):
        if prng.random() <= probability:
            parent[indexGene] = 1 - parent[indexGene]
    return parent


def offspring(parent: List[List[int]], fitness: List[float], probability: float, tournament_size) -> List[List[int]]:
    offspring = []  # create empty list of offspring
    popsize = len(fitness)
    while len(offspring) < popsize:
        ind1 = tournamentSelection(fitness, tournament_size)
        ind2 = tournamentSelection(fitness, tournament_size)
        off = onePointXover(parent[ind1], parent[ind2])
        for ind in off:
            ind = bitflipMutation(ind, probability)
        offspring.extend(off)
    return offspring[:len(parent)]


def replacement(parent: List[List[int]], fitness: List[float], offspring: List[List[int]],
                fitnessOff: List[float], elitism: int = 1):
    newpop = []  # next generation
    newfit = []  # and corresponding fitness
    popsize = len(parent)  # and corresponding fitness
    while elitism > 0:
        best_index = np.argmin(fitness)  # find best individual in parents
        newpop.append(parent.pop(best_index))  # remove best index from parent and add to newpop
        newfit.append(fitness.pop(best_index))  # same for fitness
        elitism -= 1
    newpop.extend(offspring[:popsize - len(newpop)])
    newfit.extend(fitnessOff[:popsize - len(newfit)])
    return newpop, newfit


def geneticAlgorithm(evaluator: Callable, popsize: int, indsize: int, MAX_NFE: int, observers=[],
                     probability: float = 0.05, tournament_size: int = 2) -> Tuple[
    List[int], float]:
    NFE = 0  # Number Of Function Evaluations
    # initialization
    population = [[prng.randint(0, 1) for _ in range(indsize)] for _ in range(popsize)]
    # evaluation
    fitness = [evaluator(ind) for ind in population]
    # NFE <- NFE + popsize
    NFE = NFE + len(fitness)
    # generationnal loop

    file_name = "./statistics_continuous_function.scv"
    file = open(file_name, 'w')
    while NFE < MAX_NFE:
        for obs in observers:
            obs(population, fitness, NFE, file=file)
        off = offspring(population, fitness, probability, tournament_size)
        fitnessOff = [evaluator(ind) for ind in off]
        NFE += len(fitnessOff)
        population, fitness = replacement(population, fitness, off, fitnessOff)
    file.close()
    best_index = np.argmin(fitness)

    return population[best_index], fitness[best_index]


def printBestIndividual(population: List[int], fitness: List[float], NFE: int, file=None):
    """ print to console best individual of population during the run"""

    best_index = np.argmin(fitness)
    print('{0} \t {1} \t {2}'.format(NFE, population[best_index], fitness[best_index]))


def saveStatistics(population: List[int], fitness: List[float], NFE: int, file):
    if NFE == len(fitness):
        file.write('NFE\tavg\tmin\t max\tstd')
    else:
        file.write("\n{0}\t{1}\t{2}\t{3}\t{4}".format(NFE, np.average(fitness), np.min(fitness), np.max(fitness),
                                                      np.std(fitness)))


def decode_float_values(bounds: List[List[float]], number_variables: int, genome: List[int]) -> List[float]:
    decoded = list()
    n_bits = len(genome) // number_variables
    largest = 2 ** n_bits
    for i in range(len(bounds)):
        # extract substring
        start, end = i * n_bits, (i * n_bits) + n_bits
        substring = genome[start:end]
        # convert bitstring to a string of chars
        chars = ''.join([str(s) for s in substring])
        # convert string to integer
        integer = int(chars, 2)
        # scale integer to desired range
        value = bounds[i][0] + (integer / largest) * (bounds[i][1] - bounds[i][0])
        # store
        decoded.append(value)
    return decoded


def ackley(individual: List[int]) -> float:
    """Akcley function"""
    a = 20
    b = 0.2
    c = 2 * math.pi
    number_variables = 2
    bounds = [[-5.0, 5.0], [-5.0, 5.0]]
    values = decode_float_values(bounds, number_variables, individual)
    print(f'list = {individual}, decode = {values}', 88888888888888)
    f = -a * math.exp(-b * math.sqrt(1.0 / number_variables) * sum([xi ** 2 for xi in values])) \
        - math.exp(1.0 / number_variables * sum([math.cos(c * xi) for xi in values])) + a + math.e

    return f


if __name__ == '__main__':
    prng.seed(0)
    popsize = 100
    indsize = 16
    bestInd, bestFit = geneticAlgorithm(evaluator=ackley, popsize=popsize, indsize=indsize, MAX_NFE=100 * indsize,
                                        observers=[printBestIndividual, saveStatistics])
