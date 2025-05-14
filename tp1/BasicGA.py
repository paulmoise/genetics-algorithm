import random as prng
from typing import List, Tuple, Callable
import numpy as np


# def geneticAlgorithm(popsize: int, indsize: int):
#     # initialization
#     population = [[prng.randint(0, 1) for _ in range(indsize)] for _ in range(popsize)]
#     for ind in population:
#         print(ind)
#
#
# def geneticAlnewpopgorithm(evaluate, popsize: int, indsize: int):
#     # initialization
#     population = [[prng.randint(0, 1) for _ in range(indsize)] for _ in range(popsize)]
#     # for ind in population:
#     #     print(ind)
#     fitness = [evaluate(ind) for ind in population]
#
#     for ind, fitness in zip(population, fitness):
#         print(f'ind = {ind} ,fitness = {fitness}')


def maxOnes(individual: List[int]) -> float:
    """Maximize the number of one in the individual"""
    return sum(individual)


def tournamentSelection(fitness: List[float], size: int = 2) -> int:
    """ Select one individual using
    Tournament selection of size size
    Return index of selected individual"""
    selected_index = prng.sample(range(len(fitness)), size)
    best_index = np.argmax([fitness[i] for i in selected_index])
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


def offspring(parent: List[List[int]], fitness: List[float]) -> List[List[int]]:
    offspring = []  # create empty list of offspring
    popsize = len(fitness)
    while len(offspring) < popsize:
        ind1 = tournamentSelection(fitness)
        ind2 = tournamentSelection(fitness)
        off = onePointXover(parent[ind1], parent[ind2])
        for ind in off:
            ind = bitflipMutation(ind)
        offspring.extend(off)
    return offspring[:len(parent)]


def replacement(parent: List[List[int]], fitness: List[float], offspring: List[List[int]],
                fitnessOff: List[float], elitism: int = 1):
    newpop = []  # next generation
    newfit = []  # and corresponding fitness
    popsize = len(parent)  # and corresponding fitness
    while elitism > 0:
        best_index = np.argmax(fitness)  # find best individual in parents
        newpop.append(parent.pop(best_index))  # remove best index from parent and add to newpop
        newfit.append(fitness.pop(best_index))  # same for fitness
        elitism -= 1
    newpop.extend(offspring[:popsize - len(newpop)])
    newfit.extend(fitnessOff[:popsize - len(newfit)])
    return newpop, newfit


def geneticAlgorithm(evaluator: Callable, popsize: int, indsize: int, MAX_NFE: int, observers=[]) -> Tuple[
    List[int], float]:
    NFE = 0  # Number Of Function Evaluations
    # initialization
    population = [[prng.randint(0, 1) for _ in range(indsize)] for _ in range(popsize)]
    # evaluation
    fitness = [evaluator(ind) for ind in population]
    # NFE <- NFE + popsize
    NFE = NFE + len(fitness)
    # generationnal loop

    file_name = "./statistics.scv"
    file = open(file_name, 'w')
    while NFE < MAX_NFE:
        for obs in observers:
            obs(population, fitness, NFE, file=file)
        off = offspring(population, fitness)
        fitnessOff = [evaluator(ind) for ind in off]
        NFE += len(fitnessOff)
        population, fitness = replacement(population, fitness, off, fitnessOff)
    file.close()
    best_index = np.argmax(fitness)

    return population[best_index], fitness[best_index]


def printBestIndividual(population: List[int], fitness: List[float], NFE: int, file=None):
    """ print to console best individual of population during the run"""

    best_index = np.argmax(fitness)
    print('{0} \t {1} \t {2}'.format(NFE, population[best_index], fitness[best_index]))


def saveStatistics(population: List[int], fitness: List[float], NFE: int, file):
    if NFE == len(fitness):
        file.write('NFE\tavg\tmin\t max\tstd')
    else:
        file.write("\n{0}\t{1}\t{2}\t{3}\t{4}".format(NFE, np.average(fitness), np.min(fitness), np.max(fitness),
                                                      np.std(fitness)))


if __name__ == '__main__':
    prng.seed(0)
    popsize = 10
    indsize = 20
    bestInd, bestFit = geneticAlgorithm(evaluator=maxOnes, popsize=popsize, indsize=indsize, MAX_NFE=10 * indsize,
                                        observers=[printBestIndividual, saveStatistics])
