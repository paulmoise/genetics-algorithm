import random as prng
from typing import List, Tuple, Callable

import numpy as np


def max_ones(individual: List[int]) -> float:
    """Maximize the number of one in the individual"""
    return sum(individual)


def tournament_selection(fitness: List[float], size: int = 2) -> int:
    """ Select one individual using
    Tournament selection of size size
    Return index of selected individual"""
    selected_index = prng.sample(range(len(fitness)), size)
    r = [fitness[i] for i in selected_index]
    best_index = np.argmax(r)
    return selected_index[best_index]


def one_point_xover(parent1: List[int], parent2: List[int]) -> List[List[int]]:
    child1, child2 = parent1.copy(), parent2.copy()
    min_length = min(len(parent1), len(parent2))  # if parents have not same length
    pointcut = prng.randint(1, min_length - 1)  # to be sure to be inside genome
    child1[pointcut:] = parent2[pointcut:]
    child2[pointcut:] = parent1[pointcut:]
    return [child1, child2]


def bit_flip_mutation(parent: List[int], probability: float = 0.05) -> List[int]:
    """Bit-flip mutation"""
    assert 0.0 <= probability < 1.0  # probability of mutation should be in [0,1]
    for indexGene in range(len(parent)):
        if prng.random() <= probability:
            parent[indexGene] = 1 - parent[indexGene]
    return parent


def offspring(parent: List[List[int]], fitness: List[float], n_ind=2) -> Tuple[List[List[int]], List[float]]:
    offspring = []  # create empty list of offspring
    fitness_off = []  # empty list to save fitness of spring
    popsize = len(parent) - n_ind  # population_size - lambda
    while len(offspring) < popsize:
        ind1 = tournament_selection(fitness)
        ind2 = tournament_selection(fitness)
        off = one_point_xover(parent[ind1], parent[ind2])
        fit_of = []
        for ind in off:
            ind = bit_flip_mutation(ind)
            fit_of.append((fitness[ind1] + fitness[ind2]) / 2)
        offspring.extend(off)
        fitness_off.extend(fit_of)
    return offspring, fitness_off


def replacement(parent: List[List[int]], fitness: List[float], offspring: List[List[int]],
                fitnessOff: List[float], elitism: int = 2):
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


def select_best_individuals(parents, fitness, elitism=2):
    selected_parents = []
    new_fit = []
    while elitism > 0:
        best_index = np.argmax(fitness)
        selected_parents.append(parents.pop(best_index))
        new_fit.append(fitness.pop(best_index))
        elitism -= 1
    return selected_parents, new_fit


def fitness_inheritance_averaged(parent, fitness, pop_sim):
    offspring = []  # create empty list of offspring
    popsize = len(fitness)
    while len(offspring) < popsize:
        ind1 = tournament_selection(fitness)
        ind2 = tournament_selection(fitness)
        off = one_point_xover(parent[ind1], parent[ind2])
        for ind in off:
            ind = bit_flip_mutation(ind)
        offspring.extend(off)
    return offspring[:len(parent)]


def geneticAlgorithm(evaluator: Callable, popsize: int, indsize: int, MAX_NFE: int, p_sim, observers=[]) -> Tuple[
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

        # get offspring with specific inherited fitness len(off) = popsize - 2
        off, fit_off = offspring(population, fitness, n_ind=2)
        # select two best parents from the parents list
        selected_parents, selected_parents_fitness = select_best_individuals(population, fitness)

        population = off + selected_parents
        fitness = fit_off + selected_parents_fitness  # init the fitness array

        number_of_selection = int(popsize * p_sim)  # number of selection as random
        selected_index = prng.sample(range(0, len(population)), number_of_selection)

        for i in selected_index:
            fitness[i] = evaluator(population[i])

        NFE += number_of_selection
        # rest_ind = list(set(range(len(population))).difference(set(selected_index)))
        # for i in rest_ind:
        #     if i < len(off):
        #         fitness[i] = fit_off[i]
        #     else:
        #         fitness[i] = selected_parents_fitness
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
    bestInd, bestFit = geneticAlgorithm(evaluator=max_ones, popsize=popsize, indsize=indsize, MAX_NFE=10 * indsize,
                                        p_sim=1,
                                        observers=[printBestIndividual, saveStatistics])
