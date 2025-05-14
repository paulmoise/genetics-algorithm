from platypus import *
import random
import numpy as np
import matplotlib.pyplot as plt


def gen_shuffle_list(size, start=5):
    tab = [i for i in range(size) if i != start]
    np.random.shuffle(tab)
    tab = [5] + tab
    return tab


def evalTSP(individual):
    """ Evaluate one path for the TSP """
    path = individual[0]
    distances = [euclidean_dist(cities[path[i]], cities[path[(i + 1) % len(cities)]]) for i in gen_shuffle_list(20)]
    return sum(distances)


def showBestTSP(self):
    fig = plt.figure(0, clear=True)
    path = self.fittest.variables[0]
    x_coord = [cities[city][0] for city in path]
    y_coord = [cities[city][1] for city in path]
    plt.plot(x_coord, y_coord, ':b')
    # l = gen_shuffle_list(NB_MAX_CITIES)
    for i in range(NB_MAX_CITIES):
        plt.text(x_coord[i], y_coord[i], str(i))  # TO BE COMPLETED
    plt.draw()
    plt.pause(0.5)


if __name__ == '__main__':
    # max number of cities to visit
    NB_MAX_CITIES = 20
    # create cities positions randomly in [0,1]x[0,1]
    cities = [(random.random(), random.random()) for _ in range(NB_MAX_CITIES)]
    # define a new Problem with 1 variable and 1 objective
    myProblem = Problem(1, 1, function=evalTSP)
    # type of problem variables
    myProblem.types[:] = Permutation(gen_shuffle_list(NB_MAX_CITIES))  # TO BE COMPLETED
    # set the optimization direction for each objective
    myProblem.directions = [Problem.MINIMIZE]
    # choose optimization algorithm
    algorithm = GeneticAlgorithm(myProblem)
    # execution of the algorithm
    algorithm.run(20 * 1000, callback=showBestTSP)
