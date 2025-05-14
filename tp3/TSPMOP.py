from platypus import *
import random
import numpy as np
import matplotlib.pyplot as plt


def gen_shuffle_list(size, start=5):
    tab = [i for i in range(size) if i != start]
    np.random.shuffle(tab)
    tab = [5] + tab
    return tab


def evalTSP2D(individual):
    """evaluate a path according to distance & time"""
    path = individual[0]
    distances = [euclidean_dist(cities[path[i]], cities[path[(i + 1) % len(cities)]]) for i in range(len(path))]
    times = [traffic[path[i]][path[(i + 1) % len(cities)]] for i in range(len(path))]
    # times = ...  # TO BE COMPLETED
    return sum(distances), sum(times)


def printParetoFront(self):
    """ Function called during each iteration of the algorithm"""
    for sol in unique(nondominated(self.result)):
        print('{0} {1} {2}'.format(self.nfe, sol.variables[0], sol.objectives))


def showParetoFront(self):
    # get figure indexed #2, avoid creating a new figure each time
    fig = plt.figure(2)
    # get subfigures separated horizontally called pf and path
    pf_fig, path_fig = fig.subplots(1, 2)
    fig.suptitle('Biobjective-TSP on ' + str(NB_MAX_CITIES) + ' cities')
    # non-dominated solutions in population
    non_dominated = unique(nondominated(self.result))  # TO BE COMPLETED
    # index of closest solution to the origin
    chosen = np.argmin([euclidean_dist((0, 0), s.objectives) for s in non_dominated])
    # corresponding path
    path = non_dominated[chosen].variables[0]
    # draw path =====================================
    path_fig.clear()
    path_fig.set_title('Path')
    x_coord = [cities[city][0] for city in path]
    y_coord = [cities[city][1] for city in path]
    path_fig.plot(x_coord, y_coord, color='red')
    for i in range(NB_MAX_CITIES):
        path_fig.text(x_coord[i], y_coord[i], str(path[i]))
    # draw Pareto front =============================
    pf_fig.clear()
    pf_fig.grid()
    pf_fig.scatter(x_coord, y_coord, color='lightGray', label='dominated')  # TO BE COMPLETED
    pf_fig.scatter(x_coord, y_coord, label='non-dominated')  # TO BE COMPLETED
    pf_fig.scatter(x_coord, y_coord, color='red', label='closest to (0,0)')  # TO BE COMPLETED,
    pf_fig.set_xlabel('distance')
    pf_fig.set_ylabel('time')
    pf_fig.set_xlim(0, NB_MAX_CITIES)
    pf_fig.set_ylim(0, NB_MAX_CITIES)
    pf_fig.legend(loc='lower left')
    plt.draw()
    plt.pause(0.1)


def Observers(self):
    """ Defines a set of functions to be called"""
    if hasattr(self, 'observers'):
        for obs in self.observers:
            obs(self)
    else:
        raise NameError("Unknown attribute 'observers'. No method to call.")


if __name__ == '__main__':
    # max number of cities to visit
    NB_MAX_CITIES = 20
    # create cities positions randomly in [0,1]x[0,1]
    cities = [(random.random(), random.random()) for _ in
              range(NB_MAX_CITIES)]  # define a new Problem with 1 variable and 1 objective
    # create time traffic jam between cities as a 2D matrix
    traffic = [[random.random() for _ in range(NB_MAX_CITIES)] for _ in range(NB_MAX_CITIES)]
    # define a new Problem with 1 variable and 2 objectives
    myProblem = Problem(1, 2, function=evalTSP2D)
    # type of problem variables
    myProblem.types[:] = Permutation(range(NB_MAX_CITIES))  # TO BE COMPLETED
    # set the optimization direction for each objective
    myProblem.directions = [Problem.MINIMIZE, Problem.MINIMIZE]
    # choose optimization algorithm
    algorithm = NSGAII(myProblem)
    # methods to be called at each iteration'''
    algorithm.observers = [printParetoFront, showParetoFront]
    # execute algorithm
    algorithm.run(100000, callback=Observers)  # TO BE COMPLETED
