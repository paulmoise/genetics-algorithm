import random
from platypus import Problem, Permutation, GeneticAlgorithm
import matplotlib.pyplot as plt


def evalTSP(individual):
    """ Evaluate one path for the TSP """
    path = individual[0]
    distances = [distance(cities[path[i]], cities[path[(i + 1) % len(cities)]]) for i in range(len(path))]
    return sum(distances)


def distance(city1, city2):
    """ Euclidean distance between two cities """
    return ((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2) ** 0.5


def showBestTSP(self):
    fig = plt.figure(0, clear=True)
    path = self.fittest.variables[0]
    x_coord = [cities[city][0] for city in path]
    y_coord = [cities[city][1] for city in path]
    plt.plot(x_coord, y_coord, ':b')
    for i in range(NB_MAX_CITIES):
        plt.text(cities[i][0], cities[i][1], str(i))
        plt.draw()
        plt.pause(0.1)


if __name__ == '__main__':
    # max number of cities to visit
    NB_MAX_CITIES = 20
    # create cities positions randomly in [0,1]x[0,1]
    cities = [(random.random(), random.random()) for _ in range(NB_MAX_CITIES)]
    # define a new Problem with 1 variable and 1 objective
    myProblem = Problem(1, 1, function=evalTSP)
    # type of problem variables
    myProblem.types[:] = Permutation(range(NB_MAX_CITIES))
    # set the optimization direction for each objective
    myProblem.directions = [Problem.MINIMIZE]
    # choose optimization algorithm
    algorithm = GeneticAlgorithm(myProblem)
    # execution of the algorithm
    algorithm.run(10000)
    # show the best solution
    showBestTSP(algorithm)
    input("Press Enter to finish...")
