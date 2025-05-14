from typing import List

from platypus import Generator, Solution

from platypus import *
import numpy as np
import random
import optproblems.cec2005


# Function converts the value passed as
# parameter to it's decimal representation
def decimal_converter(num):
    while num > 1:
        num /= 10
    return num


def real2bin(n, precision):
    # split() separates whole number and decimal
    # part and stores it in two separate variables
    print(str(n).split("."))
    whole, dec = str(n).split(".")

    # Convert both whole number and decimal
    # part from string type to integer type
    whole = int(whole)
    dec = int(dec)

    # Convert the whole number part to it's
    # respective binary form and remove the
    # "0b" from it.
    res = bin(whole).lstrip("0b") + "."

    # Iterate the number of times, we want
    # the number of decimal places to be
    for x in range(precision):
        # Multiply the decimal value by 2
        # and separate the whole number part
        # and decimal part
        whole, dec = str((decimal_converter(dec)) * 2).split(".")

        # Convert the decimal part
        # to integer again
        dec = int(dec)

        # Keep adding the integer parts
        # receive to the result variable
        res += whole

    return res


def bin2real(value):
    pass


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


class interceptedFunction(object):
    """ Normalize returned evaluation types in CEC 2005 functions"""

    def __init__(self, initial_function):
        self.__initFunc = initial_function

    def __call__(self, variables):
        objs = self.__initFunc(variables)
        if isinstance(objs, np.floating):
            objs = [objs]
        return objs


class RealGray(Binary):
    def __init__(self, min_value, max_value):
        super().__init__(int(math.log(int(max_value) - int(min_value), 2)) + 1)
        self.min_value = int(min_value)
        self.max_value = int(max_value)

    def rand(self):
        return [random.choice([0, 1]) for _ in range(self.nbits)]

    def encode(self, value):
        return bin2gray(int2bin(value - self.min_value, nbits=self.nbits))

    def decode(self, value):
        value = bin2int(gray2bin(value))

        if value > self.max_value - self.min_value:
            value -= self.max_value - self.min_value

        return self.min_value + value

    def __str__(self):
        return f"Real Gray({self.min_value:f}, {self.max_value:f})"


class RealBinary(Binary):
    def __init__(self, min_value, max_value):
        super().__init__(int(math.log(int(max_value) - int(min_value), 2)) + 1)
        self.min_value = float(min_value)
        self.max_value = float(max_value)

    def rand(self):
        return [random.choice([0, 1]) for _ in range(self.nbits)]

    def encode(self, value):
        return value

    def decode(self, value):
        bounds = [self.min_value, self.max_value]

        largest = 2 ** self.nbits
        # convert bitstring to a string of chars
        chars = ''.join([str(s) for s in value])
        # convert string to integer
        integer = int(chars, 2)
        # scale integer to desired range
        res = bounds[0] + (integer / largest) * (bounds[1] - bounds[0])
        return res

    # def decode(self, value):
    #     value = bin2int(value)
    #
    #     if value > int(self.max_value - self.min_value):
    #         value -= int(self.max_value) - int(self.min_value)
    #
    #     return int(self.min_value + value)

    def __str__(self):
        return f"Real Binary({self.min_value:f}, {self.max_value:f})"

import struct

# def float_to_bits(f):
#     # Pack float value as 4-byte binary string
#     b = struct.pack('f', f)
#
#     # Convert bytes object to list of binary digits
#     bits = []
#     for byte in b:
#         # Convert byte to binary string and remove prefix
#         byte_str = bin(byte)[2:]
#         # Pad binary string with leading zeros to ensure 8 bits
#         byte_str = byte_str.rjust(8, '0')
#         # Add binary string to list of bits
#         bits.extend(list(byte_str))
#
#     return bits

import struct

def float_to_bits(x, n, a, b):
    # Normalize input value to range [0, 1]
    x_norm = (x + n) / (2 * n)

    # Scale normalized value to range [a, b]
    x_scaled = a + x_norm * (b - a)

    # Pack float value as binary string
    b = struct.pack('f', x_scaled)

    # Convert bytes object to list of binary digits
    bits = []
    for byte in b:
        # Convert byte to binary string and remove prefix
        byte_str = bin(byte)[2:]
        # Pad binary string with leading zeros to ensure 8 bits
        byte_str = byte_str.rjust(8, '0')
        # Add binary string to list of bits
        bits.extend(list(byte_str))

    return bits

def printBestIndividual(self):
    """ Function called during each iteration of the algorithm"""
    # select one individual in population regarding its objective value as minimized 'larger=False'
    bestSolution = truncate_fitness(self.population, 1,
                                    larger_preferred=self.problem.directions[0] == self.problem.MAXIMIZE,
                                    getter=objective_key)[0]
    print(bestSolution.objectives)
    # print('{0} {1:4f} {2:4f} {3:7f}'.format(self.nfe, bestSolution.variables[0],
    #                                         bestSolution.variables[1],
    #                                         bestSolution.objectives[0]))


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


def func_F8():
    # # USE PLATYPUS EXPERIMENT AND DO STATISTICAL TESTS
    # use single objective functions from optproblems.cec2005
    nexec = 2

    dim = 2
    nfe = 150
    cec_function = optproblems.cec2005.F1(dim)
    problem = Problem(dim, cec_function.num_objectives, function=interceptedFunction(cec_function))
    problem.CECProblem = cec_function
    # problem.types[:] = Real(cec_function.min_bounds[0], cec_function.max_bounds[0])
    # problem.types[:] = RealBinary(cec_function.min_bounds[0], cec_function.max_bounds[0])
    problem.types[:] = RealBinary(cec_function.min_bounds[0], cec_function.max_bounds[0])
    problem.directions = [Problem.MAXIMIZE if cec_function.do_maximize else Problem.MINIMIZE]
    algorithm = GeneticAlgorithm(problem, variator=GAOperator(SBX(), PM()))
    algorithm.observers = [printBestIndividual, saveStatistics]
    algorithm.run(nfe, callback=Observers)  # 1


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


def decode_float(bounds: List[float], genome: List[int]):
    n_bits = len(genome)
    largest = 2 ** n_bits
    # convert bitstring to a string of chars
    chars = ''.join([str(s) for s in genome])
    # convert string to integer
    integer = int(chars, 2)
    # scale integer to desired range
    value = bounds[0] + (integer / largest) * (bounds[1] - bounds[0])
    return value


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
    # func_F8()
    bits = float_to_bits(1.2345, 2.0, -1.0, 1.0)
    print(bits)  #
    #
    # bits = float_to_bits(3.14159)
    # print(bits)
    print(decode_float([-2.0, 2.0], bits))