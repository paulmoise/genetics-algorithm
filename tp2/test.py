#!/usr/bin/env python3
from platypus import NSGAII, DTLZ2
import matplotlib.pyplot as plt

problem = DTLZ2()

algorithm = NSGAII(problem)
algorithm.run(10000)

plt.scatter([s.objectives[0] for s in algorithm.result],
            [s.objectives[1] for s in algorithm.result]
            )

plt.xlim([0, 1.1])
plt.ylim([0, 1.1])
plt.xlabel('$f_1(x)$')
plt.ylabel('$f_2(x)$')
plt.show()
