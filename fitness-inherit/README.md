# Fitness Inheritance for Evolutionary Algorithms (Platypus Library)

This project implements and evaluates fitness inheritance (FI) strategies in the [Platypus](https://github.com/Project-Platypus/Platypus) Python library for evolutionary computation. Fitness inheritance reduces the computational cost of genetic algorithms by estimating the fitness of some individuals based on their parents’ fitness, rather than always computing it from scratch.

## 📚 Background

Fitness inheritance was introduced by Smith, Dike, and Stegmann in 1995 as a way to speed up evolutionary algorithms by reducing the number of expensive fitness evaluations. Instead of evaluating every offspring, a portion of them inherit a fitness value estimated from their parents.

- **Reference:**  
  Smith, R.E., Dike, B.A., & Stegmann, S.A. (1995). Fitness inheritance in genetic algorithms. In Proceedings of the 1995 ACM symposium on Applied computing (pp. 345-350). [ACM Digital Library](https://dl.acm.org/doi/10.1145/315891.316030)

## 🚀 Features

- Implementation of both **average** and **proportional** fitness inheritance strategies in the Platypus framework.
- Experiments on benchmark functions (e.g., CEC 2014) to compare FI approaches against standard genetic algorithms.
- Statistical analysis of performance (solution quality, computation time, etc.).

## 📦 Usage

1. **Clone this repository:**
    ```bash
    git clone https://github.com/your-github-username/fitness-inheritance-platypus.git
    cd fitness-inheritance-platypus
    ```

2. **Install requirements:**
    ```bash
    pip install platypus-opt numpy pandas matplotlib
    ```

3. **Run experiments:**
    - Use the provided scripts to run genetic algorithms with and without fitness inheritance.
    - Configure FI parameters (inheritance probability, method) as needed in the scripts.

4. **Analyze results:**
    - Compare convergence, number of fitness evaluations, and solution quality.
    - Results and plots will be saved in the `results/` directory.

## 📊 Example

Example of running a GA with average fitness inheritance:

```python
from platypus import NSGAII, Problem, Real
from fitness_inheritance import GeneticAlgorithmWithFitnessInheritance

problem = Problem(2, 1)
problem.types[:] = [Real(-5, 5), Real(-5, 5)]

algorithm = GeneticAlgorithmWithFitnessInheritance(problem, fitness_inheritance="average", inheritance_probability=0.5)
algorithm.run(10000)
