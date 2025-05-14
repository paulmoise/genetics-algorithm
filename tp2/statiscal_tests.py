import pandas as pd
runs = 30
dim = 2
# load from saved file
bestFitness = pd.read_csv('experiment_%d_runs_dim_%d.csv'%(runs, dim), header=[0, 1], index_col=0)
print(bestFitness.describe())
bestFitness.boxplot()