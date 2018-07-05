import pandas as pd
import numpy as np

data = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

transactions = []

for i in range(0, len(data)):
    transactions.append([str(data.values[i, j]) for j in range(0, 20)])

print(transactions[-1])

from apriori_lib import apriori

rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_lenght=2)
