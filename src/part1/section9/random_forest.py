import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:].values


# regressor goes here
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(x,y)

print('result is {}'.format(regressor.predict(6.5)))

resolution = 0.01
x_grid = np.arange(min(x), max(x), resolution)
x_grid = x_grid.reshape(len(x_grid), 1)

plt.scatter(x=x, y=y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue', )
plt.title('Random forest regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
