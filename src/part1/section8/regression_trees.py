import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:].values


# regressor
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x, y)

print('Result is {}'.format(regressor.predict(6.5)))


# plt.scatter(x=x, y=y, color='red')
# plt.plot(x, regressor.predict(x), color='blue', )
# plt.title('Truth or bluff (Regression tree)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()


x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)

plt.scatter(x=x, y=y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue', )
plt.title('Truth or bluff (Regression tree)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
