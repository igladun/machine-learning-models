import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values



#simple linear regression
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(x, y)

from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree=4)
polynomial_x = polynomial_regression.fit_transform(x)

linear_regression2 = LinearRegression()
linear_regression2.fit(polynomial_x,y)


x_grid = np.arange(min(x), max(x), 0.1)
x_grid= x_grid.reshape(len(x_grid), 1)

plt.scatter(x=x, y=y, color ='red')
plt.plot(x_grid,linear_regression2.predict(polynomial_regression.fit_transform(x_grid)), color='blue', )
plt.title('Truth or bluff (polynomial linear regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')

print('estimated salary is{}'.format(linear_regression2.predict(polynomial_regression.fit_transform(6.5))))

