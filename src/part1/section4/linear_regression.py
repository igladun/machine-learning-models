import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# splitting data to training set and test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 3, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# predicting the test set

y_prediction = regressor.predict(x_test)

# visualising test set vs actual data

plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.title('Years of experince vs salary (train set)')

# visualising test set vs actual data

plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, regressor.predict(x_test), color='blue')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.title('Years of experince vs salary (test data)')
# plt.show()

# predicting salary for 20 years of experience
print(regressor.predict(20))
