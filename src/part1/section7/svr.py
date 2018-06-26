import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:].values

# scaling data
from sklearn.preprocessing import StandardScaler

scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

# SVR
from sklearn.svm import SVR

regressor = SVR()
regressor.fit(x, y)


results = scaler_y.inverse_transform(regressor.predict(scaler_x.transform(np.array([[6.5]]))))

x_grid = np.arange(min(x), max(x), 0.1)
x_grid= x_grid.reshape(len(x_grid), 1)

plt.scatter(x=x, y=y, color ='red')
plt.plot(x_grid,regressor.predict(x_grid), color='blue', )
plt.title('Truth or bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')

plt.show()
