import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:].values

# fixing missing data
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# creating dummy variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# converting row to values of 0,1,2,3...
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# dividing the converted column into separate, to remove the dependency between integers
onehotencoder = OneHotEncoder(categorical_features=[0])
x = onehotencoder.fit_transform(x).toarray()

# splitting data to training set and test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# scaling data
from sklearn.preprocessing import StandardScaler

scaler_x = StandardScaler()
x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# regressor goes here





plt.scatter(x=x, y=y, color='red')
plt.plot(x, regressor.predict(x), color='blue', )
plt.title('Truth or bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



# high resolution plot
# resolution = 0.1
# x_grid = np.arange(min(x), max(x), resolution)
# x_grid= x_grid.reshape(len(x_grid), 1)
#
# plt.scatter(x=x, y=y, color ='red')
# plt.plot(x_grid,regressor.predict(x_grid), color='blue', )
# plt.title('Truth or bluff (SVR)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()
