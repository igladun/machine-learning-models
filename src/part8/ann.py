import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# data preprocessing
#
#
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_x_1 = LabelEncoder()
x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])

labelencoder_x_2 = LabelEncoder()
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])

onehotencoder = OneHotEncoder(categorical_features=[1])
x = onehotencoder.fit_transform(x).toarray()

x = x[:, 1:]

# splitting data to training set and test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# scaling data
from sklearn.preprocessing import StandardScaler

scaler_x = StandardScaler()
x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)

# ANN
#
#


import keras
# for initializing the NN
from keras.models import Sequential
# for building layers
from keras.layers import Dense

# initializing ann
classifier = Sequential()

# adding the input layer and the hidden layer
classifier.add(Dense(output_dim=6, input_dim=11, init='uniform', activation='relu'))

# adding the second hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))


# adding the output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

#compiling ann
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#fitting the training set to the ann
classifier.fit(x_train, y_train, batch_size=10, nb_epoch=100)

x

y_predicted = classifier.predict(x_test)

y_predicted = (y_predicted > 0.5)

# making the confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_predicted)
print(cm)

# details of the model performance
accuracy_rate = (cm[0, 0] + cm[1, 1]) / (np.sum(cm))
error_rate = (cm[0, 1] + cm[1, 0]) / (np.sum(cm))

precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
f1_score = (2 * precision * recall) / (precision + recall)

print('Accuracy rate is {}'.format(accuracy_rate))
print('Error rate is {}'.format(error_rate))
print('Precision is {}'.format(precision))
print('Recall is {}'.format(recall))
print('F1 score is {}'.format(f1_score))

#
# # visualization
# from matplotlib.colors import ListedColormap
#
# # train set
#
# x_set, y_set = x_test, y_test
# X1, X2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
#                      np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha=0.75, cmap=ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
#                 c=ListedColormap(('red', 'green'))(i), label=j)
# plt.title('XXX (Train set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# # plt.show()
#
# # test set
# x_set, y_set = x_test, y_test
# X1, X2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
#                      np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha=0.75, cmap=ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
#                 c=ListedColormap(('red', 'green'))(i), label=j)
# plt.title('XXX (Test set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()
