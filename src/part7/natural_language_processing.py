# Natural language processing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# cleaning text
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

corpus = []

for i in range(data.shape[0]):
    review = re.sub(pattern='[^a-zA-z]', repl=' ', string=data['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if
              word not in set(stopwords.words('english'))]  # iteration through set is faster
    review = ' '.join(review)
    corpus.append(review)

print(corpus[-1])

# creating bag of words model

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()

y = data.iloc[:, 1].values

print(len(x[0]))

# splitting data to training set and test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

# fitting  regression into training set
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_predicted = classifier.predict(x_test)

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
