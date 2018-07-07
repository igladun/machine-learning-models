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
