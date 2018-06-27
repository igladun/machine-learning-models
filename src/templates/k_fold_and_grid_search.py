# k-fold cross validation
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=regressor, X=x, y=y, cv=10, n_jobs=-1)

print(accuracies)
print('accuracy is {}'.format(accuracies.mean()))
print('deviation is {}'.format(accuracies.std()))

# grid search to find best model and parameters
from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators': [100, 200, 300],
              'max_depth': [1, 5, 10, 40, 100],
              }

grid_search = GridSearchCV(estimator=regressor,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10,
                           n_jobs=-1)

grid_search = grid_search.fit(x, y)

print('best score is '.format(grid_search.best_score_))
print('best params are '.format(grid_search.best_params_))
