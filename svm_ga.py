from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Integer, Continuous
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Integer, Continuous
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# # #Loading Heart Dataset
# df = pd.read_csv(r'C:\Users\hassa\PycharmProjects\heart_data.csv')
# Y = df['Outcome']
# X = df.drop('Outcome', axis=1)
# y = Y.to_numpy()
# X = X.to_numpy()

# # #Loading Diabetes Dataset
df = pd.read_csv(r'C:\Users\hassa\PycharmProjects\data.csv')
print(df)
Y = df['Outcome']
X = df.drop('Outcome', axis=1)
y = Y.to_numpy()
X = X.to_numpy()



# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train)

clf = SVC()

# adjust this to better ranges for better results
param_grid = {'C': Continuous(1e-2, 1e10, distribution='log-uniform'),
              'gamma': Continuous(1e-5, 2e-5),
              'kernel': Categorical(['rbf']),
              'degree': Integer(1, 5)
             }

print('1')
# The main class from sklearn-genetic-opt
evolved_estimator = GASearchCV(estimator=clf,
                              cv=5,
                              scoring='accuracy',
                              param_grid=param_grid,
                              n_jobs=-1,
                              verbose=True,
                              population_size=10,
                              generations=25)

evolved_estimator.fit(X_train, y_train)


# Best parameters found
print(evolved_estimator.best_params_)
# Use the model fitted with the best parameters
y_predict_ga = evolved_estimator.predict(X_test)
print(accuracy_score(y_test, y_predict_ga))


from sklearn_genetic.plots import plot_fitness_evolution
plot_fitness_evolution(evolved_estimator)
plt.show()