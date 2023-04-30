import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import random
import matplotlib.pyplot as plt

pso_inert = 0.89
pso_c0 = 0.8
pso_c1 = 0.99
no_part = 70
run_count = 25
accuracies = []

def fit_func(pos, data):
    X_train, X_test, y_train, y_test = data
    svclassifier = SVC(kernel='rbf', gamma=pos[0], C=pos[1] )
    svclassifier.fit(X_train, y_train)
    y_train_pred = svclassifier.predict(X_train)
    y_test_pred = svclassifier.predict(X_test)
    train_acc = svclassifier.score(X_train, y_train)
    test_acc = svclassifier.score(X_test, y_test)
    return confusion_matrix(y_train,y_train_pred)[0][1] + confusion_matrix(y_train,y_train_pred)[1][0], confusion_matrix(y_test,y_test_pred)[0][1] + confusion_matrix(y_test,y_test_pred)[1][0], train_acc, test_acc


def PSO(data):
    pos = np.array([np.array([random.random() * 10, random.random() * 10]) for _ in range(no_part)])
    pers_best_pos = pos
    pers_best_scr = np.array([float('inf') for _ in range(no_part)])
    global_best_scr = np.array([float('inf'), float('inf')])
    global_best_pos = np.array([float('inf'), float('inf')])
    vel = ([np.array([0, 0]) for _ in range(no_part)])
    iteration = 0

    while iteration < run_count:
        for i in range(no_part):
            fit = fit_func(pos[i], data)
            print("error of particle-", i, "is (training, test)", fit, " At (gamma, c): ",
                  pos[i])
            _, _, train_acc, test_acc = fit_func(pos[i], data)
            print("Accuracy of particle-", i, "is (training, test)", train_acc, test_acc, " At (gamma, c): ",
                  pos[i])


            if (pers_best_scr[i] > fit[1]):
                pers_best_scr[i] = fit[1]
                pers_best_pos[i] = pos[i]


            if (global_best_scr[1] > fit[1]):
                global_best_scr = fit
                global_best_pos = pos[i]


            elif (global_best_scr[1] == fit[1] and global_best_scr[0] > fit[0]):
                global_best_scr = fit
                global_best_pos = pos[i]


        for i in range(no_part):
            new_vel = (pso_inert * vel[i]) + (pso_c0 * random.random()) * (
                        pers_best_pos[i] - pos[i]) + (pso_c1 * random.random()) * (
                                       global_best_pos - pos[i])
            new_pos = new_vel + pos[i]
            pos[i] = new_pos

        iteration = iteration + 1

        gamma, c = np.mean(pos,axis=0)
        X_train, X_test, y_train, y_test = data

        svm_model = SVC(kernel='rbf', gamma=gamma, C=c)
        svm_model.fit(X_train, y_train)

        y_pred = svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Iteration #:", iteration)
        print("Accuracy:", accuracy)
        accuracies.append(accuracy)

    plt.plot(accuracies)
    plt.xlabel('No. of Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Iteration')
    plt.show()


if __name__ == '__main__':
    data1 = pd.read_csv(r'C:\Users\Tariq\PycharmProjects\NeuralProjectFinal\Datasets\asgn1_data.csv')
    data2 = pd.read_csv(r'C:\Users\Tariq\PycharmProjects\NeuralProjectFinal\Datasets\diabetes1.csv')
    data3 = pd.read_csv(r'C:\Users\Tariq\PycharmProjects\NeuralProjectFinal\Datasets\heart_data.csv')

    Y = data2['target']
    X = data2.drop('target', axis=1)

    X = (X - X.mean()) / X.std()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    data = [X_train, X_test, y_train, y_test]
    PSO(data)
