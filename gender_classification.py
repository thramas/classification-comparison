import numpy as np
from sklearn import tree                                    # Decision Tree
from sklearn.linear_model import LogisticRegression         # Logistic Regression
from sklearn.linear_model import Perceptron                 # Perceptron
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier          # KNN
from sklearn.svm import SVC                                 # SVM

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

# Classifiers
# using the default values for all the hyperparameters
clf_tree = tree.DecisionTreeClassifier()
clf_svm = SVC()
clf_perceptron = Perceptron()
clf_KNN = KNeighborsClassifier(n_neighbors=4)
clf_logreg = LogisticRegression()

# Training the models
clf_tree.fit(X, Y)
clf_svm.fit(X, Y)
clf_perceptron.fit(X, Y)
clf_KNN.fit(X, Y)
clf_logreg.fit(X, Y)

# Testing using the same data
pred_tree = clf_tree.predict(X)
acc_tree = accuracy_score(Y, pred_tree) * 100
print('Accuracy for DecisionTree: {}'.format(acc_tree))

pred_svm = clf_svm.predict(X)
acc_svm = accuracy_score(Y, pred_svm) * 100
print('Accuracy for SVM: {}'.format(acc_svm))

pred_per = clf_perceptron.predict(X)
acc_per = accuracy_score(Y, pred_per) * 100
print('Accuracy for perceptron: {}'.format(acc_per))

pred_KNN = clf_KNN.predict(X)
acc_KNN = accuracy_score(Y, pred_KNN) * 100
print('Accuracy for KNN: {}'.format(acc_KNN))

pred_LR = clf_logreg.predict(X)
acc_LR = accuracy_score(Y, pred_LR) * 100
print('Accuracy for LogisticRegression: {}'.format(acc_LR))

index = np.argmax([acc_tree, acc_svm, acc_per, acc_KNN, acc_LR])
classifiers = {0: 'DecisionTree', 1: 'SVM', 2: 'Perceptron', 3: 'KNN', 4: 'LogisticRegression'}
print('Best gender classifier is {}'.format(classifiers[index]))
