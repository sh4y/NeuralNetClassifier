import numpy as np
import sklearn.neural_network as nn
import sklearn as sk
import matplotlib.pyplot as plt
import bonnerlib2 as bonner
from matplotlib import cm
import operator

#helper functions
def setPlotTitles(x, y, sup):
    plt.xlabel(x)
    plt.ylabel(y)
    plt.suptitle(sup)

def plotGeneratedData(cluster):
    plt.scatter(cluster[np.where(cluster[:, 2] == 0)][:, 0],
                cluster[np.where(cluster[:, 2] == 0)][:, 1], c='red', s=2)
    plt.scatter(cluster[np.where(cluster[:, 2] == 1)][:, 0],
                cluster[np.where(cluster[:, 2] == 1)][:, 1], c='blue', s=2)


def genData(mu0, mu1, Sigma0, Sigma1, N):
    cluster_one = np.random.multivariate_normal(mu0, Sigma0, N)
    target_one = np.zeros((N,1))
    cluster_two = np.random.multivariate_normal(mu1, Sigma1, N)
    target_two = np.ones((N,1))
    cluster_one = np.append(cluster_one, target_one, axis=1)
    cluster_two = np.append(cluster_two, target_two, axis=1)
    X = np.vstack((cluster_one, cluster_two)).astype(float)
    X = sk.utils.shuffle(X)

    return X[:,[0,1]], X[:,2].reshape((2*N, 1))

def f_sigmoid(z):
    '''N,K = np.shape(z)
    Zmax = np.max(z, axis=1)
    Zmax = np.reshape(Zmax, [N,1])
    z = z - Zmax'''
    return np.divide(1.0, (1.0 + np.exp(-z)))

def f_z(W, w0, h):
    return np.add(np.matmul(h, W), w0)

def f_h(u):
    return np.tanh(u)

def f_u(X, V, v0):
    return np.add(np.matmul(X, V), v0)

# create mean and covariance matrices
mu0 = [0, -1]
mu1 = [-1, 1]
Sigma0 = [[2.0,0.5], [0.5,1.0]]
Sigma1 = [[1.0, -1.0] , [-1.0,2.0]]

print ('\nQuestion 1(a)')
Xtrain, ytrain = genData(mu0, mu1, Sigma0, Sigma1, 1000)
Xtest, ytest = genData(mu0, mu1, Sigma0, Sigma1, 10000)
print Xtest.shape, ytest.shape
train_cluster = np.concatenate((np.asarray(Xtrain), np.asarray(ytrain)), axis=1)
test_cluster = np.concatenate((np.asarray(Xtest), np.asarray(ytest)), axis=1)
print ('\nQuestion 1(b)')

def CreateClassifier():
    classifier = nn.MLPClassifier()
    classifier.learning_rate_init = 0.01
    classifier.tol = pow(10, -10)
    classifier.max_iter = 10000
    classifier.solver = 'sgd'
    classifier.activation = 'tanh'
    return classifier

classifier = CreateClassifier()
classifier.hidden_layer_sizes = (1,)

classifier.fit(Xtrain, ytrain)
weights = np.asarray(classifier.coefs_[0][:,0])
intercepts = np.asarray(classifier.intercepts_).flatten()
#print Xtrain
print 'Weights: ' + str(weights)
print 'Ints: ' + str(intercepts)
plotGeneratedData(train_cluster)
bonner.dfContour(classifier)
setPlotTitles('X', 'Y', 'Question 1(b): Neural net with 1 hidden unit.')
plt.show()

def NeuralNetWithNHiddenUnits(n, subplot_title, best_title):
    classifier = CreateClassifier()
    classifier.hidden_layer_sizes = (n,)
    fig, ax = plt.subplots(3, 3)
    ax = ax.flatten()
    count = 0

    test_accuracies = np.zeros(9)
    classifiers = []

    for i in range(0, 9):
        classifier.fit(Xtrain, ytrain)
        predicted_test_values = np.asarray(classifier.predict(Xtest))
        test_accuracy = np.sum(predicted_test_values == ytest.flatten()) / float(predicted_test_values.shape[0])
        test_accuracies[count] = test_accuracy
        classifiers.append(classifier)
        plt.axes(ax[count])
        plotGeneratedData(train_cluster)
        bonner.dfContour(classifier)
        count += 1

    plt.suptitle(subplot_title)
    plt.show()

    max_accuracy_index = np.where(np.max(test_accuracies) == test_accuracies)[0][0]
    best_classifier = classifiers[max_accuracy_index]
    plotGeneratedData(train_cluster)
    bonner.dfContour(best_classifier)
    setPlotTitles('X', 'Y', best_title)
    print 'Highest test accuracy with ' + str(n) + ' hidden units: ' + str(test_accuracies[max_accuracy_index])
    plt.show()
    return best_classifier

def PlotDecisionBoundaryForBestNClassifier(classifier, title):
    weights = classifier.coefs_[0]
    w0 = np.asarray(weights[0]).reshape(weights[0].shape[0], 1).T
    w1 = np.asarray(weights[1]).reshape(weights[1].shape[0], 1).T
    intercepts = classifier.intercepts_[0]
    decision_boundary_x = np.linspace(-5, 6, 2000).reshape(2000, 1)
    decision_boundary_y = -(w0 * decision_boundary_x + intercepts) / w1
    plt.plot(decision_boundary_x, decision_boundary_y, c='k', linestyle='--')
    plt.xlim(-5, 5)
    plt.ylim(-5, 6)
    plotGeneratedData(train_cluster)
    bonner.dfContour(classifier)
    setPlotTitles('X','Y', title)
    plt.show()

print '\nQuestion 1(c)'
subplot_title = 'Question 1(c): Neural nets with 2 hidden units'
best_title = 'Question 1(c): Best neural net with 2 hidden units.'
two_nn = NeuralNetWithNHiddenUnits(2, subplot_title, best_title)

print '\nQuestion 1(d)'
subplot_title = 'Question 1(d): Neural nets with 3 hidden units'
best_title = 'Question 1(d): Best neural net with 3 hidden units.'
three_nn = NeuralNetWithNHiddenUnits(3, subplot_title, best_title)

print '\nQuestion 1(e)'
subplot_title = 'Question 1(e): Neural nets with 4 hidden units'
best_title = 'Question 1(e): Best neural net with 4 hidden units.'
four_nn = NeuralNetWithNHiddenUnits(4, subplot_title, best_title)

print '\nQuestion 1(f)'
print 'I dont know.'

print '\nQuestion 1(g)'
PlotDecisionBoundaryForBestNClassifier(three_nn, 'Question 1(g): Decision boundaries for 3 hidden units')
print '\nQuestion 1(h)'
PlotDecisionBoundaryForBestNClassifier(two_nn, 'Question 1(h): Decision boundaries for 2 hidden units')
print '\nQuestion 1(i)'
PlotDecisionBoundaryForBestNClassifier(four_nn, 'Question 1(i): Decision boundaries for 4 hidden units')

print '\nQuestion 1(k)'
prediction_probs = three_nn.predict_proba(Xtest)
t_values = np.linspace(0, 1, 1000)
ytest = ytest.flatten()
precision_values = []
recall_values = []
for t in t_values:
    predictions = prediction_probs > t
    true_positives = float(np.sum(predictions[:,1][np.where(ytest == 1)] == 1))
    false_positives = np.sum(predictions[:,1][np.where(ytest == 0)] == 1)
    false_negatives = np.sum(predictions[:,1][np.where(ytest == 1)] == 0)
    recall = true_positives / (true_positives + false_negatives)
    try:
        precision = true_positives / (true_positives + false_positives)
    except:
        precision = 1.0
    finally:
        precision_values.append(precision)
        recall_values.append(recall)
print recall_values
print precision_values
plt.plot(recall_values, precision_values)
setPlotTitles('Recall', 'Precision', 'Question 1(k): Precision/recall curve')
plt.show()

print '\nQuestion 1(j)'
print 'I dont know'

print '\nQuestion 1(l)'
L = sorted(zip(recall_values, precision_values), key=operator.itemgetter(0))
recall_values, precision_values = zip(*L)
auc = np.multiply(np.append(np.asarray(np.ediff1d(recall_values)), np.zeros(1)),np.asarray(precision_values)).sum()
print('Area under curve: ' + str(auc))

print '\nQuestion 3'

Xtrain, ytrain = genData(mu0, mu1, Sigma0, Sigma1, 10000)
Xtest, ytest = genData(mu0, mu1, Sigma0, Sigma1, 10000)


def forward(X, V, v0, W, w0):
    u = f_u(X, V, v0)
    h = f_h(u)
    z = f_z(W, w0, h)
    sigmoid = np.asarray(f_sigmoid(z))
    print "Sigmoid function:", sigmoid
    np.reshape(sigmoid, (sigmoid.shape[0], 1))
    class_0 = np.asarray(1 - sigmoid)
    np.reshape(class_0, (class_0.shape[0], 1))
    probabilities = np.concatenate((class_0, sigmoid), axis=1)
    return u, h, z, probabilities

def fwd(X,V,v0,W,w0):
    u = f_u(X, V, v0)
    h = f_h(u)
    z = f_z(W, w0, h)
    sigmoid = f_sigmoid(z)
    return u,h,z,sigmoid

print '\nQuestion 3(a)'
V = three_nn.coefs_[0]
W = three_nn.coefs_[1]
v0 = three_nn.intercepts_[0]
w0 = three_nn.intercepts_[1]

fwd_prob = forward(Xtest, V, v0, W, w0)[3]
nn3_prob = three_nn.predict_proba(Xtest)
print 'Computed Squared Difference: ' + str(np.sum(np.square(fwd_prob - nn3_prob)))

def predict(o):
    return np.asarray(o >= 0.5).astype(int)

def accuracy(Y, predictions):
    return np.mean(predictions == Y)

def MYdfContour(V, v0, W, w0):
    ax = plt.gca()
    # The extent of xy space
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # form a mesh/grid over xy space
    h = 0.02  # mesh granularity
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    mesh = np.c_[xx.ravel(), yy.ravel()]

    # evaluate the decision functrion at the grid points
    Z = fwd(mesh, V, v0, W, w0)[3]# *** MODIFIED ***

    # plot the contours of the decision function
    Z = Z.reshape(xx.shape)
    mylevels = np.linspace(0.0, 1.0, 11)
    ax.contourf(xx, yy, Z, levels=mylevels, cmap=cm.RdBu, alpha=0.5)

    # draw the decision boundary in solid black
    ax.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='solid')

def bgd(J, K, lrate):
    #1 row for each input factor
    #n * m cols (where n is number of rows, m is number of hidden units)
    # V should be 2x3
    # W should be 3x1
    V = np.random.randn(Xtrain.shape[1], J).astype(float)
    v0 = np.zeros((J, )).astype(float)
    W = np.random.randn(J, 1).astype(float)
    w0 = 0.0
    training_losses = []
    training_accuracies = []
    test_accuracies = []
    for i in range(K):
        #Create Parameters
        u,h,z,sigmoid = fwd(Xtrain, V, v0, W, w0)
        #Adjust weights
        cosh = 1 - h**2
        error = sigmoid - ytrain
        W = W - (lrate/Xtrain.shape[0]) * np.matmul(error.T, h).reshape(J, 1)
        w0 = w0 - (lrate/Xtrain.shape[0]) * np.sum(error, axis=0)
        V = V - (lrate/Xtrain.shape[0]) * np.matmul((np.matmul(error, W.T) * cosh).T, Xtrain).T
        v0 = v0 - (lrate/Xtrain.shape[0]) * np.sum((np.matmul(error, W.T) * cosh), axis=0)
        if i % 10 == 0 and i != 0:
            training_loss = (1.0/Xtrain.shape[0]) * np.sum(np.multiply(-1.0 * ytrain, np.log(sigmoid)) - np.multiply(1.0 - ytrain, np.log(1.0 - sigmoid)))
            training_losses.append(training_loss)

            train_predictions = predict(sigmoid)
            training_accuracy = accuracy(ytrain, train_predictions)
            training_accuracies.append(training_accuracy)

            t_u,t_h,t_z,t_sigmoid = fwd(Xtest, V, v0, W, w0)
            test_predictions = predict(t_sigmoid)
            test_accuracy = accuracy(ytest, test_predictions)
            test_accuracies.append(test_accuracy)


    print W
    print V
    print w0
    print v0

    print 'Final Training Accuracy: ' + str(training_accuracies[-1])
    print 'Final Test Accuracy: ' + str(test_accuracies[-1])
    print 'Learning Rate:' + str(lrate)

    plt.semilogx(range(10, K, 10), training_losses)
    setPlotTitles('Iteration', 'Loss', 'Question 3(b): training loss for bgd')
    plt.show()

    plt.semilogx(range(10, K, 10), test_accuracies)
    plt.semilogx(range(10, K, 10), training_accuracies)
    setPlotTitles('Iteration', 'Accuracy', 'Question 3(b): training and test accuracies for bgd')
    plt.show()

    plt.plot(range(K/2, K, 10), test_accuracies[K/20 -1:])
    setPlotTitles('Iteration', 'Accuracy', 'Question 3(b): final test accuracy')
    plt.show()

    plt.plot(range(K/2, K, 10), training_losses[K/20 -1:])
    setPlotTitles('Iteration', 'Loss', 'Question 3(b): final training loss')
    plt.show()

    plotGeneratedData(train_cluster)
    setPlotTitles('x', 'y', 'Question 3(b): decision boundary for my neural net')
    MYdfContour(V, v0, W, w0)
    plt.show()

def sgd(J, K, lrate):
    V = np.random.randn(Xtrain.shape[1], J).astype(float)
    v0 = np.zeros((J, )).astype(float)
    W = np.random.randn(J, 1).astype(float)
    w0 = 0.0

    training_losses = []
    training_accuracies = []
    test_accuracies = []

    # shuffle train and test data in order
    combined_train_dataset = np.hstack((Xtrain, ytrain.reshape(ytrain.shape[0], 1)))
    np.random.shuffle(combined_train_dataset)
    combined_test_dataset = np.hstack((Xtest, ytest.reshape(ytest.shape[0], 1)))
    np.random.shuffle(combined_test_dataset)

    # break apart the shuffled dataset into their X and Y components
    shuffledXtrain = combined_train_dataset[:, :-1]
    shuffledYtrain = combined_train_dataset[:, -1]
    shuffledXtest = combined_test_dataset[:, :-1]
    shuffledYtest = combined_test_dataset[:, -1]

    for epoch in range(K):
        for i in range(shuffledYtrain.shape[0] // 50):
            start_index = i * 50
            end_index = start_index + 50
            batchXtrain = (shuffledXtrain[start_index:end_index, ]).reshape(50, 2)
            batchYtrain = (shuffledYtrain[start_index:end_index, ]).reshape(50, 1)
            batchXtest = (shuffledXtest[start_index:end_index, ]).reshape(50, 2)
            batchYtest = (shuffledYtest[start_index:end_index, ]).reshape(50, 1)

            u, h, z, sigmoid = fwd(batchXtrain, V, v0, W, w0)
            # Adjust weights
            cosh = 1 - h ** 2
            error = sigmoid - batchYtrain
            W = W - (lrate / batchXtrain.shape[0]) * np.matmul(error.T, h).reshape(J, 1)
            w0 = w0 - (lrate / batchXtrain.shape[0]) * np.sum(error, axis=0)
            V = V - (lrate / batchXtrain.shape[0]) * np.matmul((np.matmul(error, W.T) * cosh).T, batchXtrain).T
            v0 = v0 - (lrate / batchXtrain.shape[0]) * np.sum((np.matmul(error, W.T) * cosh), axis=0)

        training_loss = (1.0 / batchXtrain.shape[0]) * np.sum(
            np.multiply(-1.0 * batchYtrain, np.log(sigmoid)) - np.multiply(1.0 - batchYtrain, np.log(1.0 - sigmoid)))
        training_losses.append(training_loss)

        train_predictions = predict(sigmoid)
        training_accuracy = accuracy(batchYtrain, train_predictions)
        training_accuracies.append(training_accuracy)

        t_u, t_h, t_z, t_sigmoid = fwd(batchXtest, V, v0, W, w0)
        test_predictions = predict(t_sigmoid)
        test_accuracy = accuracy(batchYtest, test_predictions)
        test_accuracies.append(test_accuracy)

    print 'Final Training Accuracy: ' + str(training_accuracies[-1])
    print 'Final Test Accuracy: ' + str(test_accuracies[-1])
    print 'Learning Rate:' + str(lrate)

    plt.semilogx(range(0, K), training_losses)
    setPlotTitles('Iteration', 'Loss', 'Question 3(c): training loss for sgd')
    plt.show()

    plt.semilogx(range(0, K), test_accuracies)
    plt.semilogx(range(0, K), training_accuracies)
    setPlotTitles('Iteration', 'Accuracy', 'Question 3(c): training and test accuracies for sgd')
    plt.show()

    plt.plot(range(K/2, K), test_accuracies[K/2:])
    setPlotTitles('Iteration', 'Accuracy', 'Question 3(c): final test accuracy')
    plt.show()

    plt.plot(range(K/2, K), training_losses[K/2:])
    setPlotTitles('Iteration', 'Loss', 'Question 3(c): final training loss')
    plt.show()

    plotGeneratedData(train_cluster)
    setPlotTitles('x', 'y', 'Question 3(c): decision boundary for my neural net')
    MYdfContour(V, v0, W, w0)
    plt.show()

print '\nQuestion 3(b)'
bgd(3, 1000, 0.1)
print '\nQuestion 3(c)'
sgd(3, 20, 0.1)

print '\nQuestion 3(d)'
print 'SGD is faster than BGD because SGD still utilizes all the data but in smaller sweeps.'
print 'Since the data is randomly shuffled, it prevents overfitting to the training data in smaller epoch ranges.'
