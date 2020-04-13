import time
import random
import numpy as np
import matplotlib.pyplot as plt


"""
This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2019 by Dmytro Kalpakchi.
"""
class LogisticRegression(object):
    """
    This class performs logistic regression using batch gradient descent
    or stochastic gradient descent
    """

    def __init__(self, theta=None):
        """
        Constructor. Imports the data and labels needed to build theta.

        @param theta    A ready-made model
        """
        theta_check = theta is not None

        if theta_check:
            self.FEATURES = len(theta)
            self.theta = theta

        #  ------------- Hyperparameters ------------------ #
        self.LEARNING_RATE = 0.1            # The learning rate.
        self.MINIBATCH_SIZE = 256           # Minibatch size
        self.PATIENCE = 5                   # A max number of consequent epochs with monotonously
                                            # increasing validation loss for declaring overfitting
        # ---------------------------------------------------------------------- 


    def init_params(self, x, y):
        """
        Initializes the trainable parameters of the model and dataset-specific variables
        """
        # To limit the effects of randomness
        np.random.seed(524287)

        # Number of features
        self.FEATURES = len(x[0]) + 1

        # Number of classes
        self.CLASSES = len(np.unique(y))

        # Training data is stored in self.x (with a bias term) and self.y
        self.x, self.y, self.xv, self.yv = self.train_validation_split(np.concatenate((np.ones((len(x), 1)), x), axis=1), y)

        # Number of datapoints.
        self.TRAINING_DATAPOINTS = len(self.x)

        # The weights we want to learn in the training phase.
        K = np.sqrt(1 / self.FEATURES)
        self.theta = np.random.uniform(-K, K, (self.FEATURES, self.CLASSES))

        # The current gradient.
        self.gradient = np.zeros((self.FEATURES, self.CLASSES))


        print("NUMBER OF DATAPOINTS: {}".format(self.TRAINING_DATAPOINTS))
        print("NUMBER OF CLASSES: {}".format(self.CLASSES))


    def train_validation_split(self, x, y, ratio=0.9):
        """
        Splits the data into training and validation set, taking the `ratio` * 100 percent of the data for training
        and `1 - ratio` * 100 percent of the data for validation.

        @param x        A (N, D + 1) matrix containing training datapoints
        @param y        An array of length N containing labels for the datapoints
        @param ratio    Specifies how much of the given data should be used for training
        """
        #
        # YOUR CODE HERE
        #
        return [], [], [], []


    def loss(self, x, y):
        """
        Calculates the loss for the datapoints present in `x` given the labels `y`.
        """
        #
        # YOUR CODE HERE
        #
        return -1


    def conditional_log_prob(self, label, datapoint):
        """
        Computes the conditional log-probability log[P(label|datapoint)]
        """
        #
        # YOUR CODE HERE
        #
        return -1


    def compute_gradient(self, minibatch):
        """
        Computes the gradient based on a mini-batch
        """
        #
        # YOUR CODE HERE
        #
        pass


    def fit(self, x, y):
        """
        Performs Mini-batch Gradient Descent.
        
        :param      x:      Training dataset (features)
        :param      y:      The list of training labels
        """
        self.init_params(x, y)

        self.init_plot(self.FEATURES)

        start = time.time()
        
        #
        # YOUR CODE HERE
        #

        print(f"Training finished in {time.time() - start} seconds")


    def get_log_probs(self, x):
        """
        Get the log-probabilities for all labels for the datapoint `x`
        
        :param      x:    a datapoint
        """
        if self.FEATURES - len(x) == 1:
            x = np.array(np.concatenate(([1.], x)))
        else:
            raise ValueError("Wrong number of features provided!")
        return [self.conditional_log_prob(c, x) for c in range(self.CLASSES)]


    def classify_datapoints(self, x, y):
        """
        Classifies datapoints
        """
        confusion = np.zeros((self.CLASSES, self.CLASSES))

        x = np.concatenate((np.ones((len(x), 1)), x), axis=1)

        no_of_dp = len(y)
        for d in range(no_of_dp):
            best_prob, best_class = -float('inf'), None
            for c in range(self.CLASSES):
                prob = self.conditional_prob(c, x[d])
                if prob > best_prob:
                    best_prob = prob
                    best_class = c
            confusion[best_class][y[d]] += 1

        print('                       Real class')
        print('                 ', end='')
        print(' '.join('{:>8d}'.format(i) for i in range(self.CLASSES)))
        for i in range(self.CLASSES):
            if i == 0:
                print('Predicted class: {:2d} '.format(i), end='')
            else:
                print('                 {:2d} '.format(i), end='')
            print(' '.join('{:>8.3f}'.format(confusion[i][j]) for j in range(self.CLASSES)))
        acc = sum([confusion[i][i] for i in range(self.CLASSES)]) / no_of_dp
        print("Accuracy: {0:.2f}%".format(acc * 100))


    def print_result(self):
        print(' '.join(['{:.2f}'.format(x) for x in self.theta]))
        print(' '.join(['{:.2f}'.format(x) for x in self.gradient]))

    # ----------------------------------------------------------------------

    def update_plot(self, *args):
        """
        Handles the plotting
        """
        if self.i == []:
            self.i = [0]
        else:
            self.i.append(self.i[-1] + 1)

        for index, val in enumerate(args):
            self.val[index].append(val)
            self.lines[index].set_xdata(self.i)
            self.lines[index].set_ydata(self.val[index])

        self.axes.set_xlim(0, max(self.i) * 1.5)
        self.axes.set_ylim(0, max(max(self.val)) * 1.5)

        plt.draw()
        plt.pause(1e-20)


    def init_plot(self, num_axes):
        """
        num_axes is the number of variables that should be plotted.
        """
        self.i = []
        self.val = []
        plt.ion()
        self.axes = plt.gca()
        self.lines =[]

        for i in range(num_axes):
            self.val.append([])
            self.lines.append([])
            self.lines[i], = self.axes.plot([], self.val[0], '-', c=[random.random() for _ in range(3)], linewidth=1.5, markersize=4)


def main():
    """
    Tests the code on a toy example.
    """
    def get_label(dp):
        if dp[0] == 1: return 2
        elif dp[2] == 1: return 1
        else: return 0

    from itertools import product
    x = np.array(list(product([0, 1], repeat=6)))

    #  Encoding of the correct classes for the training material
    y = np.array([get_label(dp) for dp in x])

    ind = np.arange(len(y))

    np.random.seed(524287)
    np.random.shuffle(ind)

    b = LogisticRegression()
    b.fit(x[ind][:-20], y[ind][:-20])
    b.classify_datapoints(x[ind][-20:], y[ind][-20:])



if __name__ == '__main__':
    main()