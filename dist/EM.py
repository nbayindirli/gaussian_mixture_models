#!/usr/bin/python

#########################################################
# CSE 5523 starter code (HW#5)
# Alan Ritter
#########################################################

import random
import math
import sys
import re

# GLOBALS/Constants
VAR_INIT = 1


def log_exp_sum(x):
    # TODO: implement logExpSum
    pass


def read_true(filename='wine-true.data'):
    f = open(filename)
    labels = []
    split_re = re.compile(r"\s")
    for line in f:
        labels.append(int(split_re.split(line)[0]))
    return labels

#########################################################################
# Reads and manages data in appropriate format
#########################################################################


class Data:
    def __init__(self, filename):
        self.data = []
        f = open(filename)
        (self.nRows,self.nCols) = [int(x) for x in f.readline().split(" ")]
        for line in f:
            self.data.append([float(x) for x in line.split(" ")])

    # Computers the range of each column (returns a list of min-max tuples)
    def range(self):
        ranges = []
        for j in range(self.nCols):
            min = self.data[0][j]
            max = self.data[0][j]
            for i in range(1,self.nRows):
                if self.data[i][j] > max:
                    max = self.data[i][j]
                if self.data[i][j] < min:
                    min = self.data[i][j]
            ranges.append((min,max))
        return ranges

    def __getitem__(self,row):
        return self.data[row]

#########################################################################
# Computes EM on a given data set, using the specified number of clusters
# self.parameters is a tuple containing the mean and variance for each gaussian
#########################################################################


class EM:
    def __init__(self, data, n_clusters):
        # Initialize parameters randomly...
        random.seed()
        self.parameters = []
        self.priors = []        # Cluster priors
        self.nClusters = n_clusters
        self.data = data
        ranges = data.range()
        for i in range(n_clusters):
            p = []
            init_row = random.randint(0,data.nRows-1)
            for j in range(data.nCols):
                # Randomly initalize variance in range of data
                p.append((random.uniform(ranges[j][0], ranges[j][1]), VAR_INIT*(ranges[j][1] - ranges[j][0])))
            self.parameters.append(p)

        # Initialize priors uniformly
        for c in range(n_clusters):
            self.priors.append(1 / float(n_clusters))

    def log_likelihood(self, data):
        log_likelihood = 0.0
        # TODO: compute log-likelihood of the data
        return log_likelihood

    # Compute marginal distributions of hidden variables
    def e_step(self):
        # TODO: E-step
        pass

    # Update the parameter estimates
    def m_step(self):
        # TODO: M-step
        pass

    # Computes the probability that row was generated by cluster
    def log_prob(self, row, cluster, data):
        # TODO: compute probability row i was generated by cluster k
        pass

    def run(self, maxsteps=100, testData=None):
        # TODO: Implement EM algorithm
        train_likelihood = 0.0
        test_likelihood = 0.0
        return train_likelihood, test_likelihood

if __name__ == "__main__":
    d = Data('wine.train')
    if len(sys.argv) > 1:
        e = EM(d, int(sys.argv[1]))
    else:
        e = EM(d, 3)
    e.run(100)