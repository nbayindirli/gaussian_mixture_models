#!/usr/bin/python

#########################################################
# CSE 5523 - Homework 5
# Noah Bayindirli.1
#########################################################

import matplotlib.pyplot as plt
import numpy as np
import random
import math
import sys
import re

# GLOBALS/Constants
VAR_INIT = 1


def log_exp_sum(x):
    # Retrieve largest x
    x_max = 0
    for i in range(len(x)):
        if x[i] > x_max:
            x_max = x[i]

    x_sum = 0
    for i in range(len(x)):
        x_sum += math.exp(x[i] - x_max)

    return x_max + math.log(x_sum)


# Helper function for gaussian processes
def norm_pdf(x, mean, var):
    num = math.exp(-1 * (x - mean) * (x - mean) / (2 * var))
    denom = math.sqrt(2*math.pi*var)

    return num / denom


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
        (self.nRows, self.nCols) = [int(x) for x in f.readline().split(" ")]
        for line in f:
            self.data.append([float(x) for x in line.split(" ")])

    # Computes the range of each column (returns a list of min-max tuples)
    def range(self):
        ranges = []
        for j in range(self.nCols):
            min = self.data[0][j]
            max = self.data[0][j]
            for i in range(1, self.nRows):
                if self.data[i][j] > max:
                    max = self.data[i][j]
                if self.data[i][j] < min:
                    min = self.data[i][j]
            ranges.append((min, max))
        return ranges

    def __getitem__(self, row):
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
        self.w = []             # Membership weights
        self.nClusters = n_clusters
        self.data = data
        ranges = data.range()
        for i in range(n_clusters):
            p = []
            init_row = random.randint(0, data.nRows-1)
            for j in range(data.nCols):
                # Randomly initialize variance in range of data
                p.append((random.uniform(ranges[j][0], ranges[j][1]), VAR_INIT*(ranges[j][1] - ranges[j][0])))
            self.parameters.append(p)

        # Initialize priors uniformly
        for c in range(n_clusters):
            self.priors.append(1 / float(n_clusters))

    def log_likelihood(self, data):
        log_likelihood = 0.0
        num_observations = data.nRows
        num_clusters = self.nClusters
        num_dimensions = data.nCols

        # Compute value of log likelihood
        for n in range(num_observations):
            log_cluster = np.zeros(num_clusters)
            for k in range(num_clusters):
                for d in range(num_dimensions):
                    log_cluster[k] = math.log(norm_pdf(x=data[n][d], mean=self.parameters[k][d][0], var=self.parameters[k][d][1]))
                    log_cluster[k] += math.log(self.priors[k])

            # Add probabilities
            log_sum_cluster = log_exp_sum(log_cluster)
            # Add logarithms
            log_likelihood += log_sum_cluster

        return log_likelihood

    # Compute marginal distributions of hidden variables
    def e_step(self):
        num_clusters = self.nClusters
        num_observations = self.data.nRows
        # Set responsibility matrix for membership weights
        self.w = np.zeros((num_observations, num_clusters))

        # Compute all log(p(x|j)*p(j))
        for n in range(num_observations):
            for k in range(num_clusters):
                self.w[n, k] = self.log_prob(row=n, cluster=k, data=self.data)
            self.w[n, :] = self.w[n, :]/sum(self.w[n, :])

        # Re-estimate priors
        Nk = np.zeros(num_clusters)
        for k in range(num_clusters):
            Nk[k] = sum(self.w[:, k])
        Nk /= sum(Nk)

        self.priors = Nk
        for k in range(num_clusters):
            self.priors[k] = Nk[k]

        return

    # Update the parameter estimates
    def m_step(self):
        num_clusters = self.nClusters
        num_observations = self.data.nRows
        num_dimensions = self.data.nCols

        # Workaround for immutable tuple
        mean_sum = np.zeros((num_clusters, num_dimensions))
        var_sum = np.zeros((num_clusters, num_dimensions))

        # Re-estimate mean and variance
        for k in range(num_clusters):
            for d in range(num_dimensions):
                mean_sum[k, d] = 0
                var_sum[k, d] = 0
                for n in range(num_observations):
                    mean_sum[k, d] += self.data[n][d] * self.w[n][k]
                    var_sum[k, d] += (self.data[n][d] - mean_sum[k, d]) * (self.data[n][d] - mean_sum[k, d]) * self.w[n][k]
                mean_sum[k, d] /= (self.priors[k] * num_observations)
                var_sum[k, d] /= (self.priors[k] * num_observations)

        # Insert back into tuple
        self.parameters = []
        for k in range(num_clusters):
            p = []
            for d in range(num_dimensions):
                p.append((mean_sum[k, d], var_sum[k, d]))
            self.parameters.append(p)

        return

    # Computes the probability that row was generated by cluster
    def log_prob(self, row, cluster, data):
        # TODO: compute probability row i was generated by cluster k
        log_prob = 0
        row_data = self.data[row]
        N = len(row_data)

        for n in range(N):
            log_prob += math.log(norm_pdf(x=row_data[n], mean=self.parameters[cluster][n][0], var=self.parameters[cluster][n][1]))

        return log_prob

    def run(self, max_steps=100, test_data=None):
        train_likelihood = []
        iterations = []
        test_likelihood = []
        converged = False

        # Initial value of log likelihood
        old_log_likelihood = 0
        num_iters = 1

        while num_iters < max_steps and not converged:
            # E-step
            self.e_step()

            # M-step
            self.m_step()

            new_log_likelihood = self.log_likelihood(self.data)

            log_likelihood_diff = old_log_likelihood - new_log_likelihood

            if abs(log_likelihood_diff) < 0.001:
                converged = True

                print 'CONVERGED!'
                print 'Final log likelihood:', new_log_likelihood
                print 'Total iterations:', num_iters

            else:
                print 'Log likelihood =', new_log_likelihood, '@ iteration', num_iters, '\n'
                old_log_likelihood = new_log_likelihood

            train_likelihood.append(new_log_likelihood)
            iterations.append(num_iters)
            num_iters += 1

        return train_likelihood, test_likelihood, iterations

if __name__ == "__main__":
    d = Data('wine.train')
    d_test = Data('wine.test')

    if len(sys.argv) > 1:
        e = EM(d_test, int(sys.argv[1]))
    else:
        e = EM(d_test, 3)
    (train_likelihood, test_likelihood, iterations) = e.run(100)

    labels = read_true(filename='wine-true.data')

    plt.plot(iterations, train_likelihood)
    plt.ylabel("Training Log Likelihood")
    plt.xlabel("Iterations")
    plt.interactive(False)
    plt.show()

