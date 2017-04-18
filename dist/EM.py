#!/usr/bin/python

#########################################################
# CSE 5523 - Homework 5
# Noah Bayindirli.1
#########################################################

import numpy as np
import random
import math
import sys
import re

# GLOBALS/Constants
VAR_INIT = 1


def log_exp_sum(x):
    x_max = max(x)
    x_sum = 0
    for i in range(len(x)):
        x_sum += math.exp(x[i] - x_max)

    return x_max + math.log(x_sum)


def norm_pdf(x, mean, var):
    denom = (2 * math.pi * var)**0.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))

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
        self.nClusters = n_clusters
        self.data = data
        ranges = data.range()
        for i in range(n_clusters):
            p = []
            init_row = random.randint(0, data.nRows-1)
            for j in range(data.nCols):
                # Randomly initialize variance in range of data
                p.append((random.uniform(ranges[j][0], ranges[j][1]), VAR_INIT*(ranges[j][1] - ranges[j][0])))
            self.parameters.append(p) # mean = self.parameters[0], variance = self.parameters[1]

        # Initialize priors uniformly
        for c in range(n_clusters):
            self.priors.append(1 / float(n_clusters))

    def log_likelihood(self, data):
        # TODO: compute log-likelihood of the data
        log_likelihood = 0.0
        nr = data.nRows
        kc = self.nClusters

        for n in range(nr):
            log_cluster = np.zeros(kc)
            for k in range(kc):
                log_cluster[k] = math.log(norm_pdf(data[n, :], self.parameters[0], self.parameters[1]))
                log_cluster[k] += math.log(self.priors[k])

            # add probabilities
            log_cluster_sum = log_exp_sum(log_cluster)
            # add logs
            log_likelihood += log_cluster_sum

        return log_likelihood

    # Compute marginal distributions of hidden variables
    def e_step(self):
        # TODO: E-step

        log_pri = np.zeros(self.data.nRows, self.nClusters)
        log_post = np.zeros(self.data.nRows, self.nClusters)

        # comp all log(p(x|j)*p(j))
        for n in range(self.data.nRows):
            for k in range(self.nClusters):
                log_pri[n, k] = math.log(norm_pdf(self.data[n, :], self.parameters[0], self.parameters[1]))
                log_pri[n, k] += math.log(self.priors[k])  # addition b/c log

        log_sum = log_exp_sum(log_pri)

        # now log posteriors
        for n in range(self.data.nRows):
            for k in range(self.nClusters):
                log_post[n, k] = log_pri[n, k] - log_sum[n]

        return log_post

    # Update the parameter estimates
    def m_step(self, log_post):
        # TODO: M-step
        nm = log_exp_sum(log_post)

        # update mean
        for k in range(self.nClusters):
            for d in range(self.data.nCols):
                sum = 0
                for n in range(self.data.nRows):
                    sum += math.exp(log_post[n, k] - nm[k]) * self.data[n, d]
                self.parameters[0] = sum

        print 'Mean: ', self.parameters[0]

        # update variance
        for k in range(self.nClusters):
            sum = 0
            for n in range(self.data.nRows):                # no need for transpose here ???
                sum += math.exp(log_post[n, k] - nm[k] * (self.data[n, :] - self.parameters[0]))
            self.parameters[1] = sum

        print 'Variance: ', self.parameters[1]

        # update priors
        for k in range(self.nClusters):
            self.priors[k] = math.exp(nm[k]) / self.data.nRows

        print 'Prior: ', self.priors

        return

    # Computes the probability that row was generated by cluster
    def log_prob(self, row, cluster, data):                         # how to utilize ???
        # TODO: compute probability row i was generated by cluster k
        prob = 0

        prob = log()

        return prob

    def run(self, maxsteps=100, testData=None):
        # TODO: Implement EM algorithm
        train_likelihood = 0.0
        test_likelihood = 0.0

        old_log_likelihood = self.log_likelihood(self.data)
        num_iters = 1

        print 'Old log likelihood: ', old_log_likelihood

        for i in range(maxsteps):
            # E-step
            posterior_log_prob = self.e_step()
            print 'Posterior log probability: ', posterior_log_prob

            # M-step
            self.m_step(posterior_log_prob)

            new_log_likelihood = self.log_likelihood(self.data) # (self.data) here ???
            print 'New log likelihood', new_log_likelihood

            log_likelihood_diff = new_log_likelihood - old_log_likelihood
            print 'Log likelihood difference: ', log_likelihood_diff
            print 'Iteration ', num_iters

            if abs(log_likelihood_diff) < 0.001 or num_iters > maxsteps:
                print 'CONVERGED'
                print 'Final log likelihood difference: ', abs(log_likelihood_diff)
                print 'Total iterations: ', num_iters

                return train_likelihood, test_likelihood
            else:
                num_iters += 1

        return train_likelihood, test_likelihood

if __name__ == "__main__":
    d = Data('wine.train')
    if len(sys.argv) > 1:
        e = EM(d, int(sys.argv[1]))
    else:
        e = EM(d, 3)
    e.run(100)
