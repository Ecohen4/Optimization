#!/usr/bin/env python
"""
This is an implementation of two-component Gaussian example from

Elements of Statistical Learning (pp 272)

"""


## make imports
from __future__ import division
import sys
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats

class TwoComponentGaussian():

    def __init__(self, y, num_iters=25, num_runs=20, verbose=False):
        """
        constructor
        """
        self.y = y
        self.verbose = verbose
        self.num_runs = num_runs
        self.num_iters = num_iters
        self.params = self._guess_initial_parameters()
        self.gaussian1 = None
        self.gaussian2 = None
        self.log_likelihood = None
        self.gamma_hat = np.zeros((self.params['n']), 'float') ## allocate memory for the responsibilities

    def _guess_initial_parameters(self):
        """
        make intial random guesses for the parameters
        """
        n    = len(self.y)
        mu1  = self.y[np.random.randint(0,n)]
        mu2  = self.y[np.random.randint(0,n)]
        var1 = np.random.uniform(0.5,1.5)
        var2 = np.random.uniform(0.5,1.5)
        pi   = 0.5
        return {'n':n, 'mu1':mu1, 'mu2':mu2, 'var1':var1, 'var2':var2, 'pi':pi}

    def _update_gaussian_distributions(self):
        self.gaussian1 = stats.norm(
            loc=self.params['mu1'],
            scale=np.sqrt(self.params['var1'])
            )
        self.gaussian2 = stats.norm(
            loc=self.params['mu2'],
            scale=np.sqrt(self.params['var2'])
            )

    def _update_expectation(self):
        """
        expectation step

        ARGS
        self.y: expectation is performed with respect to these data
        self.params: the most recent dictionary of parameters

        OUTPUT
        self.gamma_hat: the responsibilities
        """
        ## use the normal pdf to calculate the responsibilities
        self._update_gaussian_distributions()
        gamma_hat = (
        (self.params['pi'] * self.gaussian2.pdf(self.y)) / (
            ((1 - self.params['pi']) * self.gaussian1.pdf(self.y)) +
            (self.params['pi'] * self.gaussian2.pdf(self.y))
            )
        )
        self.gamma_hat = gamma_hat

    def _update_parameters(self):
        """
        maximization step
        ARGS
        params: the dictionary of parameters
        self.y: the data we are looking to maximize over
        self.gamma_hat: the most recently estimated responsibilities
        OUTPUT
        params: an updated dictionary of the parameters
        """
        mu_hat1 = np.sum((1-self.gamma_hat) * self.y) / np.sum(1-self.gamma_hat)
        mu_hat2 = np.sum(self.gamma_hat * self.y) / np.sum(self.gamma_hat)
        var_hat1 = np.sum((1 - self.gamma_hat) * (self.y - mu_hat1)**2) / np.sum(1 - self.gamma_hat)
        var_hat2 = np.sum(self.gamma_hat * (self.y - mu_hat2)**2) / np.sum(self.gamma_hat)
        pi_hat = np.sum(self.gamma_hat) / len(self.gamma_hat)
        self.params.update(
        {'mu1': mu_hat1, 'mu2':mu_hat2, 'var1': var_hat1, 'var2': var_hat2, 'pi': pi_hat}
        )

    def _update_log_likelihood(self):
        """
        likelihood
        returns a single value
        the likelihood function has two parts and the output is a sum of the two parts
        """
        ## using the normal pdf calculate the responsibilities
        self._update_gaussian_distributions()
        part1 = np.sum(
        (1 - self.gamma_hat) * np.log(self.gaussian1.pdf(self.y)) +
        (self.gamma_hat * np.log(self.gaussian2.pdf(self.y)))
        )
        part2 = np.sum(
        (1 - self.gamma_hat) * np.log(1 - self.params['pi']) +
        (self.gamma_hat * np.log(self.params['pi']))
        )
        self.log_likelihood = part1 + part2

    def run_em_algorithm(self, verbose=True):
        """
        main algorithm
        """

        maximum_likelihood = -np.inf
        best_estimates = None

        ## loop through the total number of runs
        for j in range(self.num_runs):
            iter_count = 0

            ## iterate between E-step and M-step
            while iter_count < self.num_iters:
                iter_count += 1

                ## ensure we have reasonable estimates
                if (self.params['var1'] < 0.0) or (self.params['var2'] < 0.0):
                    iter_count = 1
                    self._guess_initial_parameters()

                ## E-step
                self._update_expectation()
                self._update_log_likelihood()

                ## M-step
                self._update_parameters()

            if self.log_likelihood > maximum_likelihood:
                maximum_likelihood = self.log_likelihood.copy()
                best_estimates = self.params.copy()

            if self.verbose == True:
                print('run: {run} iteration {iter} --- mu1: {mu1} --- mu2: {mu2} \
                --- observed data likelihood: {likelihood}'.format(
                    run=j+1,
                    iter=iter_count,
                    mu1=round(self.params['mu1'],2),
                    mu2=round(self.params['mu2'],2),
                    likelihood=round(self.log_likelihood,4)
                    )
                )

        print("{n} runs with {m} iterations each, complete".format(
        n=self.num_runs, m=self.num_iters)
        )
        print('maximum likelihood: {}'.format(maximum_likelihood))
        print('best parameter estimates: {}'.format(best_estimates))
        self.plot_mixture_model(iteration=iter_count)
        return maximum_likelihood, best_estimates

    def plot_mixture_model(self, iteration):
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        x = self.y.copy()
        ax.hist(x, bins=25, density=True, alpha=0.6, fc='lightblue', histtype='stepfilled')
        xmin, xmax = ax.get_xlim()
        pdf_range = np.linspace(xmin, xmax, x.size)
        ax.plot(pdf_range, self.gaussian1.pdf(pdf_range),'darkblue', alpha=iteration/self.num_iters, label='pdf')
        ax.plot(pdf_range, self.gaussian2.pdf(pdf_range),'darkblue', alpha=iteration/self.num_iters, label='pdf')
        ax.set_xlabel("wait times (minutes)")
        plt.show()


if __name__ == '__main__':

    import pandas as pd
    import numpy as np

    # example dataset 1
    # y1 = np.array([-0.39,0.12,0.94,1.67,1.76,2.44,3.72,4.28,4.92,5.53])
    # y2 = np.array([ 0.06,0.48,1.01,1.68,1.80,3.25,4.12,4.60,5.28,6.22])
    # y  = np.hstack((y1,y2))

    # example dataset 2
    data_endpoint = "https://raw.githubusercontent.com/barneygovan/from-data-with-love/master/data/faithful.csv"
    data = pd.read_csv(data_endpoint)
    y = data[' waiting'].values

    mm = TwoComponentGaussian(y, num_iters=20, num_runs=10, verbose=True)
    mm.run_em_algorithm()
