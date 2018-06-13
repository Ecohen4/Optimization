#!/usr/bin/env python
"""
Implementation of a two-component Gaussian mixture-model
for the classic 'Old Faithful' dataset.
The EM-algorithm is adopted from Elements of Statistical Learning
(Hastie, Tibshirani and Friedman, pp 272)
"""


import sys
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats


class TwoComponentGaussian():

    def __init__(self, y, num_iters=25, num_runs=20, verbose=False):
        """Constructor
        """
        self.y = y
        self.verbose = verbose
        self.num_runs = num_runs
        self.num_iters = num_iters
        self.gaussian1 = None
        self.gaussian2 = None
        self.log_likelihood = None
        self.params = self._guess_initial_parameters()
        self.gamma_hat = self._allocate_memory_for_responsibilities()

    def _guess_initial_parameters(self):
        """Make intial random guesses of the model parameters.
        Assume two gaussian distributions, each defined by a mean and variance.
        """
        n = len(self.y)
        mu1 = self.y[np.random.randint(1, n)]
        mu2 = self.y[np.random.randint(1, n)]
        var1 = np.random.uniform(1, np.log2(n))
        var2 = np.random.uniform(1, np.log2(n))
        pi = 0.5
        initial_params = {
        'n': n, 'mu1': mu1, 'mu2': mu2, 'var1': var1, 'var2': var2, 'pi': pi
        }
        return initial_params

    def _allocate_memory_for_responsibilities(self):
        return np.zeros((self.params['n']), 'float')

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
        """Expectation step.

        Paramaters
        ----------
        self.y: expectation is performed with respect to the target data
        self.params: most recent dictionary of mixture-model parameters

        Returns (updates)
        -------
        self.gamma_hat: the responsibilities

        """
        #  use the normal pdf to calculate the responsibilities
        self._update_gaussian_distributions()
        gamma_hat = (
            (self.params['pi'] * self.gaussian2.pdf(self.y)) / (
                ((1 - self.params['pi']) * self.gaussian1.pdf(self.y)) +
                (self.params['pi'] * self.gaussian2.pdf(self.y))
            )
        )
        self.gamma_hat = gamma_hat

    def _update_parameters(self):
        """Maximization step.

        Paramaters
        ----------
        params: dictionary of mixture-model parameters
        self.y: data we are maximizing over
        self.gamma_hat: most recent estimated responsibilities

        Returns (updates)
        -------
        params: an updated dictionary of mixture-model parameters

        """
        mu_hat1 = (
            (np.sum((1 - self.gamma_hat) * self.y)) / np.sum(1 - self.gamma_hat)
            )
        mu_hat2 = (
            np.sum(self.gamma_hat * self.y) / np.sum(self.gamma_hat)
            )
        var_hat1 = (
            np.sum((1 - self.gamma_hat) * (self.y - mu_hat1)**2) /
                np.sum(1 - self.gamma_hat)
            )
        var_hat2 = (
            np.sum(self.gamma_hat * (self.y - mu_hat2)**2) /
                np.sum(self.gamma_hat)
            )
        pi_hat = np.sum(self.gamma_hat) / len(self.gamma_hat)
        self.params.update({'mu1': mu_hat1, 'mu2': mu_hat2,
                            'var1': var_hat1, 'var2': var_hat2,
                            'pi': pi_hat}
                           )

    def _update_log_likelihood(self):
        """Likelihood estimation.

        Paramaters
        ----------
        guassian1: data-generating process 1
        gaussian2: data-generating process 2

        Returns (updates)
        -------
        a scalar representing the sum of the log-likelihoods

        """
        # use the normal pdf to calculate the responsibilities
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

    def fit(self, verbose=True):
        """The EM algorithm for a two-component gaussian mixture model.
        """

        maximum_likelihood = -np.inf
        best_estimates = None

        # loop through the total number of runs
        for j in range(self.num_runs):
            iter_count = 0

            # iterate between E-step and M-step
            while iter_count < self.num_iters:
                iter_count += 1

                # ensure we have reasonable estimates
                if (self.params['var1'] < 0.0) or (self.params['var2'] < 0.0):
                    iter_count = 1
                    self._guess_initial_parameters()

                # E-step
                self._update_expectation()
                self._update_log_likelihood()

                # M-step
                self._update_parameters()

            if self.log_likelihood > maximum_likelihood:
                maximum_likelihood = self.log_likelihood.copy()
                best_estimates = self.params.copy()

            if self.verbose is True:
                print('run: {run} iteration {iter} --- mu1: {mu1} --- mu2: {mu2} \
                --- observed data likelihood: {likelihood}'.format(
                    run=j+1,
                    iter=iter_count,
                    mu1=round(self.params['mu1'], 2),
                    mu2=round(self.params['mu2'], 2),
                    likelihood=round(self.log_likelihood, 4)
                    )
                )

        print("{n} runs with {m} iterations each, complete".format(
                n=self.num_runs, m=self.num_iters)
              )
        print('maximum likelihood: {}'.format(maximum_likelihood))
        print('best parameter estimates: {}'.format(best_estimates))
        return maximum_likelihood, best_estimates

    def plot_mixture_model(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        x = self.y.copy()
        ax.hist(x, bins=25, density=True, alpha=0.6, fc='lightblue',
                histtype='stepfilled')
        xmin, xmax = ax.get_xlim()
        pdf_range = np.linspace(xmin, xmax, x.size)
        ax.plot(pdf_range, self.gaussian1.pdf(pdf_range), 'darkblue', alpha=1,
                label='pdf')
        ax.plot(pdf_range, self.gaussian2.pdf(pdf_range), 'darkblue', alpha=1,
                label='pdf')
        ax.set_xlabel("wait times (minutes)")
        plt.show()


if __name__ == '__main__':

    import pandas as pd
    import numpy as np

    # # example from Hastie, Tibshirani and Friedman (p 272)
    # y1 = np.array([-0.39,0.12,0.94,1.67,1.76,2.44,3.72,4.28,4.92,5.53])
    # y2 = np.array([ 0.06,0.48,1.01,1.68,1.80,3.25,4.12,4.60,5.28,6.22])
    # y  = np.hstack((y1,y2))

    # example with Old Faithful geyser eruption data
    endpoint = "https://raw.githubusercontent.com/barneygovan/from-data-with-love/master/data/faithful.csv"
    data = pd.read_csv(endpoint)
    y = data[' waiting'].values

    mm = TwoComponentGaussian(y, num_iters=20, num_runs=10, verbose=True)
    mm.fit()
    mm.plot_mixture_model()
