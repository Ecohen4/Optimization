import numpy as np


class GradientDescent(object):

    def __init__(self, X, y, model='linear', augment_design_matrix=False):
        """Estimate model parameters by convex optimization with gradient descent.

        Parameters
        ----------
        X: matrix of covariates
        y: vector of targets / labels
        model: linear or logi regression
        augment_design_matrix: add column vector of ones to design matrix
            to allow for intercept

        Returns
        -------
        Instantiated model object.
        """
        self.y = y
        self.X = self._augment_design_matrix(X) if augment_design_matrix else X
        self.has_intercept = augment_design_matrix
        self.model_type = model
        self.beta = self._random_guess()
        self._init_cost_functions()

    def _init_cost_functions(self):
        if self.model_type == 'linear':
            self.cost_funciton = self._rss_cost
            self.cost_gradient = self._rss_gradient
            self.predict = self._linear_predict
        elif self.model_type == 'logistic':
            self.cost_funciton = self._log_loss_cost
            self.cost_gradient = self._log_loss_gradient
            self.predict = self._logistic_predict
        else:
            raise ValueError(
                '''choose a linear model type:
                linear (regression) or logistic (classification)
                ''')

    def _augment_design_matrix(self, X):
        column_vector_of_ones = np.ones(len(X)).reshape(-1, 1)
        X_augmented = np.hstack((column_vector_of_ones, X))
        return X_augmented

    def _random_guess(self):
        return np.random.random(self.X.shape[1])

    def _linear_predict(self, X):
        y_hat = X.dot(self.beta)
        return y_hat

    def _rss_cost(self, X):
        y_hat = self._linear_predict(X)
        return (1/2) * np.sum((self.y - y_hat)**2)

    def _rss_gradient(self, X):
        y_hat = self._linear_predict(X)
        return -X.T.dot(self.y - y_hat)

    def _logistic_predict(self, X):
        '''the logistic function is defined as h(x) = 1/(1+exp(-x)).
        It is the inverse of the logit function.
        '''
        regression = X.dot(self.beta)
        p = 1 / (1 + np.exp(-regression))
        return p

    def _log_loss_cost(self, X, regularization=0.0):
        p = self._logistic_predict(X)
        cost_vector = -self.y * np.log(p) - (1 - self.y) * np.log(1 - p)
        ridge_penalty = regularization * np.sum(self.beta**2)
        if self.has_intercept:
            ridge_penalty -= self.beta[0]**2  # don't penalize intercept
        return cost_vector.sum() + ridge_penalty

    def _log_loss_gradient(self, X, regularization=0.0):
        p = self._logistic_predict(X)
        cost_gradient = -X.T.dot(self.y - p)
        ridge_gradient = regularization * (2 * self.beta)
        if self.has_intercept:
            ridge_gradient[0] = 0.0  # don't penalize the intercept
        return cost_gradient + ridge_gradient

    def _paramater_update(self, gradient, lr, verbose=False):
        self.beta -= lr * gradient
        if verbose:
            print("    updated parameters: {}".format(self.beta))

    def _stopping_criteria_satisfied(self, gradient, epsilon):
        if all(abs(gradient) < epsilon):
            print("    stopping criteria satified:")
            print("    gradient {} less than epsilon {}".format(
                gradient, epsilon)
                )
            return True
        else:
            return False

    def fit(self,
            model='logistic', max_iter=10000, lr=0.001, epsilon=0.0000001,
            ):
        converged = False
        for i in range(max_iter):
            gradient = self.cost_gradient(self.X)
            if self._stopping_criteria_satisfied(gradient, epsilon):
                print('    converged at {} iterations'.format(i))
                converged = True
                break
            else:
                self._paramater_update(gradient, lr)
        if not converged:
            print('stopping at {} iterations with estimated parameters {}'.
                  format(i, self.beta))

    def classify(self, X, threshold=0.5):
        if self.model_type == 'logistic':
            return self.predict(X) >= threshold
        else:
            raise ValueError(
                '''classification only available for logistic model type''')


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    def plot_results(observed, predicted, title='observed vs predicted'):
        results = pd.DataFrame({'observed': observed, 'predicted': predicted})
        sns.stripplot(x='observed', y='predicted', data=results, jitter=True)
        plt.title(title)
        plt.show()

    print("--- LINEAR REGRESSION EXAMPLE ---")
    X = np.random.random((10, 2))
    true_betas = np.array([3, 4])
    y = X.dot(true_betas)

    clf = LinearRegression()
    clf.fit(X, y)
    print("sklearn estimated parameters: {}".format(clf.coef_.squeeze()))

    linear = GradientDescent(X, y, model='linear')
    linear.fit(max_iter=10**5, lr=0.001)
    print("my estimated parameters: {}".format(linear.beta.squeeze()))

    print("--- LOGISTIC REGRESSION EXAMPLE ---")
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_classes=2
        )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    betas = clf.coef_.squeeze()
    accuracy = accuracy_score(y_true=y_test, y_pred=clf.predict(X_test))
    print("sklearn estimated parameters: {}".format(betas))
    print("sklearn classification accuracy: {}".format(accuracy))
    plot_results(observed=y_test, predicted=clf.predict_proba(X_test)[:, 1],
                 title='Sklearn')

    logistic = GradientDescent(X_train, y_train, model='logistic')
    logistic.fit(max_iter=10**5, lr=0.001)
    betas = logistic.beta.squeeze()
    accuracy = accuracy_score(y_true=y_test, y_pred=logistic.classify(X_test))
    print("my estimated parameters: {}".format(betas))
    print("my classification accuracy: {}".format(accuracy))
    plot_results(observed=y_test, predicted=logistic.predict(X_test),
                 title='Elliot')
