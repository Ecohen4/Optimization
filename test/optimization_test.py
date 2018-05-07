import unittest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.gradient_descent import GradientDescent


class TestGradientDescent(unittest.TestCase):

    def test_gradient_descent_linear(self):
        print("--- LINEAR REGRESSION EXAMPLE ---")
        X = np.random.random((10, 2))
        true_betas = np.array([3, 4])
        y = X.dot(true_betas)

        linear = GradientDescent(X, y, model='linear')
        linear.fit(max_iter=10**5, lr=0.001)
        test_result = [round(val, 4) for val in linear.beta.squeeze()]
        print("Elliot  estimated parameters: {}".format(test_result))

        clf = LinearRegression()
        clf.fit(X, y)
        expected_result = [round(val, 4) for val in clf.coef_.squeeze()]
        print("sklearn estimated parameters: {}".format(expected_result))

        self.assertListEqual(test_result, expected_result)

    def test_gradient_descent_logistic(self):
        print("--- LOGISTIC REGRESSION EXAMPLE ---")
        X, y = make_classification(
            n_samples=1000,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_classes=2
            )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        logistic = GradientDescent(X_train, y_train, model='logistic')
        logistic.fit(max_iter=10**5, lr=0.001)
        test_betas = [round(val) for val in logistic.beta.squeeze()]
        accuracy = accuracy_score(y_true=y_test, y_pred=logistic.classify(X_test))
        print("Elliot  estimated parameters: {}".format(test_betas))
        print("Elliot  classification accuracy: {}".format(accuracy))

        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        expected_betas = [round(val) for val in clf.coef_.squeeze()]
        accuracy = accuracy_score(y_true=y_test, y_pred=clf.predict(X_test))
        print("sklearn estimated parameters: {}".format(expected_betas))
        print("sklearn classification accuracy: {}".format(accuracy))

        self.assertListEqual(test_betas, expected_betas)


if __name__ == '__main__':
    unittest.main()
