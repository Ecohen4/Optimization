# Optimization for Machine Learning
Estimate machine-learning model parameters with numerical optimization.

In a handful of special cases, model parameters in machine learning can be found exactly. For example, linear regression coefficients that minimize sum-of-square-errors can be found by the normal equations. But what if we recast linear regression as a classification problem with the logit link function? Or what if we add a regularization term? All of a sudden there is no closed-form solution to estimating model parameters. Instead, we must use numerical optimization.

This repository demonstrates the use of two optimization techniques --gradient descent and expectation maximization--for estimating model parameters where there is no exact solution. Global minima are guaranteed only in the case of a convex cost function (as is the case for least-squares regression without regularization). For more complex cost functions, we simply try to find the "best" local minima given finite time and computational resources.
