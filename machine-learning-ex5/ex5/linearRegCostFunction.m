function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% X = 12x1, y = 12x1, theta = 2x1
h = X*theta;
error = h-y;
error_sq = error.^2;
sum_sq_err = sum(error_sq);

grad = (1/m)*(X'*error);

theta(1) = 0;
cost_reg_term = (lambda/(2*m))*sum(theta.^2);

J = (1/(2*m))* sum_sq_err;

J = J + cost_reg_term;

grad_reg_term = (lambda/m)*theta;
grad = grad + grad_reg_term;









% =========================================================================

grad = grad(:);

end
