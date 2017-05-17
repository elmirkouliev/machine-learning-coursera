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
h = (X * theta) - y;
% Computer regularization, omitting first theta
regularization = sum(theta(2:end,:) .^ 2) * (lambda / (2*m));
J = sum(h.^2) /(2 * m) + regularization;


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Compute regularization ommiting first theta (by making it 0)
regularization = [zeros(1); theta(2:end,:)] * (lambda / m);
% Computer gradient against each feature in data set, add regularized value
grad = ((X' * h) / m) + regularization;

% =========================================================================

grad = grad(:);

end
