function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%disp('X');
%disp(X);
%disp('theta');
%disp(theta);

costAtPoints = X * theta;

%disp('Costs');
%disp(costAtPoints);

%disp('Y');
%disp(y);

sqrErrors  = (costAtPoints - y).^2; % squared errors

%disp('Errors');
%disp(sqrErrors);

%Add them all together and devide over total results
J = sum(sqrErrors) * 1/(2 * m);

% =========================================================================

end
