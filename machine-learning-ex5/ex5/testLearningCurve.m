% Testing set learning curve

% Load from ex5data1: 
% You will have X, y, Xval, yval, Xtest, ytest in your environment
load ('ex5data1.mat');

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];    

% Map test values to higher order polynomials
X_test_val = polyFeatures(Xtest, p);
X_test_val = bsxfun(@minus, X_test_val, mu);
X_test_val = bsxfun(@rdivide, X_test_val, sigma);
X_test_val = [ones(size(X_test_val, 1), 1), X_test_val];           % Add Ones

% Set learned lambda
lambda = 3;

[error_train, error_test] = learningCurve(X_poly, y, X_test_val, ytest, lambda);

plot(1:m, error_train, 1:m, error_test);

figure(1);

title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Test set')

