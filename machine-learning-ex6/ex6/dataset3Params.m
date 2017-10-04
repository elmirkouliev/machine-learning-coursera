function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Should match nubmer of elements in the values we're testing against
steps = 8;
cvError = zeros(steps^2,1);
cValues = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigmaValues = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

for i = 1:steps
    for j = 1:steps
         % Train the SVM against training data
        model = svmTrain(X, y, cValues(i), @(x1, x2) gaussianKernel(x1, x2, sigmaValues(j)));
        % Predict off of trained model for training data
        cvPredictions = svmPredict(model, Xval);
        % Measure error by the number of wrong predictions
        cvError(((i-1) * steps) + j) = sum(cvPredictions ~= yval);
    end
end

% Find index of lowest error from CV error set
index = find(cvError==min(cvError),1);

% Return with respect to indices of cValues & sigmaValues
C = cValues(ceil(index / steps));
sigma = sigmaValues(index - (floor(index/steps) * steps));

% =========================================================================

end
