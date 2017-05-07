function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);


disp(Theta1(1:2, 1:10))
disp(size(Theta1));

% disp(size(Theta2));

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);


% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add unit (ones) to the top of the input layer
inputLayer = [ones(m, 1) X];

disp(inputLayer(1:2, 1:10))

disp(X(1:2, 1:10))

disp(size(inputLayer))
disp(Theta1(1:2, 1:10))
disp(size(Theta1))
disp('Theta transpose')
disp(size(Theta1'))

z2 = sigmoid(inputLayer * Theta1');

disp('Z2')
disp(size(z2));

% add `1` node to top of the second layer
a2 = [ones(size(z2, 1),1) z2];

disp('A2')
disp(size(a2))

% Theta2 = Theta2(1,:);

disp(size(Theta2))

% Do I need to sigmoid again?
a3 = a2 * Theta2';

disp('Theta2');
disp(size(Theta2))
disp(size(a3))

[predictions, index] = max(a3, [], 2);

p = index;






% =========================================================================


end
