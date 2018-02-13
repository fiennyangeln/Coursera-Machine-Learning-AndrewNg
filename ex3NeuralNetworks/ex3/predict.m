function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

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
a_o = X;
X_o = ones(m,1);
% 5000 * 401
a_o= [X_o a_o];
% 5000 * 25
z1 = a_o*Theta1';
a_1 = sigmoid(z1);
% 5000 * 26
a_1 = [X_o a_1];
% 5000 * 10
z2= a_1*Theta2';
a_2 = sigmoid (z2);
[value,index] = max(a_2,[],2);
p=index;







% =========================================================================


end
