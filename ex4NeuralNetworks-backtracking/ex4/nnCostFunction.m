function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

addi=ones(size(X,1),1);
a1 =[addi , X];  % 5000*401
z2 = a1*Theta1'; %5000*25
a2 = sigmoid(z2);

addi=ones(size(a2,1),1);
a2 = [addi,a2]; %5000*26
z3 = a2*Theta2'; %5000*10
a3 = sigmoid(z3);

logistic = a3;

y_modified = zeros(m,size(Theta2,1));

for i=1:m
  k=y(i,1);
  y_modified(i,k)=1;
endfor

for i=1:m
  y_row = y_modified(i,:);
  logistic_row = logistic(i,:);
  J = J + sum((y_row.*log(logistic_row)+(1.-y_row).*log(1-logistic_row)));
endfor

J = J * (-1) / m;
Theta1_modified = Theta1(:,2:end);
Theta2_modified = Theta2(:,2:end);
regularized_term = sum(sum(Theta1_modified.^2)) + sum(sum(Theta2_modified.^2));
regularized_term = regularized_term * lambda/ (2*m);

J = J + regularized_term;

k = size(Theta2,1);
for i=1:m
  y_row = y_modified(i,:);
  delta_3 = a3(i,:)-y_modified(i,:); % 1*10
  sigmoid_gradient_2 = sigmoidGradient(z2(i,:)); %1*25
  delta_2 = (Theta2' * delta_3')(2:end,:) .* sigmoid_gradient_2'; %25*1
  Theta2_grad = Theta2_grad + delta_3' * a2(i,:);
  Theta1_grad = Theta1_grad + delta_2 * a1(i,:);
endfor

Theta1_grad = Theta1_grad./m; % 25* 401
Theta2_grad = Theta2_grad ./m; % 10 * 26
  

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (Theta1(:,2:end) .*lambda ./m);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (Theta2(:,2:end) .*lambda ./m);










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
