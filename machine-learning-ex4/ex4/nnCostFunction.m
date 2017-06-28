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



y_matrix = eye(num_labels)(y,:); %5000x10

%forward prop

%a1 equals the X input matrix with a column of 1's added (bias units) as the first column.
a1 = [ones(m,1) X]; %5000x401

%z2 equals the product of a1 and Θ1
z2 = a1*Theta1'; %5000x401 * 401x25 = 5000x25

%a2 is the result of passing z2 through g()
a2 = sigmoid(z2); %5000x25

%Then add a column of bias units to a2 (as the first column).
%NOTE: Be sure you DON'T add the bias units as a new row of Theta.
%use new variable?
a2 = [ones(size(a2),1) a2]; %5000x26

%z3 equals the product of a2 and Θ2
z3 = a2*Theta2'; %5000x26 * 26x10 = 5000x10

%a3 is the result of passing z3 through g()
a3 = sigmoid(z3); %5000x10

% J(θ)=  1/m(sigma)(sigma)[−y(i)log((hθ(x(i)))k)−(1−y(i))log(1−(hθ(x(i)))k)] ,
%      mkk i=1 k=1
%https://www.coursera.org/learn/machine-learning/discussions/all/threads/AzIrrO7wEeaV3gonaJwAFA

J = (1/m)*trace(-y_matrix'*log(a3) - (1-y_matrix)'*log(1-a3));
%1x1 * trace(5000x10' * 5000x10) - (1 - 5000x10)'*log(1 - 5000x10)

%regularization
% (λ/2m)* ( (sigma, j=1-25)*(sigma, k = 1-400)* Theta1^2 + ...
%   (sigma, j=1-10)*(sigma, k = 1-25)* Theta2^2) )

Theta1_reg = sum(sum(Theta1(:, 2:end).^2));
Theta2_reg = sum(sum(Theta2(:, 2:end).^2));

regularization = (lambda/(2*m)) * (Theta1_reg + Theta2_reg);

J = J + regularization;


%Backprop!

%d3 is the difference between a3 and the y_matrix. The dimensions are the same as both, (m x r).
d3 = a3 - y_matrix;

%z2 came from the forward propagation process - it's the product of a1 and Theta1, 
%prior to applying the sigmoid() function. Dimensions are (m x n) ⋅ (n x h) --> (m x h)

%d2 is tricky. It uses the (:,2:end) columns of Theta2. d2 is the product of d3 
%and Theta2(no bias), then element-wise scaled by sigmoid gradient of z2. 
%The size is (m x r) ⋅ (r x h) --> (m x h). The size is the same as z2, as must be.
d2 = (d3 * Theta2(:,2:end)) .* sigmoidGradient(z2);

%Delta1 is the product of d2 and a1. The size is (h x m) ⋅ (m x n) --> (h x n)
Delta1 = d2'*a1;

%Delta2 is the product of d3 and a2. The size is (r x m) ⋅ (m x [h+1]) --> (r x [h+1])
Delta2 = d3'*a2;

%Theta1_grad and Theta2_grad are the same size as their respective Deltas, just scaled by 1/m.
Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;

%So, set the first column of Theta1 and Theta2 to all-zeros. 
Theta1(:,1) = 0;
Theta2(:,1) = 0;

%Scale each Theta matrix by λ/m. Use enough parenthesis so the operation is correct.
Theta1 = (lambda/m)*Theta1;
Theta2 = (lambda/m)*Theta2;

%Add each of these modified-and-scaled Theta matrices to the 
%un-regularized Theta gradients that you computed earlier.
Theta1_grad = Theta1_grad + Theta1;
Theta2_grad = Theta2_grad + Theta2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
