function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % THIS DOESN'T WORK BECAUSE THETA IS A VECTOR
%theta(1,1) = theta(1,1) - sum(alpha*(1/m)*(X*theta-y));
%theta(2,1) = theta(2,1) - alpha*(1/m)*(X'*(X*theta-y));   

h = X*theta; % m*n * n*1
err_vector = h - y; % m*1 - m*1
theta_change = alpha*(1/m)*(X'*err_vector); % k * 1/m * sum(n*m * m*1)
theta = theta - theta_change;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
