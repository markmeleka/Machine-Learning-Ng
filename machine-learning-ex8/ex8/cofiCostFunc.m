function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


pred_movie_ratings = X*Theta'; %5x3 * 4x3' = 5x4
movie_rating_error = pred_movie_ratings - Y; %5x4
error_factor = movie_rating_error .* R; %5x4 .* 5x4 = 5x4

J = (1/2)*sum(sumsq(error_factor));

X_grad = error_factor*Theta; %5x4 * 4x3 = 4x3
Theta_grad = error_factor'*X; %5x4 * 5x3 = 5x3

reg_X = (lambda/2) * sum(sumsq(X));
reg_theta = (lambda/2) * sum(sumsq(Theta));
J += reg_X + reg_theta;

X_grad += lambda*X;
Theta_grad += lambda*Theta;









% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
