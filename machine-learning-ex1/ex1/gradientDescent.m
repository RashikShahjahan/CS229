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
    sum2=0;
    sum0=0;
    for i=1:m
       h2(i) = theta(1)+theta(2)*X(i,2);
       sum2 = sum2+(h2(i)-y(i))*X(i,2);
       h(i) = theta(1)+theta(2)*X(i,2);
       sum0 = sum0+(h(i)-y(i));
    end
    

    temp0(iter)=theta(1)-(alpha/m)*sum0;
    temp1(iter)=theta(2)-(alpha/m)*sum2;
    theta(1)=temp0(iter);
    theta(2)=temp1(iter);
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
