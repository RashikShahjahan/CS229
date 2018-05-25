function [J1, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n= length(theta);
% You need to return the following variables correctly 
J1=0;
J0= 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



grad0=0;
grad1=zeros(size(theta));

theta0=0;

for i=1:m
    
    h = sigmoid(X(i,:)*theta);
    J0=J0+(-y(i)*log(h)-(1-y(i))*log(1-h));
    grad0=grad0+(h-y(i))*(X(i,:));
  
end

    theta(1)=0;
for j=1:n-1
    theta0=theta0+theta(j+1).^2;
   
end


for j=1:n-1
    grad1=grad1+(lambda)*theta(j);

end

grad=(grad1+grad0)/m;

J1=J0/m+theta0*lambda/(2*m);


% =============================================================

end
