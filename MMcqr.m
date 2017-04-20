function [beta, alpha] = MMcqr(taus, qs, y, x, gfunc, sfunc, toler, wwt, hwt, beta, alpha) 
%This function returns the Newton updates for theta = (beta, alpha)', 
%provided $q_1, \ldots, q_K$ are known. 
%INPUTS:
%   --- taus is the vector of K quantile positions.
%   --- qs is the current value of $q_1, \ldots, q_K$.
%   --- y is the response vector (n-by-1).
%   --- x is the predictor matrix (n-by-m).
%   --- gfunc is a character array containing the name of a function that
%       evaluates g(x, beta). It returns an n-by-1 vector.
%       There must also be a second MATLAB function, with the same name
%       as `gfunc` except with the letter 'd' appended to the front that
%       also takes (x, beta) as input but returns the n-by-p matrix of
%       partial derivatives of g(x, beta) evaluated at beta.
%   --- sfunc is a character array containing the name of a function that
%       evaluates s(x, alpha). It returns an n-by-1 vector.
%       There must also be a second MATLAB function, with the same name
%       as `sfunc` except with the letter 'd' appended to the front that
%       also takes (x, alpha) as input but returns the n-by-q matrix of
%       partial derivatives of s(x, beta) evaluated at alpha.
%   --- toler is a small number giving the minimum change in the value of
%       surrogate function before convergence is declared.
%   --- wwt is the K-by-1 weights vector corresponding to quantile
%       positions.
%   --- hwt is the n-by-1 weights vector corresponding to observations.
%   --- beta is the starting value of beta (p-by-1 vector).
%   --- alpha is the starting value of alpha (q-by-1 vector).
dgfunc = ['d', gfunc]; % name of the gradient of gfunc.
dsfunc = ['d', sfunc]; % name of the gradient of sfunc.
iteration = 0;
change = realmax;
n = length(y);
tn = toler/n; e0 = -tn/log(tn); epsilon = (e0 - tn)/(1 + log(e0));
% epsilon is the smoothing parameter.

p = length(beta);
q = length(alpha);
K = length(taus);

resmat = zeros(n, K);
tausmat = repmat(taus', n, 1);
wwtmat = repmat(wwt', n, 1); 
hwtmat = repmat(hwt, 1, K); 
weightsmat = 1 ./ (epsilon + abs(resmat));
% Expand the vector of weights to matrices for later calculation.

for k = 1:K
    resmat(:, k) = y - feval(gfunc, x, beta) - qs(k) * feval(sfunc, x, alpha);
end

lastsurr = sum(sum(wwtmat ./ hwtmat .* resmat.^2 .* weightsmat)) + ...
           sum(sum(wwtmat ./ hwtmat .* (4 * tausmat - 2) .* resmat));
       
while change > toler
    iteration = iteration + 1;
    Jbeta = feval(dgfunc, x, beta); % This is an n-by-p matrix.
    Jalpha = feval(dsfunc, x, alpha); % This is an n-by-q matrix.
    
    % Form the Hessian matrix:
    Hbetasq = zeros(p, p);
    for i = 1:n 
        for k = 1:K
            Hbetasq = Hbetasq + wwt(k) / hwt(i) * weightsmat(i, k) * ...
                Jbeta(i, :)' * Jbeta(i, :);
        end
    end
    
    Halphasq = zeros(q, q);
    for i = 1:n
        for k = 1:K
            Halphasq = Halphasq + wwt(k) * qs(k)^2 / hwt(i)* weightsmat(i, k) * ...
                Jalpha(i, :)' * Jalpha(i, :);
        end
    end
    
    Hbetaalpha = zeros(p, q);
    for i = 1:n
        for k = 1:K
            Hbetaalpha = Hbetaalpha + wwt(k) * qs(k) / hwt(i) * weightsmat(i, k) * ...
                Jbeta(i, :)' * Jalpha(i, :);
        end
    end
    
    Dbeta = zeros(p, 1);
    for i = 1:n
        for k = 1:K
            Dbeta = Dbeta + wwt(k) / hwt(i) * ...
                (resmat(i, k) * weightsmat(i, k) + 2 * taus(k) - 1) * ...
                Jbeta(i, :)';
        end
    end
    
    Dalpha = zeros(q, 1);
    for i = 1:n
        for k = 1:K
            Dalpha = Dalpha + wwt(k) / hwt(i) * qs(k) * ...
                (resmat(i, k) * weightsmat(i, k) + 2 * taus(k) - 1) * ...
                Jalpha(i, :)';
        end
    end
    
    % Form the (p + q)-by-(p + q) Hessian matrix:
    Hessiantemp = [Hbetasq Hbetaalpha; Hbetaalpha' Halphasq];
    % Form the (p + q)-by-1 negative gradient vector:
    NGradtemp = [Dbeta; Dalpha];
    % Give the Newton step direction:
    step = Hessiantemp \ NGradtemp;
    
    % Update the estimates:
    beta = beta + step(1:p);
    alpha = alpha + step((p + 1):end);
    

    % Update the residual matrix:
    for k = 1:K
        resmat(:, k) = y - feval(gfunc, x, beta) - qs(k) * feval(sfunc, x, alpha);
    end
    
    % Update the surrogate function value:
    surr = sum(sum(wwtmat ./ hwtmat .* resmat.^2 .* weightsmat)) + ...
           sum(sum(wwtmat ./ hwtmat .* (4 * tausmat - 2) .* resmat));
       
    % Now do the step-halving procedure to ensure a decrease in the value
    % of the surrogate function
    while surr > lastsurr
        step = step / 2;
        beta = beta - step(1:p);
        alpha = alpha - step((p + 1):end);
        for k = 1:K
            resmat(:, k) = y - feval(gfunc, x, beta) - qs(k) * feval(sfunc, x, alpha);
        end
        surr = sum(sum(wwtmat ./ hwtmat .* resmat.^2 .* weightsmat)) + ...
           sum(sum(wwtmat ./ hwtmat .* (4 * tausmat - 2) .* resmat));
    end
    
    change = lastsurr - surr;
    weightsmat = 1 ./ (epsilon + abs(resmat));
    lastsurr = sum(sum(wwtmat ./ hwtmat .* resmat.^2 .* weightsmat)) + ...
           sum(sum(wwtmat ./ hwtmat .* (4 * tausmat - 2) .* resmat));
end
end
    
        
    
    
    
    
    
    
            
    
    
    
           
           

