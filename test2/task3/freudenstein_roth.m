function [f, grad] = freudenstein_roth(x)
    % 计算目标函数值和梯度
    x1 = x(1);
    x2 = x(2);
    
    term1 = -13 + x1 + ((5 - x2) * x2 - 2) * x2;
    term2 = -29 + x1 + ((x2 + 1) * x2 - 14) * x2;
    
    f = term1^2 + term2^2;
    
    if nargout > 1
        grad = zeros(2, 1);
        grad(1) = 2 * term1 + 2 * term2;
        grad(2) = 2 * term1 * ((5 - 2*x2)*x2 + (5-x2)*x2 - 2) + ...
                  2 * term2 * ((2*x2+1)*x2 + (x2+1)*x2 - 14);
    end
end