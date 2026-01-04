function [f, grad] = freudenstein_roth(x)
    % 计算目标函数值和梯度
    x1 = x(1);
    x2 = x(2);
    
    % 第一项
    term1 = -13 + x1 + ((5 - x2) * x2 - 2) * x2;
    % 第二项
    term2 = -29 + x1 + ((x2 + 1) * x2 - 14) * x2;
    
    % 目标函数值
    f = term1^2 + term2^2;
    
    % 梯度计算（通过符号微分得到）
    if nargout > 1
        grad = zeros(2, 1);
        % 对x1偏导
        grad(1) = 2 * term1 + 2 * term2;
        % 对x2偏导
        grad(2) = 2 * term1 * ((5 - 2*x2)*x2 + (5-x2)*x2 - 2) + ...
                  2 * term2 * ((2*x2+1)*x2 + (x2+1)*x2 - 14);
    end
end