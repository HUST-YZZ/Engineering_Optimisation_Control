function [f0, df0, f, df] = truss_details(x, c1, c2)
    x1 = x(1); x2 = x(2);
    fac = sqrt(1 + x2^2);
    
    % 1. 目标函数及梯度
    f0 = c1 * x1 * fac;
    df0 = [c1 * fac; 
           c1 * x1 * x2 / fac];
    
    % 2. 约束函数 (应力 sigma1, sigma2)
    s1 = c2 * fac * (8/x1 + 1/(x1*x2));
    s2 = c2 * fac * (8/x1 - 1/(x1*x2));
    f = [s1 - 1; s2 - 1]; % 归一化为 <= 0
    
    % 3. 解析梯度 (精确推导)
    % ds1/dx1 = -s1/x1
    % ds1/dx2 = c2 * (x2/fac * (8/x1 + 1/(x1*x2)) + fac * (-1/(x1*x2^2)))
    ds1dx1 = -s1 / x1;
    ds1dx2 = c2 * (x2/fac * (8/x1 + 1/(x1*x2)) - fac / (x1*x2^2));
    
    ds2dx1 = -s2 / x1;
    ds2dx2 = c2 * (x2/fac * (8/x1 - 1/(x1*x2)) + fac / (x1*x2^2));
    
    df = [ds1dx1, ds1dx2; 
          ds2dx1, ds2dx2];
end