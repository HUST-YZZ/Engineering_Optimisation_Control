function [x_hist, grad_norm_hist, func_count] = bfgs_method(fun, x0, epsilon)
    % 基于Armijo-Goldstein的BFGS变尺度法
    % 输入：目标函数，初始点，容忍度
    % 输出：迭代点历史，梯度范数历史，函数调用次数
    
    n = length(x0);
    x = x0(:);
    [f_val, grad] = feval(fun, x);
    grad_norm = norm(grad);
    
    % 初始化近似Hessian逆矩阵为单位矩阵
    H = eye(n);
    
    x_hist = [x'];
    grad_norm_hist = [grad_norm];
    total_func_count = 1;  % 初始函数计算
    k = 0;
    max_iter = 1000;
    
    while grad_norm > epsilon && k < max_iter
        % BFGS搜索方向
        d = -H * grad;
        
        % 确保是下降方向
        if grad' * d >= 0
            d = -grad;  % 如果不是下降方向，使用负梯度
        end
        
        % 保存旧变量
        x_old = x;
        grad_old = grad;
        f_old = f_val;
        
        % Armijo-Goldstein一维搜索
        [alpha, x_new, f_new, grad_new, count] = armijo_goldstein_search(fun, x, d, f_val, grad);
        total_func_count = total_func_count + count;
        
        % 更新变量
        x = x_new;
        f_val = f_new;
        grad = grad_new;
        grad_norm = norm(grad);
        
        % BFGS更新公式：更新Hessian逆近似
        s = x - x_old;  % 参数变化
        y = grad - grad_old;  % 梯度变化
        
        % 确保满足曲率条件
        if s' * y > 1e-10
            rho = 1 / (s' * y);
            
            % BFGS更新公式
            I = eye(n);
            H = (I - rho * s * y') * H * (I - rho * y * s') + rho * (s * s');
        else
            % 曲率条件不满足，重置H为单位矩阵
            H = eye(n);
        end
        
        % 记录历史
        x_hist = [x_hist; x'];
        grad_norm_hist = [grad_norm_hist; grad_norm];
        
        k = k + 1;
        
        % 显示进度
        if mod(k, 10) == 0
            fprintf('BFGS法: 迭代 %d, 梯度范数 %.2e\n', k, grad_norm);
        end
    end
    
    func_count = total_func_count;
    fprintf('BFGS法完成: %d 次迭代, %d 次函数调用\n', k, func_count);
end