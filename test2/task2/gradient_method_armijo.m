function [x_hist, grad_norm_hist, func_count] = gradient_method_armijo(fun, x0, epsilon)
    % 基于Armijo-Goldstein的梯度法
    % 输入：目标函数，初始点，容忍度
    % 输出：迭代点历史，梯度范数历史，函数调用次数
    
    x = x0(:);
    [f_val, grad] = feval(fun, x);
    grad_norm = norm(grad);
    
    x_hist = [x'];
    grad_norm_hist = [grad_norm];
    total_func_count = 1;  % 初始函数计算
    k = 0;
    max_iter = 10000;
    
    while grad_norm > epsilon && k < max_iter
        % 确定搜索方向（负梯度方向）
        d = -grad;
        
        % Armijo-Goldstein一维搜索
        [alpha, x_new, f_new, grad_new, count] = armijo_goldstein_search(fun, x, d, f_val, grad);
        total_func_count = total_func_count + count;
        
        % 更新变量
        x = x_new;
        f_val = f_new;
        grad = grad_new;
        grad_norm = norm(grad);
        
        % 记录历史
        x_hist = [x_hist; x'];
        grad_norm_hist = [grad_norm_hist; grad_norm];
        
        k = k + 1;
        
        % 显示进度
        if mod(k, 10) == 0
            fprintf('梯度法: 迭代 %d, 梯度范数 %.2e\n', k, grad_norm);
        end
    end
    
    func_count = total_func_count;
    fprintf('梯度法完成: %d 次迭代, %d 次函数调用\n', k, func_count);
end