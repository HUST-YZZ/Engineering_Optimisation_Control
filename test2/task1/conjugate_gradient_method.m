function [x_hist, grad_norm_hist, func_count] = conjugate_gradient_method(fun, x0, epsilon)
    % 共轭梯度法（Fletcher-Reeves）
    % 输入：目标函数，初始点，容忍度
    % 输出：迭代点历史，梯度范数历史，函数调用次数
    
    x = x0(:);
    x_hist = [x'];
    grad_norm_hist = [];
    total_func_count = 0;
    k = 0;
    max_iter = 1000;
    
    % 第一次迭代
    [~, grad] = feval(fun, x);
    grad_norm = norm(grad);
    grad_norm_hist = [grad_norm_hist; grad_norm];
    
    % 检查初始点是否就是最优点
    if grad_norm <= epsilon
        func_count = 1; % 只调用了一次函数计算梯度
        return;
    end
    
    d = -grad;  % 初始方向为负梯度
    
    while k < max_iter
        % 精确线搜索
        [alpha, x_new, ~, count] = exact_line_search(fun, x, d);
        total_func_count = total_func_count + count;
        
        % 更新点
        x = x_new;
        x_hist = [x_hist; x'];
        
        % 计算新梯度
        [~, grad_new] = feval(fun, x);
        total_func_count = total_func_count + 1;  % 梯度计算包含一次函数调用
        grad_norm_new = norm(grad_new);
        grad_norm_hist = [grad_norm_hist; grad_norm_new];
        
        % 检查终止条件
        if grad_norm_new <= epsilon
            break;
        end
        
        % 计算FR参数beta
        beta = (grad_new' * grad_new) / (grad' * grad);
        
        % --- 核心修正 ---
        % 计算新的共轭方向
        d_new = -grad_new + beta * d;
        
        % 为下一次迭代准备变量
        grad = grad_new;
        d = d_new;
        
        k = k + 1;
    end
    func_count = total_func_count;
end