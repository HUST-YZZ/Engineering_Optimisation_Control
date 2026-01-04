function [x_hist, grad_norm_hist, func_count] = gradient_method(fun, x0, epsilon)
    % 梯度法（最速下降法）
    % 输入：目标函数，初始点，容忍度
    % 输出：迭代点历史，梯度范数历史，函数调用次数
    
    x = x0(:);
    x_hist = [x'];
    grad_norm_hist = [];
    total_func_count = 0;
    k = 0;
    max_iter = 2000;
    
    while true
        % 计算当前梯度和函数值
        [~, grad] = feval(fun, x);
        grad_norm = norm(grad);
        grad_norm_hist = [grad_norm_hist; grad_norm];
        
        % 检查终止条件
        if grad_norm <= epsilon || k >= max_iter
            break;
        end
        
        % 确定搜索方向（负梯度方向）
        d = -grad;
        
        % 精确线搜索
        [alpha, x_new, ~, count] = exact_line_search(fun, x, d);
        total_func_count = total_func_count + count;
        
        % 更新点
        x = x_new;
        x_hist = [x_hist; x'];
        k = k + 1;
    end
    func_count = total_func_count;
end