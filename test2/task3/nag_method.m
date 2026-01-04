function [x_hist, grad_norm_hist, func_count] = nag_method(fun, x0, alpha, beta, epsilon, max_iter)
    % NAG（Nesterov加速梯度）法
    % 输入：目标函数，初始点，学习率α，动量参数β，容忍度，最大迭代次数
    % 输出：迭代点历史，梯度范数历史，函数调用次数
    
    x = x0(:);
    v = zeros(size(x));  % 动量项初始化
    [f_val, grad] = feval(fun, x);
    grad_norm = norm(grad);
    
    x_hist = [x'];
    grad_norm_hist = [grad_norm];
    func_count = 1;
    t = 0;
    
    while grad_norm > epsilon && t < max_iter
        % NAG更新步骤
        y = x + beta * v;  % 前瞻位置
        
        % 计算前瞻位置的梯度
        [~, grad_y] = feval(fun, y);
        func_count = func_count + 1;
        
        % 更新动量项
        v = beta * v - alpha * grad_y;
        
        % 更新位置
        x = x + v;
        
        % 计算新位置的梯度
        [f_val, grad] = feval(fun, x);
        func_count = func_count + 1;
        grad_norm = norm(grad);
        
        % 记录历史
        x_hist = [x_hist; x'];
        grad_norm_hist = [grad_norm_hist; grad_norm];
        
        t = t + 1;
        
        % 显示进度
        if mod(t, 50) == 0
            fprintf('NAG: 迭代 %d, 梯度范数 %.2e\n', t, grad_norm);
        end
    end
    
    fprintf('NAG完成: %d 次迭代, %d 次函数调用\n', t, func_count);
end