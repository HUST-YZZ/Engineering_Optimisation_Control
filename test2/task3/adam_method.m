function [x_hist, grad_norm_hist, func_count] = adam_method(fun, x0, alpha, beta1, beta2, epsilon_alg, epsilon, max_iter)
    % Adam（自适应矩估计）法
    % 输入：目标函数，初始点，学习率α，衰减率β1、β2，算法参数ε，容忍度，最大迭代次数
    % 输出：迭代点历史，梯度范数历史，函数调用次数
    
    x = x0(:);
    m = zeros(size(x));  % 一阶矩估计
    v = zeros(size(x));  % 二阶矩估计
    [f_val, grad] = feval(fun, x);
    grad_norm = norm(grad);
    
    x_hist = [x'];
    grad_norm_hist = [grad_norm];
    func_count = 1;
    t = 0;
    
    while grad_norm > epsilon && t < max_iter
        t = t + 1;
        
        % Adam更新步骤
        g = grad;  % 当前梯度
        
        % 更新一阶矩估计
        m = beta1 * m + (1 - beta1) * g;
        
        % 更新二阶矩估计
        v = beta2 * v + (1 - beta2) * (g.^2);
        
        % 偏差修正
        m_hat = m / (1 - beta1^t);
        v_hat = v / (1 - beta2^t);
        
        % 参数更新
        x = x - alpha * m_hat ./ (sqrt(v_hat) + epsilon_alg);
        
        % 计算新位置的函数值和梯度
        [f_val, grad] = feval(fun, x);
        func_count = func_count + 1;
        grad_norm = norm(grad);
        
        % 记录历史
        x_hist = [x_hist; x'];
        grad_norm_hist = [grad_norm_hist; grad_norm];
        
        % 显示进度
        if mod(t, 50) == 0
            fprintf('Adam: 迭代 %d, 梯度范数 %.2e\n', t, grad_norm);
        end
    end
    
    fprintf('Adam完成: %d 次迭代, %d 次函数调用\n', t, func_count);
end