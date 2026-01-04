function [alpha, x_new, f_new, grad_new, count] = armijo_goldstein_search(fun, x, d, f_val, grad)
    % Armijo-Goldstein 非精确线搜索 (退回到Armijo条件)
    % 输入:
    %   fun     - 目标函数句柄
    %   x       - 当前点
    %   d       - 搜索方向
    %   f_val   - 当前点的函数值 f(x)
    %   grad    - 当前点的梯度 ∇f(x)
    % 输出:
    %   alpha   - 找到的步长
    %   x_new   - 新点 x + alpha*d
    %   f_new   - 新点的函数值 f(x_new)
    %   grad_new- 新点的梯度 ∇f(x_new)
    %   count   - 本次线搜索的函数调用次数

    % Armijo条件参数
    c1 = 0.01;      % 通常取一个很小的值，如 0.01
    rho = 0.5;      % 步长缩减因子，如 0.5

    alpha = 1.0;    % 初始试探步长
    count = 0;      % 函数调用计数器

    % 计算方向导数
    gradTd = grad' * d;
    
    % 如果方向导数非负，说明d不是下降方向，直接返回0步长
    if gradTd >= 0
        warning('搜索方向不是下降方向，线搜索返回步长0。');
        alpha = 0;
        x_new = x;
        f_new = f_val;
        grad_new = grad;
        return;
    end

    while true
        % 计算新点和新点的函数值
        x_candidate = x + alpha * d;
        [f_candidate, ~] = feval(fun, x_candidate);
        count = count + 1;

        % 检查Armijo条件
        % f(x + alpha*d) <= f(x) + c1*alpha*∇f(x)^T*d
        if f_candidate <= f_val + c1 * alpha * gradTd
            % 找到满足条件的alpha，跳出循环
            break;
        else
            % Armijo条件不满足，缩小步长
            alpha = alpha * rho;
        end
    end

    % 计算新点的梯度 (这是一次额外的函数调用)
    [f_new, grad_new] = feval(fun, x_candidate);
    count = count + 1;
    
    x_new = x_candidate;
end