function [alpha, x_new, f_new, func_count] = exact_line_search(fun, x, d)
    % 精确一维搜索
    % 输入：目标函数，当前点x，搜索方向d
    % 输出：最优步长alpha，新点x_new，函数值f_new，函数调用次数
    
    % 1. 用进退法确定搜索区间
    [a_low, a_high, count1] = advance_retreat_method(@(a) phi(fun, x, d, a));
    
    % 2. 使用fminbnd进行精确搜索
    phi_func = @(a) phi(fun, x, d, a);
    [alpha, f_alpha, ~, output] = fminbnd(phi_func, a_low, a_high);
    
    % 3. 计算输出
    x_new = x + alpha * d;
    f_new = feval(fun, x_new);
    func_count = count1 + output.funcCount + 1; % +1用于计算f_new
end

function f_val = phi(fun, x, d, alpha)
    % 一维辅助函数
    x_new = x + alpha * d;
    f_val = feval(fun, x_new);
end

function [a_low, a_high, count] = advance_retreat_method(phi_func)
    % 进退法确定搜索区间 [a_low, a_high]，使得 phi(a) 在该区间上是单峰的
    % 修正版：确保逻辑的严谨性，避免无限循环和重复计算

    a0 = 0;
    h0 = 0.01;  % 初始步长
    count = 0;
    
    % 1. 初始化前两个点
    a_prev = a0;
    f_prev = phi_func(a_prev);
    count = count + 1;
    
    a_curr = a0 + h0;
    f_curr = phi_func(a_curr);
    count = count + 1;
    
    % 2. 确定初始搜索方向
    if f_curr < f_prev
        % 方向正确，步长加倍，计算下一个点
        h = h0;
        a_next = a_curr + h;
        f_next = phi_func(a_next);
        count = count + 1;
    else
        % 方向错误，交换点并反向搜索
        h = -h0;
        % 交换 a_prev 和 a_curr
        temp_a = a_prev;
        temp_f = f_prev;
        a_prev = a_curr;
        f_prev = f_curr;
        a_curr = temp_a;
        f_curr = temp_f;
        
        % 计算新的 a_next
        a_next = a_curr + h;
        f_next = phi_func(a_next);
        count = count + 1;
    end
    
    % 3. 循环进退，直到找到 f_curr 是极小点的迹象 (f_prev > f_curr < f_next)
    while f_curr > f_next
        % 当前点 a_curr 的函数值不是最小的，继续前进
        % 更新三个点，将 a_curr 作为新的 a_prev, a_next 作为新的 a_curr
        a_prev = a_curr;
        f_prev = f_curr;
        a_curr = a_next;
        f_curr = f_next;
        
        % 步长加倍
        h = 2 * h;
        
        % 计算新的 a_next
        a_next = a_curr + h;
        f_next = phi_func(a_next);
        count = count + 1;
    end
    
    % 4. 循环结束，说明找到了一个“峰”（即 f_prev > f_curr < f_next）
    % 确定最终的搜索区间
    a_low = min(a_prev, a_next);
    a_high = max(a_prev, a_next);
end