function [x_opt, f_opt, history] = my_mma_solver(problem_func, x0, lb, ub, s)
    % 输入: s 为缩放因子 (对应你调用时的 0.7)
    
    max_iter = 100;
    tol = 1e-6;
    n = length(x0);
    x = x0(:);
    x_old = x;
    x_older = x;
    
    % 1. 初始化渐近线 (MMA 标准初始规则)
    L = x - (ub - lb);
    U = x + (ub - lb);
    
    % 2. 初始化 history 结构体 (为可视化准备)
    history.x = [];
    history.subproblem = {}; % 存储每一代的系数和渐近线
    history.f0 = [];

    for k = 1:max_iter
        % A. 计算当前点的函数值和梯度
        [f0, df0, f, df] = problem_func(x);
        
        % B. 构造 MMA 子问题的系数 p, q, r
        [p, q, r] = generate_mma_approx(x, f0, df0, f, df, L, U);
        
        % --- 关键：记录历史数据 (供可视化使用) ---
        history.x(:, k) = x;
        history.subproblem{k} = struct('p', p, 'q', q, 'r', r, 'L', L, 'U', U);
        history.f0(:, k) = f0;
        % ---------------------------------------
        
        % C. 求解显式凸子问题
        x_new = solve_subproblem(p, q, r, L, U, lb, ub, x);
        
        % 打印进度
        fprintf('Iter %d: f0 = %.4f\n', k, f0);
        
        % D. 收敛判断
        if norm(x_new - x) < tol
            break;
        end
        
        % E. 更新移动渐近线 (调用之前写的带 s 的更新函数)
        [L, U] = update_asymptotes(k, x, x_old, x_older, L, U, lb, ub, s);
        
        % F. 历史点迭代
        x_older = x_old;
        x_old = x;
        x = x_new;
    end
    
    x_opt = x;
    f_opt = f0;
end