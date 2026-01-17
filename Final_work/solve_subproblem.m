function x_next = solve_subproblem(p, q, r, L, U, lb, ub, x_curr)
    n = length(x_curr);
    % 子问题的目标函数
    sub_obj = @(x) r(1) + sum(p(1,:)./(U' - x') + q(1,:)./(x' - L'));
    
    % 子问题的非线性约束
    sub_con = @(x) deal(r(2:end) + sum(p(2:end,:)./(U' - x') + q(2:end,:)./(x' - L'), 2), []);
    
    % 移动限制 (Move Limits)，防止分母为0
    alpha = 0.1; % 强制安全边界因子
    sub_lb = max(lb, L + alpha*(x_curr-L));
    sub_ub = min(ub, U - alpha*(U-x_curr));
    
    options = optimoptions('fmincon','Display','none',...
        'Algorithm','sqp',...
        'ConstraintTolerance', 1e-7,...
        'OptimalityTolerance', 1e-7);
    x_next = fmincon(sub_obj, x_curr, [], [], [], [], sub_lb, sub_ub, sub_con, options);
end