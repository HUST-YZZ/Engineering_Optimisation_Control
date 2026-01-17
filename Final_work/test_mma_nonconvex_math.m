function test_mma_nonconvex_math()
    n = 2; % 试题要求可视化 n=2
    [S, P, Q] = generate_SPQ_matrices(n); 
    
    prob_func = @(x) deal(...
        x' * S * x, ...                 
        2 * S * x, ...                  
        [n/2 - x'*P*x; n/2 - x'*Q*x], ... 
        [-2*P*x, -2*Q*x]' ...           
    );

    x0 = [1; 1]; 
    lb = [-1; -1];  
    ub = [1; 1];   

    fprintf('\n--- 运行测试二：可行域非凸数学问题 ---\n');
    % 1. 运行 MMA
    [x_opt, f_opt, history] = my_mma_solver(prob_func, x0, lb, ub,0.7);
    
    % 2. 可视化
    visualize_mma_nonconvex(history, lb, ub, S, P, Q, n);
    
    fprintf('最优解 x*: [%.4f, %.4f]\n', x_opt(1), x_opt(2));
    fprintf('最小目标值 f0*: %.4f\n', f_opt);
end
function visualize_mma_nonconvex(history, lb, ub, S, P, Q, n)
    % 1. 准备网格数据
    [X1, X2] = meshgrid(linspace(lb(1), ub(1), 100), linspace(lb(2), ub(2), 100));
    F0 = zeros(size(X1)); CON1 = zeros(size(X1)); CON2 = zeros(size(X1));

    % 计算原始问题的地形
    for i = 1:numel(X1)
        xt = [X1(i); X2(i)];
        F0(i) = xt' * S * xt;           % 目标函数: x'Sx
        CON1(i) = n/2 - xt' * P * xt;   % 约束1: n/2 - x'Px <= 0
        CON2(i) = n/2 - xt' * Q * xt;   % 约束2: n/2 - x'Qx <= 0
    end

    figure('Color', 'w', 'Position', [150, 150, 1000, 400]);

    % --- 左图：非凸空间的迭代轨迹 ---
    subplot(1, 2, 1);
    contour(X1, X2, F0, 20); hold on;
    % 画出非凸约束边界 (f=0)
    contour(X1, X2, CON1, [0 0], 'r', 'LineWidth', 2);
    contour(X1, X2, CON2, [0 0], 'r--', 'LineWidth', 2);
    % 轨迹
    plot(history.x(1, :), history.x(2, :), 'bo-', 'LineWidth', 1);
    plot(history.x(1, 1), history.x(2, 1), 'gs', 'MarkerFaceColor', 'g');
    plot(history.x(1, end), history.x(2, end), 'rp', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    title('Non-convex Math Problem Path');
    legend('Obj Contours', 'Con1 (Non-convex)', 'Con2 (Non-convex)', 'MMA Path', 'Start', 'Optimal');
    xlabel('x_1'); ylabel('x_2'); grid on;

    % --- 右图：子问题在非凸边界上的近似 ---
    subplot(1, 2, 2);
    k_show = 7; % 观察第2次迭代
    sub = history.subproblem{k_show};
    xk = history.x(:, k_show);
    
    % 计算 MMA 子问题的约束 1 近似
    CON1_sub = zeros(size(X1));
    for i = 1:numel(X1)
        xt = [X1(i); X2(i)];
        % MMA 近似公式: r + sum(p/(U-x) + q/(x-L))
        CON1_sub(i) = sub.r(2) + sum(sub.p(2,:)'./(sub.U - xt) + sub.q(2,:)'./(xt - sub.L));
    end
    
    contour(X1, X2, CON1, [0 0], 'r', 'LineWidth', 1, 'LineStyle', ':'); hold on; % 原始
    contour(X1, X2, CON1_sub, [0 0], 'b', 'LineWidth', 2); % MMA 近似
    plot(xk(1), xk(2), 'ko', 'MarkerFaceColor', 'k');
    title(['MMA Convex Approximation (Iter ' num2str(k_show) ')']);
    legend('Original Non-convex Boundary', 'MMA Convex Approx Boundary', 'Current Point');
    xlabel('x_1'); ylabel('x_2'); grid on;
end