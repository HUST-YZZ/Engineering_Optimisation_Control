function test_mma_2bar_truss()
    % 常数定义 (根据 Svanberg 论文)
    c1 = 1.0; 
    c2 = 0.124;
    
    % 目标函数: 重量 w = c1 * x1 * sqrt(1 + x2^2)
    % 约束函数: 应力 sigma <= 100 (归一化处理)
    
    prob_func = @(x) truss_details(x, c1, c2);

    x0 = [1.5; 0.5];    % 初始值
    lb = [0.2; 0.1];    % 下界
    ub = [4.0; 1.6];    % 上界

    fprintf('\n--- 运行测试一：二杆桁架问题 ---\n');
    % 运行算法并获取历史记录
    [x_opt, f_opt, history] = my_mma_solver(prob_func, x0, lb, ub,0.7);
    
    % 调用可视化
    visualize_mma_truss(history, lb, ub);
    
    fprintf('最优截面积 x1: %.4f cm^2\n', x_opt(1));
    fprintf('最优半跨度 x2: %.4f m\n', x_opt(2));
    fprintf('最小重量: %.4f kg\n', f_opt);
end

function visualize_mma_truss(history, lb, ub)
    % 1. 准备网格数据
    [X1, X2] = meshgrid(linspace(lb(1), ub(1), 100), linspace(lb(2), ub(2), 100));
    F0 = zeros(size(X1)); CON1 = zeros(size(X1)); CON2 = zeros(size(X1));
    c1 = 1.0; c2 = 0.124;

    for i = 1:numel(X1)
        x_tmp = [X1(i); X2(i)];
        [f0_tmp, ~, f_tmp, ~] = truss_details(x_tmp, c1, c2);
        F0(i) = f0_tmp; CON1(i) = f_tmp(1); CON2(i) = f_tmp(2);
    end

    figure('Color', 'w', 'Position', [100, 100, 1000, 400]);

    % --- 左图：迭代轨迹与原始约束 ---
    subplot(1, 2, 1);
    contour(X1, X2, F0, 20, 'LineWidth', 0.5); hold on;
    % 画出原始约束边界 (f=0)
    contour(X1, X2, CON1, [0 0], 'r', 'LineWidth', 2);
    contour(X1, X2, CON2, [0 0], 'r--', 'LineWidth', 2);
    % 画出迭代路径
    plot(history.x(1, :), history.x(2, :), 'bo-', 'MarkerSize', 4, 'MarkerFaceColor', 'b');
    plot(history.x(1, 1), history.x(2, 1), 'gs', 'MarkerSize', 8, 'MarkerFaceColor', 'g'); % 起点
    plot(history.x(1, end), history.x(2, end), 'rp', 'MarkerSize', 10, 'MarkerFaceColor', 'r'); % 终点
    title('Iteration Path on Original Design Space');
    legend('Weight Contours', 'Stress 1 Limit', 'Stress 2 Limit', 'MMA Path', 'Start', 'Optimal');
    xlabel('x_1 (Area)'); ylabel('x_2 (Shape)'); grid on;

    % --- 右图：对比子问题近似 (取第1步或第2步展示) ---
    subplot(1, 2, 2);
    k_show = 8; % 展示第2步的近似情况
    sub = history.subproblem{k_show};
    xk = history.x(:, k_show);
    F_sub = zeros(size(X1));
    for i = 1:numel(X1)
        xt = [X1(i); X2(i)];
        % MMA 子问题目标函数近似公式
        F_sub(i) = sub.r(1) + sum(sub.p(1,:)'./(sub.U - xt) + sub.q(1,:)'./(xt - sub.L));
    end
    contour(X1, X2, F_sub, 20, 'Color', [0.5 0.5 0.5]); hold on;
    contour(X1, X2, CON1, [0 0], 'r', 'LineWidth', 1, 'LineStyle', ':'); % 原始约束
    % 计算并画出子问题的约束近似边界
    CON1_sub = zeros(size(X1));
    for i = 1:numel(X1)
        xt = [X1(i); X2(i)];
        CON1_sub(i) = sub.r(2) + sum(sub.p(2,:)'./(sub.U - xt) + sub.q(2,:)'./(xt - sub.L));
    end
    contour(X1, X2, CON1_sub, [0 0], 'b', 'LineWidth', 2); % 近似约束
    plot(xk(1), xk(2), 'ko', 'MarkerFaceColor', 'k');
    title(['MMA Subproblem Approx (Iter ' num2str(k_show) ')']);
    legend('Approx Obj', 'Original Con1', 'MMA Approx Con1', 'Current Point');
    xlabel('x_1'); ylabel('x_2'); grid on;
end