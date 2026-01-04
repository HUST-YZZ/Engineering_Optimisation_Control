function visualize_optimization(fun, x_hist, grad_norm_hist, func_count, method_name)
    % 可视化优化过程：等高线+搜索路径、梯度范数曲线
    % 输入：
    %   fun         - 目标函数句柄（返回 [f_val, grad_val]）
    %   x_hist      - 迭代主点历史（k=1,2,...，一维搜索输出点）
    %   grad_norm_hist - 主点的梯度范数历史
    %   func_count  - 目标函数总计算次数
    %   method_name - 优化方法名称（标题用）

    % ========== 1. 绘制目标函数等高线 + 搜索路径 ==========
    figure('Name', [method_name ' - 等高线与搜索路径']);
    
    % 构造设计空间网格（根据x_hist自适应范围）
    pad_ratio = 0.2;
    x1_min = min(x_hist(:, 1)); x1_max = max(x_hist(:, 1));
    x2_min = min(x_hist(:, 2)); x2_max = max(x_hist(:, 2));
    x1_range = linspace(x1_min - pad_ratio*abs(x1_max-x1_min), ...
                        x1_max + pad_ratio*abs(x1_max-x1_min), 100);
    x2_range = linspace(x2_min - pad_ratio*abs(x2_max-x2_min), ...
                        x2_max + pad_ratio*abs(x2_max-x2_min), 100);
    [X1, X2] = meshgrid(x1_range, x2_range);
    F = zeros(size(X1));
    for i = 1:size(X1,1)
        for j = 1:size(X1,2)
            x = [X1(i,j); X2(i,j)];
            F(i,j) = fun(x);  % 计算网格点的目标函数值
        end
    end
    
    % 画等高线
    contour(X1, X2, F, 50, 'LineWidth', 1); hold on;
    % 画搜索路径（主点）
    plot(x_hist(:,1), x_hist(:,2), 'ro-', 'LineWidth', 1.5, 'MarkerSize', 6);
    % 标记起点和终点
    plot(x_hist(1,1), x_hist(1,2), 'gs', 'MarkerSize', 8, 'DisplayName', '起点');
    plot(x_hist(end,1), x_hist(end,2), 'bd', 'MarkerSize', 8, 'DisplayName', '终点');
    
    % 标注标题、坐标轴
    title([method_name ' - 目标函数等高线与搜索路径']);
    xlabel('x_1'); ylabel('x_2');
    legend('show');
    grid on; hold off;


    % ========== 2. 绘制梯度范数下降曲线 + 显示函数计算次数 ==========
    figure('Name', [method_name ' - 梯度范数与函数计算次数']);
    
    % 画梯度范数下降曲线
    plot(1:length(grad_norm_hist), grad_norm_hist, 'b-o', 'LineWidth', 1.2);
    title([method_name ' - 梯度范数下降曲线（函数计算次数：' num2str(func_count) '）']);
    xlabel('迭代次数 k'); ylabel('梯度范数 ||∇f(x^{(k)})||');
    grid on;
end