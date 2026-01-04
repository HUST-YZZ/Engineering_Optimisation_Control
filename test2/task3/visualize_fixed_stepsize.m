function visualize_fixed_stepsize_improved(fun, x_hist_nag, x_hist_adam, grad_norm_nag, grad_norm_adam)
    % 固定步长方法可视化 (改进版)
    
    figure('Position', [100, 100, 1400, 600]);
    
    % --- 子图1 & 2: 单个算法的路径 (调用改进后的函数) ---
    subplot(2, 3, 1);
    plot_search_path_improved(fun, x_hist_nag, 'b', 'o', 'NAG法');
    
    subplot(2, 3, 2);
    plot_search_path_improved(fun, x_hist_adam, 'r', 's', 'Adam法');
    
    % --- 子图3: 路径对比 (调用改进后的函数) ---
    subplot(2, 3, 3);
    plot_comparison_path_improved(fun, x_hist_nag, x_hist_adam);
    
    % --- 子图4 & 5: 单个算法的梯度曲线 (无需修改) ---
    subplot(2, 3, 4);
    semilogy(0:length(grad_norm_nag)-1, grad_norm_nag, 'b-o', ...
        'LineWidth', 2, 'MarkerSize', 4, 'MarkerFaceColor', 'b');
    xlabel('迭代次数 k');
    ylabel('梯度范数 ||∇f||');
    title('NAG法 - 梯度下降曲线');
    grid on;
    ylim([1e-8, max(grad_norm_nag)*10]);
    
    subplot(2, 3, 5);
    semilogy(0:length(grad_norm_adam)-1, grad_norm_adam, 'r-s', ...
        'LineWidth', 2, 'MarkerSize', 4, 'MarkerFaceColor', 'r');
    xlabel('迭代次数 k');
    ylabel('梯度范数 ||∇f||');
    title('Adam法 - 梯度下降曲线');
    grid on;
    ylim([1e-8, max(grad_norm_adam)*10]);
    
    % --- 子图6: 梯度曲线对比 (优化标注) ---
    subplot(2, 3, 6);
    k_nag = 0:length(grad_norm_nag)-1;
    k_adam = 0:length(grad_norm_adam)-1;
    semilogy(k_nag, grad_norm_nag, 'b-o', 'LineWidth', 1.5, ...
        'MarkerSize', 3, 'MarkerFaceColor', 'b', 'DisplayName', 'NAG');
    hold on;
    semilogy(k_adam, grad_norm_adam, 'r-s', 'LineWidth', 1.5, ...
        'MarkerSize', 3, 'MarkerFaceColor', 'r', 'DisplayName', 'Adam');
    xlabel('迭代次数 k');
    ylabel('梯度范数 ||∇f||');
    title('梯度下降曲线对比');
    legend('Location', 'northeast');
    grid on;
    hold off;
    
    % 标注收敛信息
    text(0.7, 0.3, sprintf('NAG: %d次迭代', length(grad_norm_nag)-1), ...
        'Units', 'normalized', 'FontSize', 10, 'Color', 'b');
    text(0.7, 0.2, sprintf('Adam: %d次迭代', length(grad_norm_adam)-1), ...
        'Units', 'normalized', 'FontSize', 10, 'Color', 'r');
end

% =========================================================================
% 改进后的辅助函数
% =========================================================================

function plot_search_path_improved(fun, x_hist, color, marker, title_str)
    % 绘制单个算法的搜索路径 (改进版)
    
    % --- 改进 1: 动态确定绘图范围 ---
    pad_ratio = 0.2;
    all_x = x_hist(:, 1);
    all_y = x_hist(:, 2);
    x1_range = linspace(min(all_x) - pad_ratio*abs(max(all_x)-min(all_x)), ...
                        max(all_x) + pad_ratio*abs(max(all_x)-min(all_x)), 100);
    x2_range = linspace(min(all_y) - pad_ratio*abs(max(all_y)-min(all_y)), ...
                        max(all_y) + pad_ratio*abs(max(all_y)-min(all_y)), 100);
    [X1, X2] = meshgrid(x1_range, x2_range);
    
    % --- 改进 2: 使用 arrayfun 并规范函数调用 ---
    F = arrayfun(@(x, y) feval(fun, [x; y]), X1, X2);
    
    % 绘制等高线
    contour(X1, X2, F, 30, 'LineWidth', 0.5);
    hold on;
    
    % 绘制搜索路径
    plot(x_hist(:, 1), x_hist(:, 2), [color, '-', marker], 'LineWidth', 1.5, ...
        'MarkerSize', 4, 'MarkerFaceColor', color);
    
    % 绘制搜索方向箭头（向量化并稀疏化）
    if size(x_hist, 1) > 1
        step = max(1, floor(size(x_hist, 1) / 20)); % 最多画20个箭头
        indices = 1:step:size(x_hist, 1)-1;
        quiver(x_hist(indices, 1), x_hist(indices, 2), ...
               x_hist(indices+1, 1) - x_hist(indices, 1), ...
               x_hist(indices+1, 2) - x_hist(indices, 2), ...
               0, 'k', 'LineWidth', 0.8, 'MaxHeadSize', 0.4);
    end
    
    % 标记起点和终点
    plot(x_hist(1, 1), x_hist(1, 2), 'gs', 'MarkerSize', 10, 'MarkerFaceColor', 'g', 'DisplayName', '起点');
    plot(x_hist(end, 1), x_hist(end, 2), 'ms', 'MarkerSize', 10, 'MarkerFaceColor', 'm', 'DisplayName', '终点');
    
    % 标记全局最优点
    plot(5.0, 4.0, 'k*', 'MarkerSize', 15, 'LineWidth', 2, 'DisplayName', '全局最优');
    
    xlabel('x_1');
    ylabel('x_2');
    title(title_str);
    grid on;
    legend('show', 'Location', 'best'); % 显示图例
    % axis equal; % 建议注释掉，以获得更好的视觉效果
end

function plot_comparison_path_improved(fun, x_hist_nag, x_hist_adam)
    % 绘制两种方法的搜索路径对比 (改进版)
    
    % --- 改进 1: 动态确定绘图范围 ---
    pad_ratio = 0.2;
    all_x = [x_hist_nag(:, 1); x_hist_adam(:, 1)];
    all_y = [x_hist_nag(:, 2); x_hist_adam(:, 2)];
    x1_range = linspace(min(all_x) - pad_ratio*abs(max(all_x)-min(all_x)), ...
                        max(all_x) + pad_ratio*abs(max(all_x)-min(all_x)), 100);
    x2_range = linspace(min(all_y) - pad_ratio*abs(max(all_y)-min(all_y)), ...
                        max(all_y) + pad_ratio*abs(max(all_y)-min(all_y)), 100);
    [X1, X2] = meshgrid(x1_range, x2_range);
    
    % --- 改进 2: 使用 arrayfun 并规范函数调用 ---
    F = arrayfun(@(x, y) feval(fun, [x; y]), X1, X2);
    
    % 绘制等高线
    contour(X1, X2, F, 30, 'LineWidth', 0.5, 'LineColor', [0.7, 0.7, 0.7]);
    hold on;
    
    % 绘制NAG路径
    plot(x_hist_nag(:, 1), x_hist_nag(:, 2), 'b-o', 'LineWidth', 1.5, ...
        'MarkerSize', 4, 'MarkerFaceColor', 'b', 'DisplayName', 'NAG');
    
    % 绘制Adam路径
    plot(x_hist_adam(:, 1), x_hist_adam(:, 2), 'r-s', 'LineWidth', 1.5, ...
        'MarkerSize', 4, 'MarkerFaceColor', 'r', 'DisplayName', 'Adam');
    
    % 标记起点
    plot(x_hist_nag(1, 1), x_hist_nag(1, 2), 'g^', 'MarkerSize', 12, 'MarkerFaceColor', 'g', 'DisplayName', '起点');
    
    % 标记终点和全局最优
    plot(x_hist_nag(end, 1), x_hist_nag(end, 2), 'mv', 'MarkerSize', 12, 'MarkerFaceColor', 'm', 'DisplayName', 'NAG终点');
    plot(x_hist_adam(end, 1), x_hist_adam(end, 2), 'c*', 'MarkerSize', 15, 'LineWidth', 2, 'DisplayName', 'Adam终点');
    plot(5.0, 4.0, 'kp', 'MarkerSize', 15, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'w', 'LineWidth', 2, 'DisplayName', '全局最优');
    
    xlabel('x_1');
    ylabel('x_2');
    title('搜索路径对比 (NAG vs Adam)');
    legend('show', 'Location', 'best');
    grid on;
    % axis equal; % 建议注释掉
end