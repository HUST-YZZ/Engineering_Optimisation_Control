function visualize_optimization_comparison(fun, x_hist_grad, x_hist_bfgs, grad_norm_grad, grad_norm_bfgs)
    % 比较梯度法和BFGS法的可视化 (改进版)
    
    figure('Position', [100, 100, 1400, 600]);
    
    % --- 子图1 & 2: 单个算法的路径 (调用改进后的函数) ---
    subplot(2, 3, 1);
    plot_contour_and_path_improved(fun, x_hist_grad, 'b', 'o');
    title('梯度法 - 搜索路径');
    
    subplot(2, 3, 2);
    plot_contour_and_path_improved(fun, x_hist_bfgs, 'r', 's');
    title('BFGS法 - 搜索路径');
    
    % --- 子图3: 路径对比 (调用改进后的函数) ---
    subplot(2, 3, 3);
    plot_comparison_path_improved(fun, x_hist_grad, x_hist_bfgs);
    title('搜索路径对比');
    
    % --- 子图4 & 5: 单个算法的梯度曲线 (无需修改) ---
    subplot(2, 3, 4);
    semilogy(0:length(grad_norm_grad)-1, grad_norm_grad, 'b-o', ...
        'LineWidth', 2, 'MarkerFaceColor', 'b');
    xlabel('迭代次数 k');
    ylabel('梯度范数 ||∇f||');
    title('梯度法 - 梯度下降曲线');
    grid on;
    ylim([1e-8, max(grad_norm_grad)*10]);
    
    subplot(2, 3, 5);
    semilogy(0:length(grad_norm_bfgs)-1, grad_norm_bfgs, 'r-s', ...
        'LineWidth', 2, 'MarkerFaceColor', 'r');
    xlabel('迭代次数 k');
    ylabel('梯度范数 ||∇f||');
    title('BFGS法 - 梯度下降曲线');
    grid on;
    ylim([1e-8, max(grad_norm_bfgs)*10]);
    
    % --- 子图6: 梯度曲线对比 (无需修改) ---
    subplot(2, 3, 6);
    k_grad = 0:length(grad_norm_grad)-1;
    k_bfgs = 0:length(grad_norm_bfgs)-1;
    semilogy(k_grad, grad_norm_grad, 'b-o', 'LineWidth', 2, ...
        'MarkerFaceColor', 'b', 'DisplayName', '梯度法');
    hold on;
    semilogy(k_bfgs, grad_norm_bfgs, 'r-s', 'LineWidth', 2, ...
        'MarkerFaceColor', 'r', 'DisplayName', 'BFGS法');
    xlabel('迭代次数 k');
    ylabel('梯度范数 ||∇f||');
    title('梯度下降曲线对比');
    legend('Location', 'northeast');
    grid on;
    hold off;
end

% =========================================================================
% 改进后的辅助函数
% =========================================================================

function plot_contour_and_path_improved(fun, x_hist, color, marker)
    % 绘制等高线和搜索路径 (改进版)
    
    % --- 改进 1: 动态确定绘图范围 ---
    pad_ratio = 0.2;
    all_x = [x_hist(:, 1)];
    all_y = [x_hist(:, 2)];
    x1_range = linspace(min(all_x) - pad_ratio*abs(max(all_x)-min(all_x)), ...
                        max(all_x) + pad_ratio*abs(max(all_x)-min(all_x)), 100);
    x2_range = linspace(min(all_y) - pad_ratio*abs(max(all_y)-min(all_y)), ...
                        max(all_y) + pad_ratio*abs(max(all_y)-min(all_y)), 100);
    [X1, X2] = meshgrid(x1_range, x2_range);
    
    % 计算函数值
    F = arrayfun(@(x, y) feval(fun, [x; y]), X1, X2);
    
    % 绘制等高线
    contour(X1, X2, F, 30, 'LineWidth', 0.5);
    hold on;
    
    % 绘制搜索路径
    plot(x_hist(:, 1), x_hist(:, 2), [color, '-', marker], 'LineWidth', 1.5, ...
        'MarkerSize', 6, 'MarkerFaceColor', color);
    
%     % --- 改进 2: 向量化绘制箭头，提高效率 ---
%     if size(x_hist, 1) > 1
%         quiver(x_hist(1:end-1, 1), x_hist(1:end-1, 2), ...
%                x_hist(2:end, 1) - x_hist(1:end-1, 1), ...
%                x_hist(2:end, 2) - x_hist(1:end-1, 2), ...
%                0, 'k', 'LineWidth', 0.8, 'MaxHeadSize', 0.4);
%     end
    
    % 标记起点和终点
    plot(x_hist(1, 1), x_hist(1, 2), 'gs', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
    plot(x_hist(end, 1), x_hist(end, 2), 'ms', 'MarkerSize', 10, 'MarkerFaceColor', 'm');
    
    xlabel('x_1');
    ylabel('x_2');
    grid on;
    % axis equal; % 建议注释掉，以获得更好的视觉效果
end

function plot_comparison_path_improved(fun, x_hist_grad, x_hist_bfgs)
    % 绘制两种方法的搜索路径对比 (改进版)
    
    % --- 改进 1: 动态确定绘图范围 ---
    pad_ratio = 0.2;
    all_x = [x_hist_grad(:, 1); x_hist_bfgs(:, 1)];
    all_y = [x_hist_grad(:, 2); x_hist_bfgs(:, 2)];
    x1_range = linspace(min(all_x) - pad_ratio*abs(max(all_x)-min(all_x)), ...
                        max(all_x) + pad_ratio*abs(max(all_x)-min(all_x)), 100);
    x2_range = linspace(min(all_y) - pad_ratio*abs(max(all_y)-min(all_y)), ...
                        max(all_y) + pad_ratio*abs(max(all_y)-min(all_y)), 100);
    [X1, X2] = meshgrid(x1_range, x2_range);
    
    % 计算函数值
    F = arrayfun(@(x, y) feval(fun, [x; y]), X1, X2);
    
    % 绘制等高线
    contour(X1, X2, F, 30, 'LineWidth', 0.5, 'LineColor', [0.7, 0.7, 0.7]);
    hold on;
    
    % 绘制梯度法路径
    plot(x_hist_grad(:, 1), x_hist_grad(:, 2), 'b-o', 'LineWidth', 1.5, ...
        'MarkerSize', 5, 'MarkerFaceColor', 'b', 'DisplayName', '梯度法');
    
    % 绘制BFGS法路径
    plot(x_hist_bfgs(:, 1), x_hist_bfgs(:, 2), 'r-s', 'LineWidth', 1.5, ...
        'MarkerSize', 5, 'MarkerFaceColor', 'r', 'DisplayName', 'BFGS法');
    
    % 标记起点和终点
    plot(x_hist_grad(1, 1), x_hist_grad(1, 2), 'g^', 'MarkerSize', 12, 'MarkerFaceColor', 'g', 'DisplayName', '起点');
    plot(x_hist_grad(end, 1), x_hist_grad(end, 2), 'mv', 'MarkerSize', 12, 'MarkerFaceColor', 'm', 'DisplayName', '梯度法终点');
    plot(x_hist_bfgs(end, 1), x_hist_bfgs(end, 2), 'k*', 'MarkerSize', 15, 'LineWidth', 2, 'DisplayName', 'BFGS法终点');
    
    xlabel('x_1');
    ylabel('x_2');
    legend('Location', 'best');
    grid on;
    % axis equal; % 建议注释掉
end