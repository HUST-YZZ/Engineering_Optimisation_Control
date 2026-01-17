function parameter_sensitivity_study()
    s_values = [0.5, 0.7, 0.9];
    results = [];
    c1 = 1.0; 
    c2 = 0.124;
    
    % 目标函数: 重量 w = c1 * x1 * sqrt(1 + x2^2)
    % 约束函数: 应力 sigma <= 100 (归一化处理)
    
    prob_func = @(x) truss_details(x, c1, c2);
    i = 1;

    for s = s_values
        % 注意：这里需要修改你的 update_asymptotes 函数，使其接收 s 作为参数
        [~, f_opt, history{i}] = my_mma_solver(prob_func, [3; 1], [0.2; 0.1], [4.0; 1.6], s);
        iters = size(history{i}.x, 2);
        results = [results; s, iters, f_opt];
        i = i+1;

    end
    visualize_mma_convergence(history{1},history{2},history{3});
    
    T = table(results(:,1), results(:,2), results(:,3), ...
        'VariableNames', {'Scaling_Factor_s', 'Iterations', 'Final_Obj'});
    disp('--- MMA 参数敏感性分析结果 ---');
    disp(T);
end

function visualize_mma_convergence(history1, history2, history3)
    % 1. 创建画布
    figure('Color', 'w', 'Name', 'Convergence Comparison', 'Position', [200, 200, 800, 500]);
    hold on;

    % 定义颜色和线型
    colors = {'b', 'r', [0, 0.5, 0]}; % 蓝, 红, 深绿
    histories = {history1, history2, history3};
    labels = {'Case 1', 'Case 2', 'Case 3'}; % 您可以根据实际情况修改标签名

    % 2. 循环绘制三条曲线
    for i = 1:3
        h = histories{i};
        obj_values = h.f0; % 提取目标函数值
        iters = 0:length(obj_values)-1; % 迭代次数
        
        % 绘制曲线
        plot(iters, obj_values, 'Color', colors{i}, 'LineWidth', 2, ...
            'DisplayName', labels{i});
        
        % 突出显示每条曲线的终点（最优值）
        plot(iters(end), obj_values(end), 'o', 'MarkerEdgeColor', colors{i}, ...
            'MarkerFaceColor', colors{i}, 'MarkerSize', 6, 'HandleVisibility', 'off');
    end

    % 3. 图形修饰
    title('Convergence Comparison of Multiple Cases', 'FontSize', 12);
    xlabel('Iteration Number', 'FontSize', 11);
    ylabel('Objective Value (Weight)', 'FontSize', 11);
    
    grid on;
    % 如果数值差异巨大，建议取消下面一行的注释使用对数坐标
    % set(gca, 'YScale', 'log'); 

    % 4. 添加图例
    legend('show', 'Location', 'northeast');

    hold off;
end
