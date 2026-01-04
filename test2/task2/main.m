    % 主程序：基于Armijo-Goldstein的优化方法比较
    
    clear; clc; close all;
    
    % 参数设置
    epsilon = 1e-5;
    x0 = [0.5; 4];  % 初始点
    
    fprintf('========================================\n');
    fprintf('基于Armijo-Goldstein的优化方法比较\n');
    fprintf('目标函数: Freudenstein and Roth Function\n');
    fprintf('初始点: [%.2f, %.2f]\n', x0(1), x0(2));
    fprintf('终止条件: ||∇f|| <= %.0e\n', epsilon);
    fprintf('========================================\n\n');
    
    % 梯度法求解
    fprintf('\n--- 梯度法 (Armijo-Goldstein) ---\n');
    tic;
    [x_grad, grad_norm_grad, count_grad] = gradient_method_armijo(@freudenstein_roth, x0, epsilon);
    time_grad = toc;
    
    fprintf('迭代次数: %d\n', size(x_grad, 1)-1);
    fprintf('函数调用次数: %d\n', count_grad);
    fprintf('计算时间: %.4f 秒\n', time_grad);
    fprintf('最终点: [%.6f, %.6f]\n', x_grad(end, 1), x_grad(end, 2));
    fprintf('最终梯度范数: %.6e\n', grad_norm_grad(end));
    fprintf('最终函数值: %.6f\n\n', freudenstein_roth(x_grad(end, :)'));
    
    % BFGS法求解
    fprintf('\n--- BFGS变尺度法 (Armijo-Goldstein) ---\n');
    tic;
    [x_bfgs, grad_norm_bfgs, count_bfgs] = bfgs_method(@freudenstein_roth, x0, epsilon);
    time_bfgs = toc;
    
    fprintf('迭代次数: %d\n', size(x_bfgs, 1)-1);
    fprintf('函数调用次数: %d\n', count_bfgs);
    fprintf('计算时间: %.4f 秒\n', time_bfgs);
    fprintf('最终点: [%.6f, %.6f]\n', x_bfgs(end, 1), x_bfgs(end, 2));
    fprintf('最终梯度范数: %.6e\n', grad_norm_bfgs(end));
    fprintf('最终函数值: %.6f\n\n', freudenstein_roth(x_bfgs(end, :)'));
    
    % 比较结果
    fprintf('\n--- 性能比较 ---\n');
    fprintf('方法\t\t迭代次数\t函数调用\t计算时间(秒)\t最终梯度范数\n');
    fprintf('梯度法\t\t%d\t\t%d\t\t%.4f\t\t%.2e\n', ...
        size(x_grad,1)-1, count_grad, time_grad, grad_norm_grad(end));
    fprintf('BFGS法\t\t%d\t\t%d\t\t%.4f\t\t%.2e\n', ...
        size(x_bfgs,1)-1, count_bfgs, time_bfgs, grad_norm_bfgs(end));
    
    % 计算加速比
    speedup_iter = (size(x_grad,1)-1) / (size(x_bfgs,1)-1);
    speedup_count = count_grad / count_bfgs;
    speedup_time = time_grad / time_bfgs;
    
    fprintf('\n--- 加速比 (梯度法/BFGS法) ---\n');
    fprintf('迭代次数加速比: %.2f\n', speedup_iter);
    fprintf('函数调用加速比: %.2f\n', speedup_count);
    fprintf('计算时间加速比: %.2f\n', speedup_time);
    
    % 可视化
    visualize_optimization_comparison(@freudenstein_roth, x_grad, x_bfgs, ...
        grad_norm_grad, grad_norm_bfgs);