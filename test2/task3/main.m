% 主程序：固定步长优化方法比较
clear; clc; close all;

% 基本参数设置
epsilon = 1e-5;          % 终止条件
max_iter = 5000;         % 最大迭代次数
max_iter_trial = 1000;   % 步长测试的最大迭代次数
x0 = [0.5; 4];           % 初始点

fprintf('========================================\n');
fprintf('固定步长优化方法比较\n');
fprintf('目标函数: Freudenstein and Roth Function\n');
fprintf('初始点: [%.2f, %.2f]\n', x0(1), x0(2));
fprintf('终止条件: ||∇f|| <= %.0e\n', epsilon);
fprintf('最大迭代次数: %d\n', max_iter);
fprintf('========================================\n\n');

% 步骤1：寻找最佳步长
[best_alpha_nag, best_alpha_adam] = find_optimal_stepsize(...
    @freudenstein_roth, x0, epsilon, max_iter_trial);

% 步骤2：使用最佳步长运行完整优化
fprintf('\n=== 完整优化运行 ===\n');

% NAG参数
beta_nag = 0.9;

% Adam参数
beta1_adam = 0.9;
beta2_adam = 0.999;
epsilon_alg = 1e-8;

% NAG法求解
fprintf('\n--- NAG法 (α=%.4f) ---\n', best_alpha_nag);
tic;
[x_nag, grad_norm_nag, count_nag] = nag_method(...
    @freudenstein_roth, x0, best_alpha_nag, beta_nag, epsilon, max_iter);
time_nag = toc;

fprintf('迭代次数: %d\n', size(x_nag, 1)-1);
fprintf('函数调用次数: %d\n', count_nag);
fprintf('计算时间: %.4f 秒\n', time_nag);
fprintf('最终点: [%.6f, %.6f]\n', x_nag(end, 1), x_nag(end, 2));
fprintf('最终梯度范数: %.6e\n', grad_norm_nag(end));
fprintf('最终函数值: %.6f\n\n', freudenstein_roth(x_nag(end, :)'));

% Adam法求解
fprintf('\n--- Adam法 (α=%.4f) ---\n', best_alpha_adam);
tic;
[x_adam, grad_norm_adam, count_adam] = adam_method(...
    @freudenstein_roth, x0, best_alpha_adam, beta1_adam, beta2_adam, ...
    epsilon_alg, epsilon, max_iter);
time_adam = toc;

fprintf('迭代次数: %d\n', size(x_adam, 1)-1);
fprintf('函数调用次数: %d\n', count_adam);
fprintf('计算时间: %.4f 秒\n', time_adam);
fprintf('最终点: [%.6f, %.6f]\n', x_adam(end, 1), x_adam(end, 2));
fprintf('最终梯度范数: %.6e\n', grad_norm_adam(end));
fprintf('最终函数值: %.6f\n\n', freudenstein_roth(x_adam(end, :)'));

% 步骤3：性能比较
fprintf('\n=== 性能比较 ===\n');
fprintf('方法\t\t迭代次数\t函数调用\t计算时间(秒)\t最终梯度范数\n');
fprintf('NAG\t\t%d\t\t%d\t\t%.4f\t\t%.2e\n', ...
    size(x_nag,1)-1, count_nag, time_nag, grad_norm_nag(end));
fprintf('Adam\t\t%d\t\t%d\t\t%.4f\t\t%.2e\n', ...
    size(x_adam,1)-1, count_adam, time_adam, grad_norm_adam(end));

% 计算加速比
if size(x_adam,1) > 1  % 避免除零
    speedup_iter = (size(x_nag,1)-1) / (size(x_adam,1)-1);
    speedup_count = count_nag / count_adam;
    speedup_time = time_nag / time_adam;
    
    fprintf('\n=== 加速比 (NAG/Adam) ===\n');
    fprintf('迭代次数加速比: %.2f\n', speedup_iter);
    fprintf('函数调用加速比: %.2f\n', speedup_count);
    fprintf('计算时间加速比: %.2f\n', speedup_time);
end

% 步骤4：可视化
fprintf('\n=== 生成可视化结果 ===\n');
visualize_fixed_stepsize(@freudenstein_roth, x_nag, x_adam, ...
    grad_norm_nag, grad_norm_adam);
