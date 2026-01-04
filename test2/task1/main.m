% 主程序
clear; clc; close all;

% 参数设置
epsilon = 1e-5;
x0 = [0.5; 4];  % 初始点

fprintf('========================================\n');
fprintf('Freudenstein and Roth Function Optimization\n');
fprintf('Initial point: [%.2f, %.2f]\n', x0(1), x0(2));
fprintf('Tolerance: %.0e\n', epsilon);
fprintf('========================================\n\n');

% 梯度法求解
fprintf('--- Gradient Method ---\n');
[x_grad, grad_norm_grad, count_grad] = gradient_method(@freudenstein_roth, x0, epsilon);
fprintf('Iterations: %d\n', size(x_grad, 1)-1);
fprintf('Function evaluations: %d\n', count_grad);
fprintf('Final point: [%.6f, %.6f]\n', x_grad(end, 1), x_grad(end, 2));
fprintf('Final gradient norm: %.6e\n', grad_norm_grad(end));
fprintf('Final function value: %.6f\n\n', freudenstein_roth(x_grad(end, :)'));

% 共轭梯度法求解
fprintf('--- Conjugate Gradient Method ---\n');
[x_cg, grad_norm_cg, count_cg] = conjugate_gradient_method(@freudenstein_roth, x0, epsilon);
fprintf('Iterations: %d\n', size(x_cg, 1)-1);
fprintf('Function evaluations: %d\n', count_cg);
fprintf('Final point: [%.6f, %.6f]\n', x_cg(end, 1), x_cg(end, 2));
fprintf('Final gradient norm: %.6e\n', grad_norm_cg(end));
fprintf('Final function value: %.6f\n\n', freudenstein_roth(x_cg(end, :)'));

% 比较结果
fprintf('--- Comparison ---\n');
fprintf('Method\t\tIterations\tFuncEvals\tFinalGradNorm\n');
fprintf('Gradient\t%d\t\t%d\t\t%.2e\n', ...
    size(x_grad,1)-1, count_grad, grad_norm_grad(end));
fprintf('ConjGrad\t%d\t\t%d\t\t%.2e\n', ...
    size(x_cg,1)-1, count_cg, grad_norm_cg(end));

% 可视化
visualize_optimization(@freudenstein_roth, x_grad, grad_norm_grad, count_grad, '梯度法');
visualize_optimization(@freudenstein_roth, x_cg, grad_norm_cg, count_cg, '共轭梯度法');

visualize_optimization(@freudenstein_roth, x_grad, 'Gradient Method');
visualize_optimization(@freudenstein_roth, x_cg, 'Conjugate Gradient Method');

