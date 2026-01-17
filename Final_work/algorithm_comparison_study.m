function algorithm_comparison_study()
    x0 = [1.5; 0.5]; lb = [0.2; 0.1]; ub = [4.0; 1.6];
    c1 = 1.0; 
    c2 = 0.124;
    
    % 目标函数: 重量 w = c1 * x1 * sqrt(1 + x2^2)
    % 约束函数: 应力 sigma <= 100 (归一化处理)
    
    prob_func = @(x) truss_details(x, c1, c2);
    % 原始问题定义 (fmincon 使用)
    obj_fun = @(x) 1.0 * x(1) * sqrt(1 + x(2)^2);
    nonlcon = @(x) truss_constraints_fmincon(x);

    % 定义对比算法
    algos = {'interior-point', 'sqp', 'active-set'};
    comp_results = struct();

    % 1. 运行 MMA (作为基准)
    [~, f_mma, history_mma] = my_mma_solver(prob_func, x0, lb, ub,0.7);
    comp_results(1).Name = 'MMA';
    comp_results(1).Iters = size(history_mma.x, 2);
    comp_results(1).Obj = f_mma;

    % 2. 运行 fmincon 各类算法
    for i = 1:length(algos)
        options = optimoptions('fmincon', 'Algorithm', algos{i}, 'Display', 'off');
        tic;
        [~, f_val, ~, output] = fmincon(obj_fun, x0, [], [], [], [], lb, ub, nonlcon, options);
        time_cost = toc;
        comp_results(i+1).Name = ['fmincon-', algos{i}];
        comp_results(i+1).Iters = output.iterations;
        comp_results(i+1).Obj = f_val;
    end

    % 打印对比表
    fprintf('\n--- 算法性能对比表 (二杆桁架) ---\n');
    fprintf('%-20s | %-10s | %-10s\n', 'Algorithm', 'Iterations', 'Final Obj');
    for i = 1:length(comp_results)
        fprintf('%-20s | %-10d | %-10.6f\n', comp_results(i).Name, comp_results(i).Iters, comp_results(i).Obj);
    end
end

function [c, ceq] = truss_constraints_fmincon(x)
    c1 = 1.0; c2 = 0.124;
    fac = sqrt(1 + x(2)^2);
    s1 = c2 * fac * (8/x(1) + 1/(x(1)*x(2)));
    s2 = c2 * fac * (8/x(1) - 1/(x(1)*x(2)));
    c = [s1 - 1; s2 - 1]; ceq = [];
end