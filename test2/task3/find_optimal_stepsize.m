function [best_alpha_nag, best_alpha_adam] = find_optimal_stepsize(fun, x0, epsilon, max_iter_trial)
    % 寻找最优步长
    % 通过简单的网格搜索找到能使算法收敛的步长范围
    
    fprintf('\n=== 步长选择实验 ===\n');
    
    % NAG参数范围
    alpha_range_nag = logspace(-4, 0, 20);  % 10^-4 到 1
    beta_nag = 0.9;  % 固定动量参数
    
    % Adam参数范围
    alpha_range_adam = logspace(-4, 0, 20);
    beta1_adam = 0.9;
    beta2_adam = 0.999;
    epsilon_alg = 1e-8;
    
    % 测试NAG
    fprintf('\n--- NAG步长测试 ---\n');
    nag_results = [];
    
    for i = 1:length(alpha_range_nag)
        alpha = alpha_range_nag(i);
        
        try
            [~, grad_norm_hist, ~] = nag_method(fun, x0, alpha, beta_nag, epsilon, max_iter_trial);
            
            if grad_norm_hist(end) <= epsilon
                converged = true;
                iterations = length(grad_norm_hist) - 1;
            else
                converged = false;
                iterations = max_iter_trial;
            end
            
            nag_results = [nag_results; alpha, converged, iterations, grad_norm_hist(end)];
            fprintf('α=%.2e: %s, 迭代=%d, 最终梯度=%.2e\n', ...
                alpha, converged_str(converged), iterations, grad_norm_hist(end));
        catch
            fprintf('α=%.2e: 运行失败\n', alpha);
        end
    end
    
    % 选择最佳NAG步长
    if ~isempty(nag_results)
        converged_idx = nag_results(:, 2) == 1;
        if any(converged_idx)
            converged_results = nag_results(converged_idx, :);
            [~, min_idx] = min(converged_results(:, 3));  % 选择迭代次数最少的
            best_alpha_nag = converged_results(min_idx, 1);
        else
            % 如果没有收敛的，选择最终梯度最小的
            [~, min_idx] = min(nag_results(:, 4));
            best_alpha_nag = nag_results(min_idx, 1);
        end
    else
        best_alpha_nag = 0.01;  % 默认值
    end
    
    % 测试Adam
    fprintf('\n--- Adam步长测试 ---\n');
    adam_results = [];
    
    for i = 1:length(alpha_range_adam)
        alpha = alpha_range_adam(i);
        
        try
            [~, grad_norm_hist, ~] = adam_method(fun, x0, alpha, beta1_adam, beta2_adam, epsilon_alg, epsilon, max_iter_trial);
            
            if grad_norm_hist(end) <= epsilon
                converged = true;
                iterations = length(grad_norm_hist) - 1;
            else
                converged = false;
                iterations = max_iter_trial;
            end
            
            adam_results = [adam_results; alpha, converged, iterations, grad_norm_hist(end)];
            fprintf('α=%.2e: %s, 迭代=%d, 最终梯度=%.2e\n', ...
                alpha, converged_str(converged), iterations, grad_norm_hist(end));
        catch
            fprintf('α=%.2e: 运行失败\n', alpha);
        end
    end
    
    % 选择最佳Adam步长
    if ~isempty(adam_results)
        converged_idx = adam_results(:, 2) == 1;
        if any(converged_idx)
            converged_results = adam_results(converged_idx, :);
            [~, min_idx] = min(converged_results(:, 3));  % 选择迭代次数最少的
            best_alpha_adam = converged_results(min_idx, 1);
        else
            % 如果没有收敛的，选择最终梯度最小的
            [~, min_idx] = min(adam_results(:, 4));
            best_alpha_adam = adam_results(min_idx, 1);
        end
    else
        best_alpha_adam = 0.001;  % 默认值
    end
    
    fprintf('\n最佳步长选择:\n');
    fprintf('NAG: α = %.4f\n', best_alpha_nag);
    fprintf('Adam: α = %.4f\n', best_alpha_adam);
end

function str = converged_str(converged)
    if converged
        str = '收敛';
    else
        str = '未收敛';
    end
end