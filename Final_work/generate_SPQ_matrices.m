function [S, P, Q] = generate_SPQ_matrices(n)
    % 根据试题第2页公式生成 S, P, Q 矩阵
    % 输入: n (变量维度)
    
    S = zeros(n, n);
    P = zeros(n, n);
    Q = zeros(n, n);
    
    ln_n = log(n); % 公式中的 ln n
    
    for i = 1:n
        for j = 1:n
            % 1. 计算 alpha_ij
            alpha_ij = (i + j - 2) / (2 * n - 2);
            
            % 分母项 (1 + |i-j|) * ln n
            denom = (1 + abs(i - j)) * ln_n;
            
            % 2. 计算 s_ij
            S(i, j) = (2 + sin(4 * pi * alpha_ij)) / denom;
            
            % 3. 计算 p_ij
            P(i, j) = (1 + 2 * alpha_ij) / denom;
            
            % 4. 计算 q_ij
            Q(i, j) = (3 - 2 * alpha_ij) / denom;
        end
    end
end