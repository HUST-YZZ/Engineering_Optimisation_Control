function [p, q, r] = generate_mma_approx(x, f0, df0, f, df, L, U)
    % i=0 是目标函数，i=1..m 是约束
    % 这里演示简化的逻辑
    % p_ij = (Uj - xj)^2 * max(0, df_i/dx_j)
    % q_ij = (xj - Lj)^2 * max(0, -df_i/dx_j)
    all_f = [f0; f];
    all_df = [df0'; df]; % 每一行是一个函数的梯度
    [M, N] = size(all_df);
    
    p = zeros(M, N); q = zeros(M, N);
    for i = 1:M
        for j = 1:N
            if all_df(i,j) > 0
                p(i,j) = (U(j) - x(j))^2 * all_df(i,j);
                q(i,j) = 0;
            else
                p(i,j) = 0;
                q(i,j) = -(x(j) - L(j))^2 * all_df(i,j);
            end
        end
    end
    % 计算 r (常数项)
    r = all_f - sum(p./(U'-x') + q./(x'-L'), 2);
end