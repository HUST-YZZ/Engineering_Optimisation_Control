function [L, U] = update_asymptotes(k, x, x_old, x_older, L, U, lb, ub, s)
    % s 是用户定义的缩放因子
    n = length(x);
    
    if k <= 2
        % 初始两步给一个较宽的包络
        L = x - 0.5 * (ub - lb);
        U = x + 0.5 * (ub - lb);
    else
        for j = 1:n
            % 核心逻辑：判断位移方向的符号变化
            change = (x(j) - x_old(j)) * (x_old(j) - x_older(j));
            
            if change < 0
                % 震荡：说明步子跨大了，需要让渐近线靠近，s 越小收缩越快
                L(j) = x(j) - s * (x_old(j) - L(j));
                U(j) = x(j) + s * (U(j) - x_old(j));
            elseif change > 0
                % 单调：说明方向是对的，可以迈大步，扩大渐近线距离
                L(j) = x(j) - (1/s) * (x_old(j) - L(j));
                U(j) = x(j) + (1/s) * (U(j) - x_old(j));
            end
        end
    end
    
    % 边界保护逻辑：防止分母为零或超出物理约束
    % 设置一个极小的偏移 eps
    eps_dist = 0.01 * (ub - lb);
    L = min(L, x - eps_dist);
    U = max(U, x + eps_dist);
end