function y = zeta(x)
    % Load 'zeta_mat' and 'T_omega' from 'eta_zeta.mat' in a single operation
    data = load('eta_zeta.mat', 'zeta_mat', 'T_omega');
    zeta_mat = data.zeta_mat;
    t_omega = data.T_omega;

    % Ensure x is a column vector
    x_v = x(:);
    
    % Preallocate y
    y = zeros(3, numel(x));

    % Evaluate the Chebyshev polynomials for each row of eta_mat
    for i = 1:3
        y(i, :) = chebyshev(zeta_mat(i, :), x_v, 0, t_omega);
    end
end
