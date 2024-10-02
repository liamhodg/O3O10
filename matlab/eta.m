function y = eta(x)
    % Load 'eta_mat' and 'T_alpha' from 'eta_zeta.mat' in a single operation
    data = load('eta_zeta.mat', 'eta_mat', 'T_alpha');
    eta_mat = data.eta_mat;
    t_alpha = data.T_alpha;

    % Ensure x is a column vector
    x_v = x(:);
    
    % Preallocate y
    y = zeros(3, numel(x));

    % Evaluate the Chebyshev polynomials for each row of eta_mat
    for i = 1:3
        y(i, :) = chebyshev(eta_mat(i, :), x_v, 0, t_alpha);
    end
end
