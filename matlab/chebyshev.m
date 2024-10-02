function y = chebyshev(coeff, x, a, b)
    % Function to evaluate a Chebyshev series using the Clenshaw method
    %
    % INPUTS:
    %   coeff: Coefficients of the Chebyshev series [c0, c1, c2, ..., cN]
    %   x: Value(s) at which to evaluate the Chebyshev series
    %
    % OUTPUT:
    %   y: Evaluated series value(s) at x

    % Number of coefficients
    N = length(coeff);

    % Initialize variables for Clenshaw recurrence
    b_kp2 = 0;  % b_{k+2}
    b_kp1 = 0;  % b_{k+1}

    z = 2*x/(b - a) - (b + a)/(b - a);
    z_m = z - sqrt(z.*z - 1);
    z_p = z + sqrt(z.*z - 1);
    y = 0;

    z_m_pow = 1;
    z_p_pow = 1;
    for k = 1:N
        term = coeff(k) * (z_m_pow + z_p_pow);
        y = y + 0.5*real(term);
        z_m_pow = z_m_pow.*z_m;
        z_p_pow = z_p_pow.*z_p;
    end
end
