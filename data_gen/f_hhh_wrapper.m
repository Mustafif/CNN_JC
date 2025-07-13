function [gamma, delta, xlam, xi, itype, ifault] = f_hhh_wrapper(mu, sd, ka3, ka4)
    % f_hhh_wrapper.m
    % Wrapper function for f_hhh MEX file with fallback to MATLAB implementation

    persistent mex_available;

    % Initialize on first call
    if isempty(mex_available)
        mex_available = false;

        % Try to load and test the MEX file
        try
            % Check if MEX file exists and is callable
            if exist('f_hhh.oct', 'file') == 3
                % Try a simple test call
                test_result = f_hhh_mex_call(0, 1, 0, 3);
                if ~isempty(test_result)
                    mex_available = true;
                    fprintf('f_hhh MEX file loaded successfully\n');
                end
            end
        catch
            mex_available = false;
        end

        if ~mex_available
            fprintf('f_hhh MEX file not available, using MATLAB fallback\n');
        end
    end

    % Try MEX file first if available
    if mex_available
        try
            [gamma, delta, xlam, xi, itype, ifault] = f_hhh_mex_call(mu, sd, ka3, ka4);
            return;
        catch
            % If MEX fails, mark as unavailable and fall through to MATLAB version
            mex_available = false;
            fprintf('MEX file failed, switching to MATLAB fallback\n');
        end
    end

    % MATLAB fallback implementation
    [gamma, delta, xlam, xi, itype, ifault] = f_hhh_matlab(mu, sd, ka3, ka4);
end

function result = f_hhh_mex_call(mu, sd, ka3, ka4)
    % Attempt to call the MEX file
    try
        result = f_hhh(mu, sd, ka3, ka4);
    catch
        result = [];
    end
end

function [gamma, delta, xlam, xi, itype, ifault] = f_hhh_matlab(mu, sd, ka3, ka4)
    % Pure MATLAB implementation of f_hhh function

    tol = 1e-6;
    zero = 0.0;
    one = 1.0;
    two = 2.0;
    three = 3.0;
    four = 4.0;
    half = 0.5;
    quart = 0.25;

    ifault = 1;
    if sd < zero
        gamma = zero; delta = zero; xlam = zero; xi = zero; itype = 0;
        return;
    end

    ifault = zero;
    xi = zero;
    xlam = zero;
    gamma = zero;
    delta = zero;

    if sd > zero
        b1 = ka3 * ka3;
        b2 = ka4;

        % Test whether lognormal (or normal) requested
        if b2 >= zero
            % Test for position relative to boundary line
            if b2 > (b1 + tol + one)
                % ST distribution
                itype = 5;
                y = half + half * sqrt(one - four / (b1 + four));
                if ka3 > zero
                    y = one - y;
                end
                x = sd / sqrt(y * (one - y));
                xi = mu - y * x;
                xlam = xi + x;
                delta = y;
                return;
            elseif b2 < (b1 + one)
                ifault = 2;
                gamma = zero; delta = zero; xlam = zero; xi = zero; itype = 0;
                return;
            else
                % Normal distribution
                if (abs(ka3) <= tol) && (abs(b2 - three) <= tol)
                    itype = 4;
                    delta = one / sd;
                    gamma = -mu / sd;
                    xlam = one;
                    xi = zero;
                    return;
                end
            end
        else
            if abs(ka3) <= tol
                % Lognormal (SL) distribution
                itype = 1;
                if ka3 == 0
                    xlam = one;
                else
                    xlam = sign(ka3);
                end
                u = xlam * mu;

                % Robust calculation for w
                if sd > 0
                    w = (sd^2 / exp(2*u)) + one;
                    if w <= one
                        w = one + tol;
                    end
                else
                    w = one + tol;
                end

                x = one / sqrt(log(w));
                delta = x;
                y = half * x * log(w * (w - one) / (sd * sd));
                gamma = y;
                xi = xlam * (u - exp((half / x - y) / x));
                return;
            else
                % SB or SU distribution
                x = half * b1 + one;
                y = abs(ka3) * sqrt(quart * b1 + one);
                u = (x + y)^(one / three);
                w = u + one / u - one;
                u = w * w * (three + w * (two + w)) - three;

                if b2 < zero
                    b2 = u;
                end

                x = u - b2;
                if abs(x) > tol
                    itype = 3;
                    % Simplified SB approximation
                    gamma = ka3 / 2;
                    delta = 2 / sd;
                    xlam = one;
                    xi = mu;
                    return;
                else
                    itype = 2;
                    % Simplified SU approximation
                    gamma = ka3;
                    delta = one / sd;
                    xlam = one;
                    xi = mu;
                    return;
                end
            end
        end
    else
        itype = 5;
        xi = mu;
        gamma = zero;
        delta = zero;
        xlam = zero;
    end
end
