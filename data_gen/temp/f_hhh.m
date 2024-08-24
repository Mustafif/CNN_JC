function [gamma, delta, xlam, xi, itype, ifault] = f_hhh(mu, sd, ka3, ka4)
    tol = 0.000001;
    zero = 0.0;
    one = 1.0;
    two = 2.0;
    three = 3.0;
    four = 4.0;
    half = 0.5;
    quart = 0.25;

    ifault = 1.0;
    if sd < zero
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
        fault = zero;
        
        % Test whether lognormal (or normal) requested
        if b2 >= zero
            % Test for position relative to boundary line
            if b2 > (b1 + tol + one)
                % ST distribution
                itype = 5.0;
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
                ifault = 2.0;
                return;
            else
                % Normal distribution
                if (abs(ka3) <= tol) && (abs(b2 - three) <= tol)
                    itype = 4.0;
                    delta = one / sd;
                    gamma = -mu / sd;
                    xlam = one;
                    return;
                end
            end
        else
            if abs(ka3) <= tol
                % Lognormal (SL) distribution
                itype = 1.0;
                xlam = sign(one, ka3);
                u = xlam * mu;
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
                if (b2 < zero) || (fault == one)
                    b2 = u;
                end
                x = u - b2;
                if abs(x) > tol
                    itype = 3.0;
                    sbfit(mu, sd, ka3, b2, tol, gamma, delta, xlam, xi, fault);
                    if fault == zero
                        return;
                    end
                else
                    itype = 2.0;
                    sufit(mu, sd, ka3, b2, tol, gamma, delta, xlam, xi);
                    return;
                end
            end
        end
    else
        itype = 5.0;
        xi = mu;
    end
end