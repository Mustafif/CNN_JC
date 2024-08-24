function [gamma, delta, xlam, xi] = sufit(xbar, sd, rb1, b2, tol)
    b1 = rb1 * rb1;
    b3 = b2 - 3;

    % W is first estimate of exp(delta ** (-2))
    w = sqrt(sqrt(2 * b2 - 2.8 * b1 - 2) - 1);
    if abs(rb1) > tol 
        % Johnson iteration (using y for his M)
        while true
            w1 = w + 1;
            wm1 = w - 1;
            z = w1 * b3;
            v = w * (6 + w * (3 + w));
            a = 8 * (wm1 * (3 + w * (7 + v)) - z);
            b = 16 * (wm1 * (6 + v) - b3);
            y = (sqrt(a * a - 2 * b * (wm1 * (3 + w * (9 + w * (10 + v))) - 2 * w1 * z)) - a) / b;
            z = y * wm1 * ((4 * (w + 2) * y + 3 * w1 * w1) ^ 2) / (2 * (2 * y + w1) ^ 3);
            v = w * w;
            w = sqrt(sqrt(1 - 2 * (1.5 - b2 + (b1 * (b2 - 1.5 - v * (1 + 0.5 * v))) / z)) - 1);
            if abs(b1 - z) < tol 
                break;
            end
        end
        y = y / w;
        y = log(sqrt(y) + sqrt(y + 1));
        if rb1 > 0 
            y = -y;
        end
    else
        % Symmetrical case - results are known
        y = 0;
    end

    % End of iteration
    x = sqrt(1 / log(w));
    delta = x;
    gamma = y * x;
    y = exp(y);
    z = y * y;
    x = sd / sqrt(0.5 * (w - 1) * (0.5 * w * (z + 1 / z) + 1));
    xlam = x;
    xi = (0.5 * sqrt(w) * (y - 1 / y)) * x + xbar;
end