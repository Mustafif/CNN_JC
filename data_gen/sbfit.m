function [gamma, delta, xlam, xi, fault] = sbfit(xbar, sigma, rtb1, b2, tol)
    % Declarations
    a1 = 0.0124;  a2 = 0.0623;  a3 = 0.4043;  a4 = 0.408;  a5 = 0.479;    a6 = 0.485;
    a7 = 0.5291;  a8 = 0.5955;  a9 = 0.626;   a10 = 0.64;  a11 = 0.7077;  a12 = 0.7466;
    a13 = 0.8;    a14 = 0.9281; a15 = 1.0614; a16 = 1.25;  a17 = 1.7973;  a18 = 1.8;
    a19 = 2.163;  a20 = 2.5;    a21 = 8.5245; a22 = 11.346;

    % Allocate space for vectors
    hmu = zeros(1, 6); deriv = zeros(1, 4); dd = zeros(1, 4);

    rb1 = abs(rtb1);
    b1 = rb1 * rb1;
    if rtb1 < 0 
        neg = 1.0;
    else
        neg = 0.0;
    end

    % Get d as a first estimate of delta
    e = b1 + 1;
    x = 0.5 * b1 + 1;
    y = abs(rb1) * sqrt(0.25 * b1 + 1);
    u = ((x + y)^(1/3));
    w = u + 1/u - 1;
    f = w^2 * (3 + w * (2 + w)) - 3;
    e = (b2 - e) / (f - e);
    if abs(rb1) > tol 
        d = 1 / sqrt(log(w));
        if d < a10 
            f = a16 * d;
        else
            f = 2 - a21 / (d * (d * (d - a19) + a22));
        end
        d = e * f + 1;
        if d < a18 
            d = a13 * (f - 1);
        else
            d = (a9 * f - a4) * (3 - f)^(-a5);
        end
    else
        f = 2;
    end

    % Get G as first estimate of gamma
    g = 0;
    if b1 < 1e-4 
        g = (a12 * d^a17 + a8) * b1^a6;
    elseif d > 1 
        u = a1;
        y = a7;
    else
        u = a2;
        y = a3;
    end
    g = b1^(u * d + y) * (a14 + d * (a15 * d - a11));

    m = 0;

    % Main iteration starts here
    while true
        m = m + 1;
        if m > 50 
            fault = 1;
            break;
        end

        % Get first six moments for latest g and d values
        [hmu, fault] = mom(g, d);
        if fault == 1 
            break;
        end
        s = hmu(2);
        h2 = hmu(3) - s;
        if h2 <= 0 
            fault = 1;
            break;
        end
        t = sqrt(h2);
        h2a = t * h2;
        h2b = h2 * h2;
        h3 = hmu(4) - hmu(1) * (3 * hmu(2) - 2 * s);
        rbet = h3 / h2a;
        h4 = hmu(5) - hmu(1) * (4 * hmu(3) - hmu(1) * (6 * hmu(2) - 3 * s));
        bet2 = h4 / h2b;

        w = g * d;
        u = d * d;

        % Get derivatives
        for j = 1:2
            for k = 1:4
                t = k;
                if j == 1 
                    s = hmu(k + 1) - hmu(k);
                else
                                        s = ((w - t) * (hmu(k) - hmu(k + 1)) + (t + 1) * (hmu(k + 1) - hmu(k + 2))) / u;
                end
                dd(k) = t * s / d;
            end
            t = 2 * hmu(1) * dd(1);
            s = hmu(1) * dd(2);
            y = dd(2) - t;
            deriv(j) = (dd(3) - 3 * (s + hmu(2) * dd(1) - t * hmu(1)) - 1.5 * h3 * y / h2) / h2a;
            deriv(j + 2) = (dd(4) - 4 * (dd(3) * hmu(1) + dd(1) * hmu(3)) + 6 * (hmu(2) * t + hmu(1) * (s - t * hmu(1))) - 2 * h4 * y / h2) / h2b;
        end
        t = 1 / (deriv(1) * deriv(4) - deriv(2) * deriv(3));
        u = (deriv(4) * (rbet - rb1) - deriv(2) * (bet2 - b2)) * t;
        y = (deriv(1) * (bet2 - b2) - deriv(3) * (rbet - rb1)) * t;

        % Form new estimates of g and d
        g = g - u;
        if (b1 == 0) || (g < 0) 
            g = 0;
        end
        d = d - y;
        if (abs(u) > 1e-4) || (abs(y) > 1e-4) 
            continue;
        end

        % End of iteration
        break;
    end

    delta = d;
    xlam = sigma / sqrt(h2);
    if neg == 1 
        gamma = -g;
        hmu(1) = 1 - hmu(1);
    else
        gamma = g;
    end
    xi = xbar - xlam * hmu(1);
end