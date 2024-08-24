function [a, fault] = mom(g, d)
    % Declarations
    zz = 1.0e-5; vv = 1.0e-8; limit = 500.0;
    rttwo = 1.414213562; rrtpi = 0.5641895835; expa = 80.0; expb = 23.7;
    zero = 0.0; quart = 0.25; half = 0.5; p75 = 0.75; one = 1.0; two = 2.0; three = 3.0;

    % Initialize vectors
    a = zeros(1, 6);
    b = zeros(1, 6);
    c = zeros(1, 6);

    % Assign values
    w = g / d;
    e = exp(w) + one;
    r = rttwo / d;
    h = p75;
    if d < three
        h = quart * d;
    end
    k = 1.0;

    % Main code starts here
    fault = 0.0;
    for i = 1:6
        c(i) = zero;
    end

    % Trial value of h
    if w > expa
        fault = 1.0;
        return;
    end

    % Start of outer loop
    while true
        k = k + 1.0;
        if k > limit
            fault = 1.0;
            return;
        end
        for i = 1:6
            c(i) = a(i);
        end

        % No convergence yet - try smaller h
        h = half * h;

        % Start inner loop to evaluate infinite series
        t = w;
        u = t;
        y = h * h;
        x = two * y;
        a(1) = one / e;
        for i = 2:6
            a(i) = a(i - 1) / e;
        end
        v = y;
        f = r * h;
        m = 0.0;
        while true
            m = m + 1.0;
            if m > limit
                fault = 1.0;
                return;
            end
            for i = 1:6
                b(i) = a(i);
            end
            u = u - f;
            z = one;
            if u > (-expb)
                z = exp(u) + z;
            end
            t = t + f;
            if t > expb
                l = one;
            else
                l = zero;
            end
            if l == zero
                s = exp(t) + one;
            end
            p = exp(-v);
            q = p;
            for i = 1:6
                aa = a(i);
                p = p / z;
                ab = aa;
                aa = aa + p;
                if aa == ab
                    break;
                end
                if l == one
                    a(i) = aa;
                else
                    q = q / s;
                    ab = aa;
                    aa = aa + q;
                    if aa == ab
                        l = one;
                    else
                        l = zero;
                    end
                    a(i) = aa;
                end
            end
            y = y + x;
            v = v + y;
            if all(abs((a - b) ./ a) < vv)
                break;
            end
        end

        % End of inner loop
        v = rrtpi * h;
        for i = 1:6
            a(i) = v * a(i);
        end
        if all(abs((a - c) ./ a) < zz)
            break;
        end
    end
end