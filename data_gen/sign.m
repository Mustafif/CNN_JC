function s = sign(x, y)
    if y >= 0
        s = abs(x);
    else
        s = -abs(x);
    end
end