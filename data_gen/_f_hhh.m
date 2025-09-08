function [gamma, delta, xlam, xi, itype, ifault] = f_hhh(mu, sd, ka3, ka4)
    % f_hhh.m
    % Johnson curve fitting function
    % This function determines Johnson curve parameters for given moments

    [gamma, delta, xlam, xi, itype, ifault] = f_hhh_wrapper(mu, sd, ka3, ka4);
end
