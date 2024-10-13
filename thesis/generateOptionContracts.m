function contracts = generateOC(S, monthExp, weekExp)
    interval = 2.5 * (S < 50) + 5 * (S >= 50 && S < 200) + 10 * (S >= 200);
    genStrikes = @(n) [max(0, S - interval * (1:floor(n*0.4))), S, S + interval * (1:floor(n*0.4))];
    genCats = @(n) [repmat({'ITM'}, 1, floor(n*0.4)), {'ATM'}, repmat({'OTM'}, 1, floor(n*0.4))];
    ternary = @(cond, t, f) ternary_helper(cond, t, f);
    contracts = [];
    for isMonthly = [true, false]
        exps = ternary(isMonthly, monthExp, weekExp);
        strikes = genStrikes(ternary(isMonthly, 15, 5));
        cats = genCats(ternary(isMonthly, 15, 5));
        type = ternary(isMonthly, ', monthly', ', weekly');
        for exp = exps
            contracts = [contracts, struct('type', repmat({'call'}, 1, length(strikes)), ...
                                           'strike_price', num2cell(strikes), ...
                                           'expiration_date', repmat({exp}, 1, length(strikes)), ...
                                           'category', strcat(cats, type))];
        end
    end
    function out = ternary_helper(cond, t, f)
        if cond, out = t; else, out = f; end
    end
end
% Example usage
S = 100; monthExp = datetime({'2024-10-18'}); weekExp = datetime({'2024-10-11', '2024-10-04'});
contracts = generateOC(S, monthExp, weekExp);
% Output results
writetable(struct2table(contracts), 'option_contracts.csv');
cellfun(@(varargin) fprintf('Type: %s, Strike Price: %.2f, Expiration Date: %s, Category: %s\n', varargin{:}), ...
        {contracts.type}, {contracts.strike_price}, ...
        cellfun(@datetime, {contracts.expiration_date}, 'UniformOutput', false), ...
        {contracts.category});