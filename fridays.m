function thirdFridays = findThirdFridays(year)
    thirdFridays = datetime(year, 1, 1):calmonths(1):datetime(year, 12, 1);
    for month = 1:12
        firstDay = datetime(year, month, 1);
        firstFriday = firstDay + mod(6 - weekday(firstDay), 7); % Calculate the first Friday
        thirdFridays(month) = firstFriday + days(14); % Calculate the third Friday
    end
end

function fridays = findFridays(year)
    fridays = cell(3, 12); % Create a cell array for first, second, and third Fridays
    for month = 1:12
        firstDay = datetime(year, month, 1);
        firstFriday = firstDay + days(mod(6 - weekday(firstDay), 7)); % Calculate the first Friday
        fridays{1, month} = firstFriday;        % First Friday
        fridays{2, month} = firstFriday + days(7);  % Second Friday
        fridays{3, month} = firstFriday + days(14); % Third Friday
    end
end

function saveFridaysToCSV(year)
    % Preallocate arrays for formatted date strings
    formattedDates = cell(4, 12); % Create a cell array for formatted Friday and expiration dates

    for month = 1:12
        firstDay = datetime(year, month, 1);
        firstFriday = firstDay + days(mod(6 - weekday(firstDay), 7)); % Calculate the first Friday

        % Calculate Fridays and their expiration dates
        fridays = [
            firstFriday;              % First Friday
            firstFriday + days(7);    % Second Friday
            firstFriday + days(14);    % Third Friday
            firstFriday + days(21)     % Fourth Friday
        ];
        
        expirationDates = [
            fridays(1) + days(7);      % Expiration of 1st Friday
            fridays(2) + days(7);      % Expiration of 2nd Friday
            fridays(3) + calmonths(1); % Expiration of 3rd Friday (next month)
            fridays(4) + days(7)       % Expiration of 4th Friday
        ];

        % Format the output as <Friday Date> - <Expiration Date>
        for i = 1:4
            formattedDates{i, month} = sprintf('%s - %s', datetime(fridays(i), 'dd-mmm-yyyy'), datetime(expirationDates(i), 'dd-mmm-yyyy'));
        end
    end

    % Create a table with consistent rows
    monthNumbers = (1:12)'; % Month numbers
    formattedTable = table(monthNumbers, formattedDates', 'VariableNames', {'Month', 'Period'});

    % Write to CSV file
    writetable(formattedTable, 'FridaysAndExpirations.csv');
    
    disp('Fridays and expiration dates saved to FridaysAndExpirations.csv');
end

% usage
third = findThirdFridays(2024);
friday = findFridays(2024);

saveFridaysToCSV(2024);