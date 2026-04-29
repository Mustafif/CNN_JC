function write_csv_compat(data, filename)
% write_csv_compat - Write cell array to CSV file (compatible with MATLAB and Octave)
%
% Inputs:
%   data     - Cell array to write
%   filename - Output filename

    % Check if we're in MATLAB or Octave
    is_octave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
    
    if is_octave
        % Octave: use csvwrite or manual writing
        fid = fopen(filename, 'w');
        if fid == -1
            error('Cannot open file: %s', filename);
        end
        
        for i = 1:size(data, 1)
            for j = 1:size(data, 2)
                val = data{i, j};
                if ischar(val)
                    fprintf(fid, '%s', val);
                elseif isnumeric(val)
                    fprintf(fid, '%.10g', val);
                else
                    fprintf(fid, '%s', char(val));
                end
                
                if j < size(data, 2)
                    fprintf(fid, ',');
                end
            end
            fprintf(fid, '\n');
        end
        
        fclose(fid);
    else
        % MATLAB: use writecell
        writecell(data, filename);
    end
end
