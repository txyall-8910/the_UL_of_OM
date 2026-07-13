% =========================================================================
% Data Processing: Dynamic Truncation of Distance-Decay Curves
%
% Description:
% Zero flows at certain distance rings can occur for two reasons: 
% (1) local geographical anomalies (e.g., water bodies or parks) causing 
% temporary "sampling zeros", or 
% (2) reaching the absolute boundary of the spatial unit's commuting shed, 
% resulting in continuous "structural zeros". 
% 
% Including structural zeros in downstream distance-decay curve fitting 
% (e.g., exponential or power-law models) introduces severe downward bias. 
% This algorithm uses a dynamic look-ahead window to distinguish between 
% temporary anomalies and the true edge of the commuting shed, truncating 
% the data appropriately to ensure unbiased statistical modeling.
% =========================================================================

[m, ~] = size(T_cleaned);   % m: Number of active spatial units
n = 200;                    % n: Total number of distance bins (annuli) per unit
x = (1 : n);                % Independent variable: Distance intervals
y = table2array(T_cleaned(:, 3 : 202)); % Dependent variable: Commuting flow volumes

% ========== 2. Dynamic Data Truncation Pipeline ==========
% Initialize cell arrays to store variable-length valid flow sequences
[y_processed, x_processed] = deal(cell(m, 1)); 

for i = 1 : m
    y_row = y(i, :);
    x_row = x;
    
    % Locate the first zero-flow instance in the distance-decay sequence
    first_zero = find(y_row == 0, 1);
    
    % If no zeros exist in the entire sequence, retain the full array
    if isempty(first_zero)
        y_processed{i} = y_row; 
        x_processed{i} = x_row;
        continue; 
    end
    
    % Initial truncation point: just before the first observed zero
    end_idx = first_zero - 1;
    last_nonzero = max(1, end_idx); % Ensure the index is at least 1
    
    % ---------------------------------------------------------------------
    % Dynamic Look-ahead Window Algorithm
    % Evaluates a moving window (size = 4 bins) to determine if the zero is 
    % a temporary local anomaly or the true boundary of the commuting shed.
    % ---------------------------------------------------------------------
    while true
        next_start = max(1, last_nonzero);  % Start of the look-ahead window
        next_end = min(next_start + 3, n);  % End of the window (preventing out-of-bounds)
        
        if next_start > next_end
            break; % Exit if the window becomes invalid
        end
        
        window = y_row(next_start : next_end);
        
        % Condition: If at least 2 non-zero flows exist within this window,
        % the previous zero is deemed a local anomaly. Extend the valid boundary.
        if sum(window ~= 0) >= 2
            % Update the boundary to the last non-zero index within the current window
            last_nonzero = next_start - 1 + find(window ~= 0, 1, 'last');
            end_idx = last_nonzero;
        else
            % If fewer than 2 non-zero flows exist, we assume the true 
            % boundary of the commuting shed has been reached. Stop extending.
            break;
        end
        
        % Terminate if the window reaches the maximum spatial extent
        if next_end >= n, break; end
    end
    
    % Store the truncated, valid sequence for the current spatial unit
    y_processed{i} = y_row(1 : end_idx);
    x_processed{i} = x_row(1 : end_idx);
end

% -------------------------------------------------------------------------
% Matrix Reconstruction with NaN Padding
% 
% Downstream vectorized operations require uniform matrix dimensions. 
% Padding the truncated tails with NaN (Not-a-Number) ensures that these 
% out-of-bound regions are strictly ignored during statistical curve fitting,
% preventing them from being erroneously treated as zero values.
% -------------------------------------------------------------------------
max_len = max(cellfun(@length, y_processed)); % Find the maximum commuting shed radius
[y_final, x_final] = deal(nan(m, max_len));   % Initialize matrices with NaN

for i = 1 : m
    % Insert the valid truncated sequences into the uniform NaN matrices
    y_final(i, 1:length(y_processed{i})) = y_processed{i};
    x_final(i, 1:length(x_processed{i})) = x_processed{i};
end