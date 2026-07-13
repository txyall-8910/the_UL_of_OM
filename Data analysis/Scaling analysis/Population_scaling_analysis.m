% =========================================================================
% Population Scaling Analysis
%
% Description:
% This script estimates population scaling behaviors with regard to
% the structural breaks (crossover points) and scaling regimes previously 
% identified from flow patterns. 
% =========================================================================

% Extract dimensions and initialize the dependent variable matrix
% Note: P_cleaned contains the empirical population distribution data
[m, ~] = size(P_cleaned);
[~, n] = size(x_final);
n = n + 2;
y = table2array(P_cleaned(:, 3 : n)); 

% Define output table variable names to capture comprehensive regression statistics
% for the population scaling
varNames = {'sampleID', 'type', ...
    'global_slope', 'global_intercept', 'global_r2', 'global_F', 'global_pValue', ...
    'seg_slope', 'seg_intercept', 'seg_r2', 'seg_F', 'seg_p', ...
    's_global_slope', 's_global_intercept', 's_global_r2', 's_global_F', 's_global_pValue', ...
    'seg_slope1', 'seg_intercept1', 'seg_r2_1', 'seg_F1', 'seg_p1', ...
    'seg_slope2', 'seg_intercept2', 'seg_r2_2', 'seg_F2', 'seg_p2', ...
    'seg_slope3', 'seg_intercept3', 'seg_r2_3', 'seg_F3', 'seg_p3'};

% Preallocate an empty table with explicitly defined variable types to ensure 
% computational efficiency and data integrity during the main loop
resultTable = table('Size', [m, length(varNames)], ...
    'VariableTypes', [{'double'}, {'categorical'}, repmat({'double'}, 1, 5), ...
                      repmat({'cell'}, 1, 5), repmat({'double'}, 1, 20)], ...
    'VariableNames', varNames);

[y_final] = deal(nan(m, n));

% Standardize input matrices by filtering out missing population observations (NaNs) 
% across the spatial axis for all samples
for i = 1 : m
    non_empty_indices = find(~isnan(x_final(i, :)));
    y_final(i,1:length(non_empty_indices)) = y(i, non_empty_indices);
end

coefficients = zeros(m,5); 

% =========================================================================
% Main Processing Loop: Projecting Flow Regimes onto Population Data
% =========================================================================
for i = 1 : m
    
    % Verify if the current sample has a valid flow-derived model classification 
    % (stored in the reference table T from the previous flow scaling regime analysis)
    if any(T.sampleID == i)
        
        rowIndices = find(T.sampleID == i);
        current_type = char(T.type(rowIndices)); % Flow-derived optimal model type
        
        % Isolate the valid spatial sequence and compute the cumulative population
        valid_idx = ~isnan(y_final(i,:));
        x_full = x_final(i,valid_idx)';
        y_full = cumsum(y_final(i,valid_idx)');
        
        % Identify the effective spatial origin (first non-zero observation)
        startp = T.first_nonzero_idx(rowIndices);
        
        % Trim the arrays to begin exactly at the effective spatial origin
        x_current = x_full(startp:end);
        y_current = y_full(startp:end);
        n_samples = length(x_current);  % Define the effective sample size      
        
        % Populate baseline identification fields
        resultTable.sampleID(i) = i;
        resultTable.type(i) = categorical({current_type});

        % -----------------------------------------------------------------
        % (1) Baseline Global Model
        % Evaluates the baseline population scaling assuming a single, invariant 
        % exponent governs the entire radial profile.
        % -----------------------------------------------------------------
        if strcmp(string(current_type), 'global')
            log_y = log(y_current(1:n_samples));
            log_x = log(x_current(1:n_samples));
            X = [ones(n_samples, 1), log_x];
            
            % Fit ordinary least squares (OLS) regression in log-log space
            [beta, ~, ~, ~, stats] = regress(log_y, X);
            
            resultTable.global_slope(i) = beta(2);
            resultTable.global_intercept(i) = beta(1);
            resultTable.global_r2(i) = stats(1);
            resultTable.global_F(i) = stats(2);
            resultTable.global_pValue(i) = stats(3);

        % -----------------------------------------------------------------
        % (2) Segmented Regression (Flow-Dictated Breakpoints)
        % Applies the exact structural breakpoints identified from flow scaling 
        % to partition the population scaling patterns. 
        % -----------------------------------------------------------------
        elseif strcmp(string(current_type), 'segmented')
            % Retrieve the optimal segment lengths dictated by flow dynamics
            sl = T.seg_length_1(rowIndices);
            s2 = T.seg_length_2(rowIndices);
            s3 = T.seg_length_3(rowIndices);
            
            % Reconstruct exact breakpoint locations based on flow-derived lengths
            bp = [];
            if ~isnan(sl) && sl > 0
                bp(1) = sl;
                if ~isnan(s2) && s2 > 0
                    bp(2) = bp(1) + s2;
                    if ~isnan(s3) && s3 > 0
                        bp(3) = bp(2) + s3;
                    end
                end
            end
            
            % Ensure reconstructed breakpoints do not exceed the empirical data bounds
            bp = bp(bp < n_samples);
            
            % Define the topological boundaries for each population scaling regime
            segment_edges = [1, bp+1, n_samples+1];
            num_segments = length(segment_edges)-1;
            
            % Initialize cell arrays to store regime-specific population parameters
            seg_slope = cell(1, num_segments);
            seg_intercept = cell(1, num_segments);
            seg_r2 = cell(1, num_segments);
            seg_F = cell(1, num_segments);
            seg_p = cell(1, num_segments);
            
            % Execute independent OLS regressions for each flow-dictated regime
            for s = 1:num_segments
                start_idx = segment_edges(s);
                end_idx = segment_edges(s+1)-1;
                
                % Enforce minimum data requirements for valid regression inference
                if (end_idx - start_idx) < 1
                    seg_slope{s} = NaN;
                    seg_intercept{s} = NaN;
                    seg_r2{s} = NaN;
                    seg_F{s} = NaN;
                    seg_p{s} = NaN;
                    continue;
                end
                
                % Extract regime-specific population data and transform to log-log space
                seg_x = x_current(start_idx:end_idx);
                seg_y = y_current(start_idx:end_idx);
                
                log_xs = log(seg_x);
                log_ys = log(seg_y);
                
                Xs = [ones(length(log_xs),1), log_xs];
                [betas, ~, ~, ~, stats_s] = regress(log_ys, Xs);
                
                % Store regime-specific population scaling exponents and fit statistics
                seg_slope{s} = betas(2);
                seg_intercept{s} = betas(1);
                seg_r2{s} = stats_s(1);
                seg_F{s} = stats_s(2);
                seg_p{s} = stats_s(3);
            end
            
            % Aggregate multi-regime results into cell arrays for flexible downstream use
            resultTable.seg_slope{i} = seg_slope;
            resultTable.seg_intercept{i} = seg_intercept;
            resultTable.seg_r2{i} = seg_r2;
            resultTable.seg_F{i} = seg_F;
            resultTable.seg_p{i} = seg_p;
            
            % Unpack cell arrays into fixed columns (supporting up to 3 distinct regimes)
            for k = 1:3
                if length(seg_slope) >= k && ~isempty(seg_slope{k})
                    resultTable.(['seg_slope' num2str(k)])(i) = seg_slope{k};
                else
                    resultTable.(['seg_slope' num2str(k)])(i) = NaN;
                end
                
                if length(seg_intercept) >= k && ~isempty(seg_intercept{k})
                    resultTable.(['seg_intercept' num2str(k)])(i) = seg_intercept{k};
                else
                    resultTable.(['seg_intercept' num2str(k)])(i) = NaN;
                end
                
                if length(seg_r2) >= k && ~isempty(seg_r2{k})
                    resultTable.(['seg_r2_' num2str(k)])(i) = seg_r2{k};
                else
                    resultTable.(['seg_r2_' num2str(k)])(i) = NaN;
                end
                
                if length(seg_F) >= k && ~isempty(seg_F{k})
                    resultTable.(['seg_F' num2str(k)])(i) = seg_F{k};
                else
                    resultTable.(['seg_F' num2str(k)])(i) = NaN;
                end
                
                if length(seg_p) >= k && ~isempty(seg_p{k})
                    resultTable.(['seg_p' num2str(k)])(i) = seg_p{k};
                else
                    resultTable.(['seg_p' num2str(k)])(i) = NaN;
                end
            end

        % -----------------------------------------------------------------
        % (3) Trimmed Linear Regression (Flow-Dictated Trimming)
        % Fits a robust linear model to the population data by removing the 
        % exact boundary observations (tails) that were identified as noise 
        % or edge effects in the flow dynamics analysis.
        % -----------------------------------------------------------------
        elseif strcmp(string(current_type), 's-global')
            % Retrieve trimming thresholds dictated by the flow distribution tails
            bp1 = T.sglobal_bp_front(rowIndices);
            bp2 = T.sglobal_bp_back(rowIndices);
            
            % Calculate the effective index range for the population data after trimming
            idx1 = bp1+1;
            idx2 = n_samples-bp2;
            
            % Skip if trimming results in an invalid or inverted sequence
            if idx2 < idx1, continue; end
            
            log_y = log(y_current(idx1:idx2));
            log_x = log(x_current(idx1:idx2));
            X = [ones(length(log_x),1), log_x];
            
            % Perform OLS regression on the trimmed population dataset
            [beta, ~, ~, ~, stats] = regress(log_y, X);
            
            resultTable.s_global_slope(i) = beta(2);
            resultTable.s_global_intercept(i) = beta(1);
            resultTable.s_global_r2(i) = stats(1);
            resultTable.s_global_F(i) = stats(2);
            resultTable.s_global_pValue(i) = stats(3);
        end
    else
        % Handle missing or unclassified samples by assigning empty identifiers
        resultTable.sampleID(i) = i;
        resultTable.type(i) = categorical({''});
    end
end