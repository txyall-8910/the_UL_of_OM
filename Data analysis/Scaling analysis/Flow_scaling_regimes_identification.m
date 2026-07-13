% =========================================================================
% Identification of Flow Scaling Regimes
%
% Distribution:
% This script implements a hierarchical, statistically driven algorithm to 
% objectively identify these deviations (crossover points) by comparing a 
% global scaling model with alternative multi-regime specifications (Segmented and Trimmed).
% =========================================================================
%% Main Processing Loop: Data Preparation and Initialization

[m, ~] = size(T_cleaned);
[~, n] = size(x_final);
n = n + 2;
y = table2array(T_cleaned(:, 3 : n)); 

[y_final] = deal(nan(m, n));

% Standardize input matrices by filtering out NaN values from the spatial axis
for i = 1 : m
    non_empty_indices = find(~isnan(x_final(i, :)));
    y_final(i, 1:length(non_empty_indices)) = y(i, non_empty_indices);
end

resultTable = createEmptyResultTable();

fprintf('Starting hierarchical regime identification for %d samples ...\n', m);

for i = 1 : m
    valid_idx = ~isnan(y_final(i,:));
    x_valid = x_final(i, valid_idx);
    y_valid = y_final(i, valid_idx);

    % Isolate the effective spatial extent by locating the first non-zero observation
    first_nonzero = find(y_valid ~= 0, 1, 'first');
    if isempty(first_nonzero)
        nanRow = createNaNRow(resultTable, i);
        nanRow.first_nonzero_idx = NaN;  
        resultTable = [resultTable; nanRow];
        continue;
    end
    first_nonzero_idx = first_nonzero;

    % Extract the valid sequence and compute the cumulative distribution 
    % for subsequent log-log scaling analysis
    x_current = x_valid(first_nonzero:end)';
    y_current = cumsum(y_valid(first_nonzero:end)');

    n_samples = length(x_current) - 1; 

    non_zero_y = (y_valid(first_nonzero:end) ~= 0);
    n_non_zero = sum(non_zero_y);

    % Progress indicator
    fprintf('Processed %d/%d samples (%.1f%%)\n', i, m, 100*i/m);

    % Ensure sufficient statistical power (minimum 10 valid observations)
    if n_samples >= 10 && n_non_zero >= 10  
        % Transform to log-log space for scaling exponent estimation
        logy = log(y_current(1:n_samples+1));
        logx = log(x_current(1:n_samples+1));
        
        % Execute the core hierarchical model selection algorithm
        result = findSingleBreakPoint(logx, logy, 5, 2, 1000, [], true);      
        
        % --- Extract and format multi-regime statistics if a segmented model is selected
        if isfield(result, 'type') && strcmp(result.type, 'segmented') && isfield(result, 'breakpoints') && ~isempty(result.breakpoints)
            segStats = getSegmentsStats(logx, logy, result.breakpoints);
            result.segmented.seg_slope     = cellfun(@(s) s.slope, segStats);
            result.segmented.seg_intercept = cellfun(@(s) s.intercept, segStats);
            result.segmented.seg_r2        = cellfun(@(s) s.r2, segStats);
            result.segmented.seg_adj_r2    = cellfun(@(s) s.adj_r2, segStats);
            result.segmented.seg_SSE       = cellfun(@(s) s.SSE, segStats);
            result.segmented.seg_AIC       = cellfun(@(s) s.AIC, segStats);
            result.segmented.seg_pValue    = cellfun(@(s) s.pValue, segStats);
            result.segmented.seg_length    = cellfun(@(s) s.length, segStats);
        end
        resultTable = appendResult(resultTable, result, i, logx, first_nonzero_idx);
    else
        nanRow = createNaNRow(resultTable, i);
        nanRow.first_nonzero_idx = first_nonzero_idx;
        resultTable = [resultTable; nanRow];
    end
end

% Expand segmented statistics cell arrays into discrete columns (supporting up to 3 segments)
for k = 1:3
    cname_slope     = sprintf('seg_slope%d', k);
    cname_intercept = sprintf('seg_intercept%d', k);
    cname_r2        = sprintf('seg_r2_%d', k);
    cname_adj_r2    = sprintf('seg_adj_r2_%d', k);
    cname_SSE       = sprintf('seg_SSE%d', k);
    cname_AIC       = sprintf('seg_AIC%d', k);
    cname_p         = sprintf('seg_pValue%d', k);
    cname_length    = sprintf('seg_length%d', k);
    
    resultTable.(cname_slope)     = NaN(height(resultTable),1);
    resultTable.(cname_intercept) = NaN(height(resultTable),1);
    resultTable.(cname_r2)        = NaN(height(resultTable),1);
    resultTable.(cname_adj_r2)    = NaN(height(resultTable),1);
    resultTable.(cname_SSE)       = NaN(height(resultTable),1);
    resultTable.(cname_AIC)       = NaN(height(resultTable),1);
    resultTable.(cname_p)         = NaN(height(resultTable),1);
    resultTable.(cname_length)    = NaN(height(resultTable),1);
end

% Populate the expanded columns with regime-specific metrics
for i = 1:height(resultTable)
    for k = 1:3
        if ~isempty(resultTable.seg_slope{i}) && length(resultTable.seg_slope{i}) >= k
            resultTable.(sprintf('seg_slope%d',k))(i) = resultTable.seg_slope{i}(k);
        end
        if ~isempty(resultTable.seg_intercept{i}) && length(resultTable.seg_intercept{i}) >= k
            resultTable.(sprintf('seg_intercept%d',k))(i) = resultTable.seg_intercept{i}(k);
        end
        if ~isempty(resultTable.seg_r2{i}) && length(resultTable.seg_r2{i}) >= k
            resultTable.(sprintf('seg_r2_%d',k))(i) = resultTable.seg_r2{i}(k);
        end
        if ~isempty(resultTable.seg_adj_r2{i}) && length(resultTable.seg_adj_r2{i}) >= k
            resultTable.(sprintf('seg_adj_r2_%d',k))(i) = resultTable.seg_adj_r2{i}(k);
        end
        if ~isempty(resultTable.seg_SSE{i}) && length(resultTable.seg_SSE{i}) >= k
            resultTable.(sprintf('seg_SSE%d',k))(i) = resultTable.seg_SSE{i}(k);
        end
        if ~isempty(resultTable.seg_AIC{i}) && length(resultTable.seg_AIC{i}) >= k
            resultTable.(sprintf('seg_AIC%d',k))(i) = resultTable.seg_AIC{i}(k);
        end
        if ~isempty(resultTable.seg_pValue{i}) && length(resultTable.seg_pValue{i}) >= k
            resultTable.(sprintf('seg_pValue%d',k))(i) = resultTable.seg_pValue{i}(k);
        end
        if ~isempty(resultTable.seg_length{i}) && length(resultTable.seg_length{i}) >= k
            resultTable.(sprintf('seg_length%d',k))(i) = resultTable.seg_length{i}(k);
        end
    end
end

fprintf('All %d samples have been processed!\n', m);

%% --------------------- Core Algorithm & Helper Functions ---------------------

function result = findSingleBreakPoint(logx, logy, minSamples, numFolds, numPerm, bpList, isTopLevel)
% FINDSINGLEBREAKPOINT: Hierarchical algorithm to identify scaling regimes.
% Evaluates Global, Segmented, and Trimmed (S-global) models.

if nargin < 7, isTopLevel = true; end
if nargin < 6, bpList = []; end
if nargin < 5, numPerm = 1000; end
if nargin < 4, numFolds = 2; end
if nargin < 3, minSamples = 5; end

n = length(logx);
result = createEmptyResult();
result.type = 'global';
result.breakpoints = [];

% =========================================================================
% Stage (1): Baseline Global Model
% Fits a single OLS regression to the full dataset in log-log space.
% Serves as the null hypothesis (invariant scaling exponent).
% =========================================================================
[globalSlope, globalIntercept, globalR2, globalSSE, globalAIC, globalStats] = linearFitWithStats(logx, logy);
globalCVError = crossValError(logx, logy, [], numFolds);
globalAdjR2 = 1 - (1-globalR2)*(n-1)/(n-2);

result.global.slope     = globalSlope;
result.global.intercept = globalIntercept;
result.global.r2        = globalR2;
result.global.SSE       = globalSSE;
result.global.AIC       = globalAIC;
result.global.CVError   = globalCVError;
result.global.pValue    = globalStats.pValue;
result.global.adj_r2    = globalAdjR2;

% =========================================================================
% Stage (2a): Segmented Regression (Detection of structural breaks)
% Exhaustively evaluates candidate breakpoint locations for multistage scaling.
% =========================================================================
candidatePoints = minSamples:(n-minSamples);
validCandidates = [];

for bp = candidatePoints
    x1 = logx(1:bp); y1 = logy(1:bp);
    x2 = logx(bp+1:end); y2 = logy(bp+1:end);

    [slope1, intercept1, r2_1, SSE1, aic1, stats1] = linearFitWithStats(x1, y1);
    [slope2, intercept2, r2_2, SSE2, aic2, stats2] = linearFitWithStats(x2, y2);

    adj_r2_1 = 1 - (1-r2_1)*(length(x1)-1)/(length(x1)-2);
    adj_r2_2 = 1 - (1-r2_2)*(length(x2)-1)/(length(x2)-2);

    SSE_seg = SSE1 + SSE2;
    segAIC = aic1 + aic2;

    % Segmented model provides significantly better fit (F-test)
    F_improve = ((globalSSE - SSE_seg)/2)/(SSE_seg/(n-4));
    pImprove = 1 - fcdf(F_improve, 2, n-4);

    % Slopes of adjacent segments are statistically distinct
    [pSlopeDiff, ~] = testSlopeDifference(x1, y1, x2, y2);

    segResult = struct( ...
        'bp', bp, ...
        'slope1', slope1, 'intercept1', intercept1, 'r2_1', r2_1, 'adj_r2_1', adj_r2_1, 'SSE1', SSE1, 'p1', stats1.pValue, ...
        'slope2', slope2, 'intercept2', intercept2, 'r2_2', r2_2, 'adj_r2_2', adj_r2_2, 'SSE2', SSE2, 'p2', stats2.pValue, ...
        'length1', length(x1), 'length2', length(x2), ...
        'SSE', SSE_seg, 'AIC', segAIC, ...
        'pImprove', pImprove, 'pSlopeDiff', pSlopeDiff, ...
        'R2Var', var([r2_1, r2_2]), 'R2Min', min([r2_1, r2_2]), 'minAdjR2', min([adj_r2_1, adj_r2_2]), 'maxSSE', max([SSE1, SSE2]));

    % Enforce all criteria: significant linearity, distinct slopes, 
    % better fit via F-test, adjusted R2 >= global model
    if pImprove < 0.05 && pSlopeDiff < 0.05 && ...
       stats1.pValue < 0.05 && stats2.pValue < 0.05 && ...
       adj_r2_1 >= globalAdjR2 && adj_r2_2 >= globalAdjR2
        validCandidates = [validCandidates; segResult];
    end
end

% --- Stage (3): Permutation Test for 1-breakpoint candidates ---
% Verifies that AIC reduction is unlikely to arise from random variation.
best1 = [];
best1Stats = [];
best1MinAdjR2 = -Inf;
best1Breaks = [];
if ~isempty(validCandidates)
    permPassedCandidates = [];
    for i = 1:length(validCandidates)
        candidate = validCandidates(i);
        obsStat = globalAIC - candidate.AIC;
        permStats = zeros(numPerm,1);
        for pi = 1:numPerm
            yperm = logy(randperm(n));
            [~, ~, ~, ~, permAIC] = linearFitWithStats(logx, yperm);
            x1 = logx(1:candidate.bp); y1p = yperm(1:candidate.bp);
            x2 = logx(candidate.bp+1:end); y2p = yperm(candidate.bp+1:end);
            [~, ~, ~, ~, aic1] = linearFitWithStats(x1, y1p);
            [~, ~, ~, ~, aic2] = linearFitWithStats(x2, y2p);
            permImprove = permAIC - (aic1 + aic2);
            permStats(pi) = permImprove;
        end
        permP = mean(permStats >= obsStat);
        if permP < 0.05
            candidate.permP = permP;
            candidate.obsStat = obsStat;
            permPassedCandidates = [permPassedCandidates; candidate];
        end
    end

    % Optimal breakpoint configuration selected by maximizing the minimum adjusted R2
    if ~isempty(permPassedCandidates)
        [~, idx] = max([permPassedCandidates.minAdjR2]);
        best1 = permPassedCandidates(idx);
        best1Breaks = [bpList, best1.bp];
        best1Breaks = best1Breaks(best1Breaks > 0 & best1Breaks < n);
        best1Breaks = unique(best1Breaks);
        best1Stats = getSegmentsStats(logx, logy, best1Breaks);
        best1MinAdjR2 = min(cellfun(@(s) s.adj_r2, best1Stats));
    end
end

% --- Evaluate 2-breakpoint solutions (3 segments) ---
best2MinAdjR2 = -Inf;
best2Breaks = [];
best2Stats = [];
best2 = [];
if ~isempty(best1) && numel(best1Breaks)==1
    bp = best1Breaks(1);

    % Iteratively test splitting the left segment
    leftRange = minSamples:(bp-minSamples);
    for bp2 = leftRange
        candidateBreaks = sort([bp2 bp]);
        segStats = getSegmentsStats(logx, logy, candidateBreaks);
        if allSegmentsMeetCriteria(segStats, globalAdjR2, minSamples)
            minAdjR2 = min(cellfun(@(s) s.adj_r2, segStats));
            if minAdjR2 > best2MinAdjR2
                best2MinAdjR2 = minAdjR2;
                best2Breaks = candidateBreaks;
                best2Stats = segStats;
            end
        end
    end

    % Iteratively test splitting the right segment
    rightRange = (bp+minSamples):(n-minSamples);
    for bp2 = rightRange
        candidateBreaks = sort([bp bp2]);
        segStats = getSegmentsStats(logx, logy, candidateBreaks);
        if allSegmentsMeetCriteria(segStats, globalAdjR2, minSamples)
            minAdjR2 = min(cellfun(@(s) s.adj_r2, segStats));
            if minAdjR2 > best2MinAdjR2
                best2MinAdjR2 = minAdjR2;
                best2Breaks = candidateBreaks;
                best2Stats = segStats;
            end
        end
    end
end

% Select the most robust segmented model (balancing complexity and fit)
bestSegmented = [];
bestSegBreaks = [];
if ~isempty(best2Breaks) && best2MinAdjR2 > best1MinAdjR2
    bestSegmented = best2;
    bestSegmented.breakpoints = best2Breaks;
    bestSegmented.segStats = best2Stats;
    bestSegBreaks = best2Breaks;
elseif ~isempty(best1)
    bestSegmented = best1;
    bestSegmented.breakpoints = best1Breaks;
    bestSegmented.segStats = best1Stats;
    bestSegBreaks = best1Breaks;
end

% Early exit for recursive calls
if nargin >= 7 && ~isTopLevel
    if ~isempty(bestSegmented)
        result.segmented = bestSegmented;
        result.type = 'segmented';
        result.breakpoints = bestSegBreaks;
    end
    return;
end

% =========================================================================
% Stage (2b): Trimmed Linear Regression (S-global)
% Accounts for deviations arising from boundary effects or local noise.
% Iteratively removes observations from the tails up to a predefined threshold.
% =========================================================================
sglobalCandidates = [];
minLengthRatio = 2/3;
maxSingleRemove = 4;  % Maximum points to trim from a single end
maxDoubleRemove = 5;  % Maximum points to trim from both ends combined

% Evaluate front-tail trimming
for bp_front = 1:min(maxSingleRemove, n-1)
    x_segment = logx(bp_front+1:end);
    y_segment = logy(bp_front+1:end);
    n_segment = length(x_segment);
    if n_segment >= n*minLengthRatio
        [slope, intercept, r2, SSE, AIC, stats] = linearFitWithStats(x_segment, y_segment);
        adj_r2 = 1 - (1-r2)*(n_segment-1)/(n_segment-2);
        % Retain if slope is significant and adjusted R2 exceeds global model
        if stats.pValue < 0.05 && adj_r2 > globalAdjR2
            sglobalCandidates = [sglobalCandidates; struct( ...
                'bp_front', bp_front, 'bp_back', 0, ...
                'slope', slope, 'intercept', intercept, ...
                'r2', r2, 'adj_r2', adj_r2, 'SSE', SSE, ...
                'pValue', stats.pValue, 'length', n_segment, 'AIC', AIC)];
        end
    end
end

% Evaluate back-tail trimming
for bp_back = 1:min(maxSingleRemove, n-1)
    x_segment = logx(1:end-bp_back);
    y_segment = logy(1:end-bp_back);
    n_segment = length(x_segment);
    if n_segment >= n*minLengthRatio
        [slope, intercept, r2, SSE, AIC, stats] = linearFitWithStats(x_segment, y_segment);
        adj_r2 = 1 - (1-r2)*(n_segment-1)/(n_segment-2);
        if stats.pValue < 0.05 && adj_r2 > globalAdjR2
            sglobalCandidates = [sglobalCandidates; struct( ...
                'bp_front', 0, 'bp_back', bp_back, ...
                'slope', slope, 'intercept', intercept, ...
                'r2', r2, 'adj_r2', adj_r2, 'SSE', SSE, ...
                'pValue', stats.pValue, 'length', n_segment, 'AIC', AIC)];
        end
    end
end

% Evaluate simultaneous double-end trimming
for total_remove = 2:maxDoubleRemove
    for bp_front = 1:total_remove-1
        bp_back = total_remove - bp_front;
        if bp_back < 1 || (n - bp_front - bp_back) < 1
            continue;
        end
        x_segment = logx(bp_front+1:end-bp_back);
        y_segment = logy(bp_front+1:end-bp_back);
        n_segment = length(x_segment);
        if n_segment >= n*minLengthRatio
            [slope, intercept, r2, SSE, AIC, stats] = linearFitWithStats(x_segment, y_segment);
            adj_r2 = 1 - (1-r2)*(n_segment-1)/(n_segment-2);
            if stats.pValue < 0.05 && adj_r2 > globalAdjR2
                sglobalCandidates = [sglobalCandidates; struct( ...
                    'bp_front', bp_front, 'bp_back', bp_back, ...
                    'slope', slope, 'intercept', intercept, ...
                    'r2', r2, 'adj_r2', adj_r2, 'SSE', SSE, ...
                    'pValue', stats.pValue, 'length', n_segment, 'AIC', AIC)];
            end
        end
    end
end

% --- Stage (3): Permutation Test for Trimmed Models ---
permPassedSglobal = [];
if ~isempty(sglobalCandidates)
    for i = 1:length(sglobalCandidates)
        candidate = sglobalCandidates(i);
        obsStat = globalAIC - candidate.AIC;
        permStats = zeros(numPerm,1);
        for pi = 1:numPerm
            yperm = logy(randperm(n));
            [~, ~, ~, ~, permAIC] = linearFitWithStats(logx, yperm);
            if candidate.bp_back == 0
                x_perm = logx(candidate.bp_front+1:end);
                y_perm = yperm(candidate.bp_front+1:end);
            elseif candidate.bp_front == 0
                x_perm = logx(1:end-candidate.bp_back);
                y_perm = yperm(1:end-candidate.bp_back);
            else
                x_perm = logx(candidate.bp_front+1:end-candidate.bp_back);
                y_perm = yperm(candidate.bp_front+1:end-candidate.bp_back);
            end
            [~, ~, ~, ~, aic_perm] = linearFitWithStats(x_perm, y_perm);
            permStats(pi) = permAIC - aic_perm;
        end
        permP = mean(permStats >= obsStat);
        if permP < 0.05
            candidate.permP = permP;
            candidate.obsStat = obsStat;
            permPassedSglobal = [permPassedSglobal; candidate];
        end
    end
end

bestSglobal = [];
if ~isempty(permPassedSglobal)
    [~, idx] = max([permPassedSglobal.adj_r2]);
    bestSglobal = permPassedSglobal(idx);
end

% =========================================================================
% Stage (3): Final Model Selection
% Prioritizes explanatory performance by comparing the mean adjusted R2 
% across segments (Segmented) with the adjusted R2 of the Trimmed or Global model.
% =========================================================================
if ~isempty(bestSegmented) && isempty(bestSglobal)
    result.type = 'segmented';
    result.segmented = bestSegmented;
    result.breakpoints = bestSegBreaks;
    result.hypothesis.permP   = getfieldifexist(bestSegmented,'permP',NaN);
    result.hypothesis.obsStat = getfieldifexist(bestSegmented,'obsStat',NaN);
elseif isempty(bestSegmented) && ~isempty(bestSglobal)
    result.type = 's-global'; % Trimmed model
    result.sglobal = bestSglobal;
    result.breakpoints = [];
    result.hypothesis.permP   = getfieldifexist(bestSglobal,'permP',NaN);
    result.hypothesis.obsStat = getfieldifexist(bestSglobal,'obsStat',NaN);
elseif ~isempty(bestSegmented) && ~isempty(bestSglobal)
    % Compare mean adjusted R2 of segmented regimes vs trimmed model
    segStats = getSegmentsStats(logx, logy, bestSegBreaks);
    adjR2s = cellfun(@(s) s.adj_r2, segStats);
    avgAdjR2 = mean(adjR2s);
    
    if bestSglobal.adj_r2 > avgAdjR2
        result.type = 's-global';
        result.sglobal = bestSglobal;
        result.breakpoints = [];
        result.hypothesis.permP   = getfieldifexist(bestSglobal,'permP',NaN);
        result.hypothesis.obsStat = getfieldifexist(bestSglobal,'obsStat',NaN);
    else
        result.type = 'segmented';
        result.segmented = bestSegmented;
        result.breakpoints = bestSegBreaks;
        result.hypothesis.permP   = getfieldifexist(bestSegmented,'permP',NaN);
        result.hypothesis.obsStat = getfieldifexist(bestSegmented,'obsStat',NaN);
    end
else
    % Fallback to the baseline global model if no complex structure is justified
    result.type = 'global';
    result.breakpoints = [];
    result.hypothesis.permP = NaN;
    result.hypothesis.obsStat = NaN;
end
end

% -------------------------------------------------------------------------
% Utility Functions
% -------------------------------------------------------------------------

function val = getfieldifexist(S, fieldname, default)
    % Safely extracts a field from a structure, returning a default if missing
    if isstruct(S) && isfield(S, fieldname)
        val = S.(fieldname);
    else
        val = default;
    end
end

function segStats = getSegmentsStats(logx, logy, breakpoints)
    % Computes OLS regression statistics for each identified segment
    n = length(logx);
    breakpoints = breakpoints(breakpoints > 0 & breakpoints < n);
    breakpoints = unique(breakpoints);
    segments = [1, breakpoints + 1, n + 1];
    segStats = cell(1, length(segments) - 1);
    for i = 1:(length(segments) - 1)
        idx = segments(i):(segments(i + 1) - 1);
        if length(idx) < 2
            segStats{i} = struct('slope', NaN, 'intercept', NaN, 'r2', NaN, ...
                'adj_r2', NaN, 'SSE', NaN, 'AIC', NaN, 'pValue', NaN, 'length', length(idx));
            continue;
        end
        xseg = logx(idx);
        yseg = logy(idx);
        [slope, intercept, r2, SSE, AIC, stats] = linearFitWithStats(xseg, yseg);
        adj_r2 = 1 - (1-r2)*(length(xseg)-1)/(length(xseg)-2);
        segStats{i} = struct( ...
            'slope', slope, 'intercept', intercept, 'r2', r2, ...
            'adj_r2', adj_r2, 'SSE', SSE, 'AIC', AIC, ...
            'pValue', stats.pValue, 'length', length(xseg));
    end
end

function pass = allSegmentsMeetCriteria(segStats, globalAdjR2, minSamples)
    % Validates that all segments in a multi-regime model meet the strict 
    % statistical inclusion criteria (significance, length, and explanatory power)
    pass = true;
    for i = 1:numel(segStats)
        s = segStats{i};
        if isnan(s.pValue) || isnan(s.adj_r2) || isnan(s.length)
            pass = false;
            break;
        end
        if ~(s.pValue < 0.05 && s.adj_r2 >= globalAdjR2 && s.length >= minSamples)
            pass = false;
            break;
        end
    end
end

function T = createEmptyResultTable()
    % Initializes the standardized output table for regime classification
    varNames = { ...
        'sampleID', 'type', ...
        'global_slope', 'global_intercept', 'global_r2', 'global_SSE', 'global_AIC', 'global_CVError', 'global_pValue', ...
        'breakpoints', ... 
        'seg_slope', 'seg_intercept', 'seg_r2', 'seg_adj_r2', 'seg_SSE', 'seg_pValue', 'seg_length', 'seg_AIC', ...
        'sglobal_bp_front', 'sglobal_bp_back', ...
        'sglobal_slope', 'sglobal_intercept', 'sglobal_r2', 'sglobal_adj_r2', 'sglobal_SSE', 'sglobal_pValue', 'sglobal_length', 'sglobal_AIC', ...
        'permP', 'obsStat', 'length', 'first_nonzero_idx'
    };
    varTypes = [{ 'string' }, {'string'}, repmat({'double'}, 1, 7), ...
                {'cell'}, ... 
                repmat({'cell'}, 1, 8), ... 
                repmat({'double'}, 1, 10), ... 
                {'double'}, {'double'}, {'double'}, {'double'}];
    T = table('Size', [0, numel(varNames)], 'VariableTypes', varTypes, 'VariableNames', varNames);
end

function nanRow = createNaNRow(resultTable, sampleID)
    % Generates a null entry for samples that fail data quality requirements
    nanCell = cell(1, width(resultTable));
    nanCell{1} = string(sampleID);    
    nanCell{2} = "";                  
    for i = 3:9, nanCell{i} = NaN; end
    nanCell{10} = {[]};               
    for i = 11:18, nanCell{i} = {[]}; end
    for i = 19:28, nanCell{i} = NaN; end
    for i = 29:31, nanCell{i} = NaN; end
    nanCell{32} = NaN;                
    nanRow = cell2table(nanCell, 'VariableNames', resultTable.Properties.VariableNames);
end

function resultTable = appendResult(resultTable, result, sampleID, logx, first_nonzero_idx)
    % Appends the optimal model configuration and its statistics to the results table
    function val = getOrNaN(s, f, isCell)
        if isfield(s, f) && ~isempty(s.(f))
            if nargin > 2 && isCell
                val = {s.(f)};
            else
                val = s.(f);
            end
        else
            if nargin > 2 && isCell
                val = {[]};
            else
                val = NaN;
            end
        end
    end
    nTotal = length(logx);
    if isfield(result, 'type') && isequal(result.type, 'missing')
        nanRow = createNaNRow(resultTable, sampleID);
        nanRow.first_nonzero_idx = first_nonzero_idx;
        resultTable = [resultTable; nanRow];
        return;
    end
    gs = result.global;
    seg = result.segmented;
    if isstruct(seg) && ~isempty(seg) && isfield(seg, 'seg_slope')
        seg_slope_cell     = {seg.seg_slope};
        seg_intercept_cell = {seg.seg_intercept};
        seg_r2_cell        = {seg.seg_r2};
        seg_adj_r2_cell    = {seg.seg_adj_r2};
        seg_SSE_cell       = {seg.seg_SSE};
        seg_pValue_cell    = {seg.seg_pValue};
        seg_length_cell    = {seg.seg_length};
        seg_AIC_cell       = {seg.seg_AIC};
    else
        seg_slope_cell     = {[]};
        seg_intercept_cell = {[]};
        seg_r2_cell        = {[]};
        seg_adj_r2_cell    = {[]};
        seg_SSE_cell       = {[]};
        seg_pValue_cell    = {[]};
        seg_length_cell    = {[]};
        seg_AIC_cell       = {[]};
    end
    sgs = result.sglobal;
    if isempty(sgs) || all(structfun(@(x) isnan(x), sgs))
        sgs = struct();
    end
    if isfield(result, 'type') && (ischar(result.type) || isstring(result.type))
        typeVal = string(result.type);
    else
        typeVal = "";
    end
    if isfield(result, 'breakpoints') && ~isempty(result.breakpoints)
        breakpointsVal = {result.breakpoints};
    else
        breakpointsVal = {[]};
    end
    newRow = { ...
        string(sampleID), typeVal, ...
        getOrNaN(gs, 'slope'), getOrNaN(gs, 'intercept'), getOrNaN(gs, 'r2'), getOrNaN(gs, 'SSE'), ...
        getOrNaN(gs, 'AIC'), getOrNaN(gs, 'CVError'), getOrNaN(gs, 'pValue'), ...
        breakpointsVal, ...
        seg_slope_cell, seg_intercept_cell, seg_r2_cell, seg_adj_r2_cell, seg_SSE_cell, seg_pValue_cell, seg_length_cell, seg_AIC_cell, ...
        getOrNaN(sgs, 'bp_front'), getOrNaN(sgs, 'bp_back'), getOrNaN(sgs, 'slope'), getOrNaN(sgs, 'intercept'), ...
        getOrNaN(sgs, 'r2'), getOrNaN(sgs, 'adj_r2'), getOrNaN(sgs, 'SSE'), getOrNaN(sgs, 'pValue'), getOrNaN(sgs, 'length'), getOrNaN(sgs, 'AIC'), ...
        getOrNaN(result.hypothesis, 'permP'), getOrNaN(result.hypothesis, 'obsStat'), nTotal, ...
        first_nonzero_idx
    };
    if length(newRow) ~= width(resultTable)
        error('appendResult:DimensionMismatch', ...
            'newRow length = %d, table width = %d', length(newRow), width(resultTable));
    end
    rowTable = cell2table(newRow, 'VariableNames', resultTable.Properties.VariableNames);
    resultTable = [resultTable; rowTable];
end

function [slope, intercept, r2, SSE, AIC, stats] = linearFitWithStats(x, y)
    % Performs Ordinary Least Squares (OLS) regression and computes fit metrics
    if length(x) < 2
        slope = NaN; intercept = NaN; r2 = NaN; SSE = NaN; AIC = NaN;
        stats = struct('r2',NaN,'F',NaN,'pValue',NaN,'MSE',NaN);
        return;
    end
    X = [x(:), ones(length(x),1)];
    [b, ~, ~, ~, regressStats] = regress(y(:), X);
    slope = b(1);
    intercept = b(2);
    yfit = X*b;
    residuals = y(:) - yfit;
    SSE = sum(residuals.^2);
    r2 = 1 - SSE / sum((y(:)-mean(y(:))).^2);
    n = length(x);
    k = 2; % Number of parameters (slope + intercept)
    sigma2 = SSE/n;
    AIC = n*log(sigma2) + 2*k; % Akaike Information Criterion
    stats = struct('r2', regressStats(1), 'F', regressStats(2), 'pValue', regressStats(3), 'MSE', regressStats(4));
end

function [p, tstat] = testSlopeDifference(x1, y1, x2, y2)
    % Two-sided t-test to evaluate if the slopes of adjacent segments are statistically distinct
    if length(x1) < 2 || length(x2) < 2
        p = NaN; tstat = NaN; return;
    end
    [b1, ~, ~, ~, stats1] = regress(y1, [x1, ones(length(x1),1)]);
    [b2, ~, ~, ~, stats2] = regress(y2, [x2, ones(length(x2),1)]);
    se1 = sqrt(stats1(4));
    se2 = sqrt(stats2(4));
    slope_diff = b1(1) - b2(1);
    se_diff = sqrt(se1^2 + se2^2);
    tstat = slope_diff / se_diff;
    df = length(x1) + length(x2) - 4;
    p = 2 * (1 - tcdf(abs(tstat), df));
end

function err = crossValError(x, y, bp, kfold)
    % Computes K-fold cross-validation error for model robustness checks
    n = length(x);
    indices = crossvalind('Kfold', n, kfold);
    err = 0;
    for i = 1:kfold
        test = (indices == i);
        train = ~test;
        xtr = x(train); ytr = y(train);
        xte = x(test); yte = y(test);
        if isempty(bp)
            Xtr = [xtr(:), ones(sum(train),1)];
            b = Xtr \ ytr(:);
            Xte = [xte(:), ones(sum(test),1)];
            ypred = Xte*b;
        else
            bpVal = x(bp);
            ypred = zeros(size(yte));
            ind1 = xte <= bpVal;
            ind2 = ~ind1;
            if any(ind1)
                Xtr1 = [xtr(xtr <= bpVal), ones(sum(xtr <= bpVal),1)];
                ytr1 = ytr(xtr <= bpVal);
                b1 = Xtr1 \ ytr1;
                Xte1 = [xte(ind1), ones(sum(ind1),1)];
                ypred(ind1) = Xte1 * b1;
            end
            if any(ind2)
                Xtr2 = [xtr(xtr > bpVal), ones(sum(xtr > bpVal),1)];
                ytr2 = ytr(xtr > bpVal);
                b2 = Xtr2 \ ytr2;
                Xte2 = [xte(ind2), ones(sum(ind2),1)];
                ypred(ind2) = Xte2 * b2;
            end
        end
        err = err + sum((yte - ypred).^2);
    end
    err = err / n;
end

function result = createEmptyResult()
    % Initializes the standardized hierarchical model structure
    result = struct();
    result.type = 'global';
    result.global = struct('slope', NaN, 'intercept', NaN, 'r2', NaN, ...
                          'SSE', NaN, 'AIC', NaN, 'CVError', NaN, 'pValue', NaN);
    result.segmented = struct('bp', NaN, 'slope1', NaN, 'intercept1', NaN, ...
                             'r2_1', NaN, 'adj_r2_1', NaN, 'SSE1', NaN, 'p1', NaN, ...
                             'slope2', NaN, 'intercept2', NaN, ...
                             'r2_2', NaN, 'adj_r2_2', NaN, 'SSE2', NaN, 'p2', NaN, ...
                             'length1', NaN, 'length2', NaN);
    result.sglobal = struct('bp_front', NaN, 'bp_back', NaN, 'slope', NaN, 'intercept', NaN, ...
        'r2', NaN, 'adj_r2', NaN, 'SSE', NaN, 'pValue', NaN, 'length', NaN, 'AIC', NaN);
    result.hypothesis = struct('permP', NaN, 'obsStat', NaN);
end