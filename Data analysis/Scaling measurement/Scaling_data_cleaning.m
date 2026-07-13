% =========================================================================
% Data Cleaning: Filtering Inactive Spatial Units and Aligning Datasets
% 
% Spatial units with absolutely zero commuting flows across all distance 
% bins (annuli) act as inactive nodes in the network. This section filters 
% out these inactive nodes.
% =========================================================================

% Extract the flow matrices across all distance rings.
Records = totalflow(:, 3 : 202);

% Convert the MATLAB table to a numeric array for efficient matrix operations
Records_a = table2array(Records);

% Generate a logical mask for active spatial units.
nonZeroRows = any(Records_a, 2);

% Apply the logical mask to the original flow dataset to remove inactive nodes,
% yielding a cleaned dataset for subsequent analysis.
T_cleaned = totalflow(nonZeroRows, :);

% Apply the exact same logical mask to the corresponding demographic dataset (Pop).
% This guarantees that the spatial units in the population dataset remain 
% perfectly aligned (1-to-1 mapping) with the cleaned commuting flow dataset.
P_cleaned = Pop(nonZeroRows, :);