% =========================================================================
% Script Name: Omnicentric_scaling_measurement
% Description: This script measures distance-decay commuting flows and 
%              spatial population around each spatial unit. It 
%              evaluates Origin-Destination (OD) commuting patterns and 
%              demographic distributions within concentric distance rings.
%
% Inputs:
%   - MUnit:   A MATLAB Table containing spatial unit attributes 
%              (Required columns: 'ORIG_FID', 'x', 'y', 'id', 'Pop').
%   - Commute: A MATLAB Table containing OD commuting flows 
%              (Required columns: 'home', 'work', and flow volume at index 3).
%
% Outputs:
%   - Result1: Commuting flows originating from the central unit (Outflows).
%   - Result2: Commuting flows destined for the central unit (Inflows).
%   - Result3: Spatial population variance between central and neighboring units.
%   - Result4: Total population within each specific distance ring.
%   - Result5: Number of spatial units within each specific distance ring.
% =========================================================================

% Extract spatial coordinates (x, y) and unique identifiers for all units
Index = table2array(MUnit(:, {'ORIG_FID', 'x', 'y'}));

n = length(Index); % Total number of spatial units
m = 199.5;         % Maximum search radius for spatial neighborhood (e.g., 199.5 km)

% Initialize matrices to store the final aggregated results for all units
% Note: Each row will represent a central spatial unit, and each column 
% will represent a specific distance ring (annulus).
Result1 = []; % Commuting outflows (Home to Work)
Result2 = []; % Commuting inflows (Work to Home)
Result3 = []; % Population spatial variance 
Result4 = []; % Total population in the distance ring
Result5 = []; % Count of spatial units in the distance ring

% Iterate through each spatial unit to calculate localized spatial metrics
for i = 1 : n
    % ---------------------------------------------------------------------
    % Step 1: Spatial Neighborhood Identification
    % Perform a range search to find all neighboring units within radius 'm'
    % using Euclidean distance.
    % ---------------------------------------------------------------------
    [ind, dis] = rangesearch(Index(:,2:3), Index(i,2:3), m, "Distance", "euclidean");
    
    % Convert cell arrays to standard arrays and transpose to column vectors
    I = cell2mat(ind)'; 
    D = cell2mat(dis)'; 
    
    % Extract attributes of the current central unit (focal node)
    location = MUnit{i, "id"}; % Unique ID of the central unit
    cp = MUnit{i, "Pop"};      % Population of the central unit
    
    % ---------------------------------------------------------------------
    % Step 2: Commuting Flow Filtering
    % Isolate the OD flows associated with the current central unit.
    % ---------------------------------------------------------------------
    work_index = strcmp(Commute.home, location); % Flows originating from this unit
    home_index = strcmp(Commute.work, location); % Flows destined to this unit
    
    work = Commute(work_index, :); % Subset of outflow data
    home = Commute(home_index, :); % Subset of inflow data
    
    % Initialize temporary arrays to store metrics for concentric rings
    R1 = []; R2 = []; R3 = []; R4 = []; R5 = [];
    
    % ---------------------------------------------------------------------
    % Step 3: Concentric Ring (Annulus) Analysis
    % Iterate through distance bins (width = 1 unit) from 0.5 to m.
    % This measures annular commuting and demographic statistics.
    % ---------------------------------------------------------------------
    for j = 0.5 : 1 : m
        % Identify neighboring units located within the current distance ring
        Z = find(D > j - 0.5 & D <= j + 0.5);
        A = I(Z);                     % Indices of units in the current ring
        B = MUnit{A, "id"};           % IDs of units in the current ring
        D_pop = table2array(MUnit(A, {'Pop'})); % Populations of units in the ring
        
        % Calculate total commuting outflows to units in the current ring
        dis_index_w = ismember(work.work, B);
        dis_work = table2array(work(dis_index_w, 3));
        sum_work = sum(dis_work);
        
        % Calculate total commuting inflows from units in the current ring
        dis_index_h = ismember(home.home, B);
        dis_home = table2array(home(dis_index_h, 3));
        sum_home = sum(dis_home);
        
        % Calculate spatial population variance
        E = (D_pop - cp).^ 2; 
        F = sum(E);
        k = length(D_pop); % Number of units in the current ring
        
        % Prevent division by zero if no units are found in the ring
        if k > 0
            G = F / (k * 2); 
        else
            G = 0; 
        end
        
        P = sum(D_pop); % Total population within the current ring
        
        % Append ring-specific metrics to the temporary arrays
        R1 = [R1, sum_work];
        R2 = [R2, sum_home];
        R3 = [R3, G];
        R4 = [R4, P];
        R5 = [R5, k];
    end
    
    % ---------------------------------------------------------------------
    % Step 4: Aggregate Results
    % Append the metrics of the current central unit to the global matrices.
    % ---------------------------------------------------------------------
    Result1 = [Result1; R1];
    Result2 = [Result2; R2];
    Result3 = [Result3; R3];
    Result4 = [Result4; R4];
    Result5 = [Result5; R5];
    
    % Display progress (Current iteration index)
    disp(i);
end