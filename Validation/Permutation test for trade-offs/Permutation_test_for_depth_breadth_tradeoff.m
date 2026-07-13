% =========================================================================
% Permutation-Based Test for Depth-Breadth Trade-off
% 
% Description : This script tests the robustness of the observed systematic 
% trade-off between population-normalized core flow intensity (ln A_i) and 
% spatial reach (\lambda_i). It isolates genuine physical mechanisms from 
% the parameterization artifacts (geometric 'lever effect') inherent to OLS regression.
% (Ref: Supplementary Text S3)
% =========================================================================

% --- Extract fundamental variables ---
lambda_i   = T.a; % \lambda_i: Spatial reach
eta_i      = T.b; % \eta_i: Allometric scaling exponent
log_r_mean = T.c; % Mean of ln r_i (Spatial center of mass X)
log_P_mean = T.d; % Mean of ln P_i(r) (Allometric center of mass X)
log_F_mean = T.e; % Mean of ln F_i(r) (Allometric center of mass Y)

% --- Data Cleaning ---
% Remove NaN values to ensure robust computation
valid_idx = ~isnan(lambda_i) & ~isnan(eta_i) & ~isnan(log_r_mean) & ~isnan(log_P_mean) & ~isnan(log_F_mean);
lambda_i   = lambda_i(valid_idx);
eta_i      = eta_i(valid_idx);
log_r_mean = log_r_mean(valid_idx);
log_P_mean = log_P_mean(valid_idx);
log_F_mean = log_F_mean(valid_idx);
n_samples  = length(lambda_i);

% --- Step 1: Empirical Data Preparation ---
% Calculate the empirical central intensity (ln A_i) constrained by the 
% allometric center of mass and the scaling exponent \eta_i.
log_A_true = log_F_mean - eta_i .* log_P_mean;

% Calculate the empirical correlation coefficient (Observed trade-off)
obs_corr = corr(log_A_true, lambda_i, 'Type', 'Pearson');
fprintf('Empirical r (Correlation between observed ln A_i and \lambda_i): %.4f\n', obs_corr);

% --- Step 2: Permutation-Based Null Model Construction ---
num_permutations = 10000; % Number of permutations for rigorous statistical testing
perm_corrs = zeros(num_permutations, 1); % Pre-allocate array for null distributions

for i = 1:num_permutations
    % Randomly shuffle the parameter pair (\eta_i, \lambda_i) across grids.
    % This maintains their empirical distribution and inherent covariance 
    % while decoupling them from their local geometric constraints.
    shuffle_idx = randperm(n_samples);
    lambda_null = lambda_i(shuffle_idx);
    eta_null    = eta_i(shuffle_idx);
    
    % Recalculate pseudo-intercepts (ln A_null).
    % We preserve the empirical allometric centers of mass for all spatial grids
    % and apply the shuffled scaling exponents to simulate the geometric lever effect.
    log_A_null = log_F_mean - eta_null .* log_P_mean;
    
    % Compute the baseline correlation driven purely by the mechanical 
    % propagation of the geometric constraint.
    perm_corrs(i) = corr(log_A_null, lambda_null, 'Type', 'Pearson');
end

% --- Step 3: Statistical Testing ---
% Calculate the mean of the null distribution (artifactual baseline correlation)
baseline_corr = mean(perm_corrs);
fprintf('Null model r (Baseline correlation driven by geometric lever effect): %.4f\n', baseline_corr);

% Calculate adaptive one-tailed P-value
% Hypothesis: The empirically observed negative correlation is significantly 
% stronger than the mean of the artifactual null distribution.
p_value = sum(perm_corrs <= obs_corr) / num_permutations;

fprintf('Permutation Test P-value: %.6f\n', p_value);
if p_value < 0.001
    fprintf('Conclusion: Highly significant (P < 0.001).\n');
    fprintf('The geometric lever effect mechanically introduces a baseline correlation (r ¡Ö %.2f),\n', baseline_corr);
    fprintf('but the observed trade-off (r = %.2f) robustly diverges from this parameterization artifact.\n', obs_corr);
    fprintf('This confirms the trade-off reflects substantial, underlying physical mechanisms governing spatial resource allocation.\n');
elseif p_value < 0.05
    fprintf('Conclusion: Significant (P < 0.05). The observed spatial trade-off cannot be fully explained by parameterization constraints.\n');
else
    fprintf('Conclusion: Not significant (P >= 0.05). Cannot reject the null hypothesis of geometric constraints.\n');
end

% --- Step 4: Visualization (Publication-Quality Figure) ---
% 1. Figure Setup
fig_size_cm = 9; 
font_main = 'Helvetica';
font_sz_axis = 12;

h_fig = figure('Name', 'Robustness of Spatial Trade-offs', 'Units', 'centimeters');
h_fig.Position = [10, 10, fig_size_cm, fig_size_cm];
set(h_fig, 'Color', 'w');
set(h_fig, 'DefaultAxesFontName', font_main);
set(h_fig, 'DefaultTextFontName', font_main);

ax_pos = [0.15, 0.15, 0.76, 0.76]; 

% 2. Core Plotting: Main Axis (h_ax)
h_ax = axes('Units', 'normalized', 'Position', ax_pos);
hold on;

% Plot the null distribution (Histogram): Muted blue-grey, semi-transparent, no edges
h_hist = histogram(perm_corrs, 50, 'Normalization', 'pdf');
h_hist.FaceColor = [0.6, 0.7, 0.8]; 
h_hist.EdgeColor = 'none'; 
h_hist.FaceAlpha = 0.6; 

% Plot the null model mean (Mathematical Artifact Baseline): Black dashed line
xline(baseline_corr, 'Color', [0, 0, 0], 'LineStyle', '--', 'LineWidth', 1.5);

% Plot the empirical observation (True Physical Trade-off): Solid red line
xline(obs_corr, 'Color', [0.9, 0.1, 0.1], 'LineStyle', '-', 'LineWidth', 1.8);

hold off;

% 3. Format Main Axis (h_ax)
set(h_ax, 'FontSize', font_sz_axis); 
set(h_ax, 'LineWidth', 1.5); 
set(h_ax, 'TickDir', 'out', 'TickLength', [0.02, 0.02]);
set(h_ax, 'XGrid', 'off', 'YGrid', 'off');

xtickformat(h_ax, '%.2f'); 

box off;

% 4. Create a transparent secondary axis
h_ax2 = axes('Units', 'normalized', 'Position', ax_pos);
box on;
set(h_ax2, 'Color', 'none', 'HitTest', 'off');
set(h_ax2, 'XTick', [], 'YTick', [], 'XTickLabel', [], 'YTickLabel', []);
set(h_ax2, 'LineWidth', 1.5);

% 5. Enforce strict 1:1 aspect ratio for both axes
pbaspect(h_ax, [1 1 1]);
pbaspect(h_ax2, [1 1 1]);

% 6. Synchronize axis limits
linkaxes([h_ax, h_ax2], 'xy');

% 7. Dynamically compute optimal axis limits
% Determine optimal X-axis limits based on data extrema
x_min = min([min(perm_corrs), obs_corr, baseline_corr]);
x_max = max([max(perm_corrs), obs_corr, baseline_corr]);
x_range = x_max - x_min;

% Apply a 10% buffer to the X-axis
xlim(h_ax, [x_min - 0.1 * x_range, x_max + 0.1 * x_range]);

% Determine optimal Y-axis limits based on histogram peak
y_max = max(h_hist.Values);
% Apply a 10% buffer to the top of the Y-axis, anchoring the bottom at 0
ylim(h_ax, [0, y_max * 1.1]);