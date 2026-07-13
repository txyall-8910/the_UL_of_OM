% =========================================================================
% Permutation-Based Test for Spatial Population Scaling Trade-off
% 
% Description:
% This script rigorously tests the robustness of the observed spatial trade-off 
% between central population capacity (\ln G_0) and radial dimension (\beta). 
% (Ref: Supplementary Text S3)
% To isolate the genuine physical trade-off from the parameterization artifact, 
% we implement a permutation test. By shuffling the scaling exponents (\beta) 
% while strictly preserving the empirical centers of mass for all spatial grids, 
% we generate a null distribution of spurious baseline correlations. The 
% statistical divergence of the empirical correlation from this null distribution 
% confirms the physical validity of the spatial resource allocation mechanisms.
% =========================================================================

% Extract fundamental variables representing the scaling parameters and centers of mass
beta_i     = T.a; % \beta_i: Empirical spatial scaling exponent (Slope)
log_r_mean = T.b; % \overline{\ln r_i}: Empirical center of mass
log_G_mean = T.c; % \overline{\ln G_i}: Empirical center of mass

% --- Data Cleansing ---
% Ensure computational integrity by removing any NaN values
valid_idx = ~isnan(beta_i) & ~isnan(log_r_mean) & ~isnan(log_G_mean);
beta_i     = beta_i(valid_idx);
log_r_mean = log_r_mean(valid_idx);
log_G_mean = log_G_mean(valid_idx);
n_samples  = length(beta_i);

% --- Step 1: Empirical Baseline Evaluation ---
% Calculate the true empirical intercept (\ln G_{0,i}) evaluated at the 
% standardized unit scale (\ln r = 0) using the empirical center of mass.
log_G0_true = log_G_mean - beta_i .* log_r_mean;

% Compute the empirically observed Pearson correlation coefficient
obs_corr = corr(log_G0_true, beta_i, 'Type', 'Pearson');
fprintf('Empirical r (Observed correlation between \\ln G_{0,i} and \\beta_i): %.4f\n', obs_corr);

% --- Steps 2 & 3: Permutation-Based Null Model Construction ---
num_permutations = 10000; % Number of iterations for robust statistical inference
perm_corrs = zeros(num_permutations, 1); % Preallocate array for null correlations

for i = 1:num_permutations
    % Step 2: Decouple the physical relationship
    % Randomly shuffle the scaling exponent (\beta_i) across the urban grids.
    % The centers of mass (\overline{\ln r_i}, \overline{\ln G_i}) 
    % remain fixed to preserve the intrinsic spatial scale and magnitude of each local system.
    shuffle_idx = randperm(n_samples);
    beta_null   = beta_i(shuffle_idx);
    
    % Step 3: Reconstruct the pseudo-intercepts (Mathematical Artifact)
    % Calculate the artifactual intercept driven purely by the geometric lever effect.
    log_G0_null = log_G_mean - beta_null .* log_r_mean;
    
    % Step 4: Compute the spurious baseline correlation for the current permutation
    perm_corrs(i) = corr(log_G0_null, beta_null, 'Type', 'Pearson');
end

% --- Step 5: Statistical Inference and Divergence Evaluation ---
% Calculate the baseline correlation driven entirely by the mechanical propagation 
% of the geometric lever effect (Mean of the null distribution).
baseline_corr = mean(perm_corrs);
fprintf('Null model r (Spurious baseline correlation driven by lever effect): %.4f\n', baseline_corr);

% Calculate the exact P-value (Adaptive one-tailed test)
% Dynamically determine the direction of divergence based on the empirical observation
if obs_corr < baseline_corr
    p_value = sum(perm_corrs <= obs_corr) / num_permutations;
    tail_str = 'Left-tailed';
else
    p_value = sum(perm_corrs >= obs_corr) / num_permutations;
    tail_str = 'Right-tailed';
end

fprintf('Permutation Test P-value (%s): %.6f\n', tail_str, p_value);

% Output rigorous scientific conclusions based on statistical significance
if p_value < 0.001
    fprintf('Conclusion: Highly Significant (P < 0.001).\n');
    fprintf('Although the geometric lever effect introduces a spurious baseline correlation (r \approx %.2f),\n', baseline_corr);
    fprintf('the empirically observed correlation (r = %.2f) statistically diverges from this mathematical artifact.\n', obs_corr);
    fprintf('This rigorously confirms that the trade-off between central capacity and spatial decay reflects genuine physical mechanisms.\n');
elseif p_value < 0.05
    fprintf('Conclusion: Significant (P < 0.05). The empirical trade-off robustly diverges from parameterization constraints.\n');
else
    fprintf('Conclusion: Not Significant (P \geq 0.05). The observed correlation cannot be distinguished from the geometric lever effect.\n');
end

% =========================================================================
% --- Visualization: Publication-Quality Null Distribution vs Empirical ---
% =========================================================================

% 1. Figure Setup
fig_size_cm = 9;
font_main = 'Helvetica'; 
font_sz_axis = 12;

h_fig = figure('Name', 'Permutation Null Model Evaluation', 'Units', 'centimeters');
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