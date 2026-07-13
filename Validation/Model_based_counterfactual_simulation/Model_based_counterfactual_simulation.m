% =========================================================================
% Model-based counterfactual testing 
% friction (d_i) as an example (Corresponding to Supplementary Text S8)
%
% Description:
% This script conducts model-based counterfactual simulations to evaluate 
% the efficacy of an analytical decision rule for targeted urban interventions. 
% It operationalizes the analytical sensitivity of the systemic depth-breadth 
% trade-off (\gamma) to local distance friction (d_i) using a local composite 
% index.
% =========================================================================

% =========================================================================
% ========================= 1. Parameter Setup & Initialization ===========
% =========================================================================
top_percent = 0.3;           % Proportion of nodes to intervene (e.g., 30%)
select_mode = 'total_ratio'; % Selection mode ('total_ratio' or 'pos_ratio')
num_random_trials = 1000;    % Number of Monte Carlo trials for random control
random_seed = 42;            % Fix random seed for full reproducibility

% 'VarName9'      : \lambda_i (Spatial reach)
% 'Mi'            : ln(K_i p_i \beta_i)   % Used to calculate theoretical lnA
% 'errors'        : \epsilon_i = lnA_obs - (Mi + (d_i/\beta_i)*lnG_i - ln\lambda_i)
% 'VarName12'     : \beta_i (Radial dimension)
% 'Intercept1'    : lnG_i (Central population capacity)
% 'd'             : d_i (Local distance friction)
% 'Intercept2'    : Observed lnA_i (Population-normalized core flow intensity)

lambda = T.VarName9;
lnA_obs = T.Intercept2;
beta = T.VarName12;
lnG = T.Intercept1;
errors = T.errors;
d = T.d;

N = height(T); % Total number of nodes in the system

% =========================================================================
% ========================= 2. Core Computation & Intervention Simulation =
% =========================================================================

% ----- 2.1 Baseline State (Original) -----
mdl_orig = fitlm(lambda, lnA_obs);
gamma_orig = -mdl_orig.Coefficients.Estimate(2);   
intercept_orig = mdl_orig.Coefficients.Estimate(1);
r2_orig = mdl_orig.Rsquared.Ordinary;
p_orig = mdl_orig.Coefficients.pValue(2);

% ----- 2.2 Identify Targeted Intervention Nodes (Based on analytical \Delta_i) -----
lambda_bar = mean(lambda);
lnA_bar = mean(lnA_obs);
sigma2_lambda = var(lambda, 1);  % Population variance (divided by N), for statistical reference

lambda_safe = max(lambda, 1e-6); 

Delta_i = (lnA_obs - lnA_bar) - (lambda - lambda_bar) .* ( (lnG ./ beta) + (1 ./ lambda_safe) - 2*gamma_orig );

% Benefit score under a proportional perturbation
% Interventions should prioritize locations with the highest positive sensitivity
Benefit_i = d .* Delta_i;

% Filter nodes (decreasing d_i effectively reduces systemic \gamma)
idx_pos = (Benefit_i > 0);                     
num_pos = sum(idx_pos);
Benefit_pos = Benefit_i(idx_pos);
[~, sort_idx] = sort(Benefit_pos, 'descend');   % Sort descending by benefit score to prioritize high-yield nodes

% Determine final intervention nodes based on the selected mode
switch select_mode
    case 'total_ratio'
        num_select_total = ceil(top_percent * N);
        num_select = min(num_select_total, num_pos);
        select_idx_in_pos = sort_idx(1:num_select);
        selected_nodes = find(idx_pos);
        selected_nodes = selected_nodes(select_idx_in_pos);
        fprintf('Mode: %.1f%% of total nodes, Target: %d, Selected: %d (Total positive nodes: %d)\n', ...
            top_percent*100, num_select_total, num_select, num_pos);
    case 'pos_ratio'
        num_select = ceil(top_percent * num_pos);
        select_idx_in_pos = sort_idx(1:num_select);
        selected_nodes = find(idx_pos);
        selected_nodes = selected_nodes(select_idx_in_pos);
        fprintf('Mode: %.1f%% of positive nodes, Selected: %d (Total positive nodes: %d)\n', ...
            top_percent*100, num_select, num_pos);
    otherwise
        error('Unknown select_mode. Please use ''total_ratio'' or ''pos_ratio''.');
end
num_target = length(selected_nodes);
fprintf('Final number of intervened nodes: %d (%.1f%% of total)\n', num_target, num_target/N*100);

% ----- 2.3 Global Intervention (Uniform 10% reduction of d_i across all N locations) -----
d_new_global = d * 0.9;
lambda_new_global = beta - d_new_global;
if any(lambda_new_global <= 0)
    warning('Global intervention: Some nodes have \lambda_new <= 0, physical meaning may be invalid.');
end
Qi_new_global = (d_new_global ./ beta) .* lnG;
lnA_sim_global = T.Mi + Qi_new_global - log(max(lambda_new_global, 1e-6)) + errors;
mdl_global = fitlm(lambda_new_global, lnA_sim_global);
gamma_global = -mdl_global.Coefficients.Estimate(2);
intercept_global = mdl_global.Coefficients.Estimate(1);
r2_global = mdl_global.Rsquared.Ordinary;
p_global = mdl_global.Coefficients.pValue(2);

% ----- 2.4 Targeted Intervention (10% reduction of d_i exclusively for selected nodes) -----
d_new_local = d;
lambda_new_local = lambda;
d_new_local(selected_nodes) = d(selected_nodes) * 0.9;
lambda_new_local(selected_nodes) = beta(selected_nodes) - d_new_local(selected_nodes);
if any(lambda_new_local(selected_nodes) <= 0)
    warning('Targeted intervention: Some selected nodes have \lambda_new <= 0.');
end
Qi_new_local = (d_new_local ./ beta) .* lnG;
lnA_sim_local = T.Mi + Qi_new_local - log(max(lambda_new_local, 1e-6)) + errors;
mdl_local = fitlm(lambda_new_local, lnA_sim_local);
gamma_local = -mdl_local.Coefficients.Estimate(2);
intercept_local = mdl_local.Coefficients.Estimate(1);
r2_local = mdl_local.Rsquared.Ordinary;
p_local = mdl_local.Coefficients.pValue(2);

% ----- 2.5 Random Monte Carlo Control (1,000 independent trials) -----
rng(random_seed); % Fix random seed to ensure reproducibility
gamma_random_list = zeros(num_random_trials, 1);
for i = 1:num_random_trials
    rand_idx = randperm(N, num_target);
    d_new_rand = d;
    d_new_rand(rand_idx) = d(rand_idx) * 0.9;
    lambda_new_rand = beta - d_new_rand;
    Qi_new_rand = (d_new_rand ./ beta) .* lnG;
    lnA_sim_rand = T.Mi + Qi_new_rand - log(max(lambda_new_rand, 1e-6)) + errors;
    mdl_rand = fitlm(lambda_new_rand, lnA_sim_rand);
    gamma_random_list(i) = -mdl_rand.Coefficients.Estimate(2);
end

% =========================================================================
% ========================= 3. Significance Testing (Console Output) ======
% =========================================================================
mean_gamma_rand = mean(gamma_random_list);
std_gamma_rand = std(gamma_random_list);

% 1. Calculate Z-score and theoretical p-value (left-tailed test)
z_score = (gamma_local - mean_gamma_rand) / std_gamma_rand;
p_normal = normcdf(z_score); 

% 2. Calculate empirical p-value (left-tailed test based on exact percentile)
p_empirical = sum(gamma_random_list <= gamma_local) / num_random_trials;

if p_empirical == 0
    p_emp_display = sprintf('< %.3f', 1/num_random_trials);
else
    p_emp_display = sprintf('= %.4f', p_empirical);
end

fprintf('\n================ Significance Testing Results ================\n');
fprintf('Z-score (Standardized deviation): %.2f\n', z_score);
fprintf('Theoretical p-value (Parametric)  : %.2e\n', p_normal);
fprintf('Empirical p-value (Monte Carlo)   %s\n', p_emp_display);
fprintf('==============================================================\n\n');

% =========================================================================
% ========================= 4. Plotting ===================================
% =========================================================================

% Global plotting parameters
fig_size_cm = 9; 
font_main = 'Helvetica';
font_sz_axis = 12; 
ax_pos = [0.15, 0.15, 0.75, 0.75]; 

% Color definitions
color_orig_line    = [0.50, 0.50, 0.50]; 
color_orig_scatter = [0.25, 0.55, 0.35]; 
color_global       = [0.20, 0.40, 0.80]; 
color_target       = [0.90, 0.10, 0.10]; 
color_fitline      = [0.30, 0.30, 0.30]; 

% -------------------------------------------------------------------------
% Figure 1: Probability Density Plot - No axis labels
% -------------------------------------------------------------------------
h_fig1 = figure('Name', 'Fig 1: Probability Density', 'Units', 'centimeters', 'Position', [2, 10, fig_size_cm, fig_size_cm]);
set(h_fig1, 'Color', 'w', 'DefaultAxesFontName', font_main, 'DefaultTextFontName', font_main);

h_ax1 = axes('Units', 'normalized', 'Position', ax_pos); hold on;
h_hist = histogram(gamma_random_list, 40, 'Normalization', 'pdf', 'FaceColor', [0.6, 0.7, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.6);
xline(gamma_orig, 'Color', color_orig_line, 'LineStyle', ':', 'LineWidth', 2.0); % Use grey line for baseline
xline(gamma_global, 'Color', color_global, 'LineStyle', '--', 'LineWidth', 2.0);
xline(gamma_local, 'Color', color_target, 'LineStyle', '-', 'LineWidth', 2.0);
hold off;

set(h_ax1, 'FontSize', font_sz_axis, 'LineWidth', 1.5, 'TickDir', 'out', 'TickLength', [0.02, 0.02], 'XGrid', 'off', 'YGrid', 'off');
xtickformat(h_ax1, '%.3f'); box off;
h_ax1_overlay = axes('Units', 'normalized', 'Position', ax_pos); box on;
set(h_ax1_overlay, 'Color', 'none', 'HitTest', 'off', 'XTick', [], 'YTick', [], 'LineWidth', 1.5);
pbaspect(h_ax1, [1 1 1]); pbaspect(h_ax1_overlay, [1 1 1]); linkaxes([h_ax1, h_ax1_overlay], 'xy');

x_min1 = min([min(gamma_random_list), gamma_local, gamma_global, gamma_orig]);
x_max1 = max([max(gamma_random_list), gamma_local, gamma_global, gamma_orig]);
xlim(h_ax1, [x_min1 - 0.15*(x_max1-x_min1), x_max1 + 0.15*(x_max1-x_min1)]);
ylim(h_ax1, [0, max(h_hist.Values) * 1.15]);

% -------------------------------------------------------------------------
% Prepare unified axis limits for scatter plots
% -------------------------------------------------------------------------
all_x = [lambda; lambda_new_global; lambda_new_local];
all_y = [lnA_obs; lnA_sim_global; lnA_sim_local];
x_range = max(all_x) - min(all_x);
y_range = max(all_y) - min(all_y);
unified_xlim = [min(all_x) - 0.1*x_range, max(all_x) + 0.1*x_range];
unified_ylim = [min(all_y) - 0.1*y_range, max(all_y) + 0.1*y_range];

% Define scatter plot data structure (including R2 and p-value)
scatter_data = {
    'Original', lambda, lnA_obs, intercept_orig, gamma_orig, color_orig_scatter, r2_orig, p_orig;
    'Global', lambda_new_global, lnA_sim_global, intercept_global, gamma_global, color_global, r2_global, p_global;
    'Targeted', lambda_new_local, lnA_sim_local, intercept_local, gamma_local, color_target, r2_local, p_local
};

% -------------------------------------------------------------------------
% Figures 2-4: Scatter Plots (No axis labels, uniform dark grey fitted line, with stats)
% -------------------------------------------------------------------------
for k = 1:3
    name = scatter_data{k, 1};
    x_data = scatter_data{k, 2};
    y_data = scatter_data{k, 3};
    intercept = scatter_data{k, 4};
    slope = -scatter_data{k, 5}; 
    color = scatter_data{k, 6};
    r2_val = scatter_data{k, 7};
    p_val = scatter_data{k, 8};
    
    h_fig = figure('Name', ['Fig ', num2str(k+1), ': Scatter - ', name], 'Units', 'centimeters', 'Position', [2 + k*10, 10, fig_size_cm, fig_size_cm]);
    set(h_fig, 'Color', 'w', 'DefaultAxesFontName', font_main, 'DefaultTextFontName', font_main);
    
    h_ax = axes('Units', 'normalized', 'Position', ax_pos); hold on;
    
    % 1. Plot semi-transparent scatter points
    scatter(h_ax, x_data, y_data, 30, color, 'filled', 'MarkerFaceAlpha', 0.4, 'MarkerEdgeColor', 'none');
    
    % 2. Plot uniform dark grey fitted line
    x_fit = linspace(unified_xlim(1), unified_xlim(2), 100);
    y_fit = intercept + slope * x_fit;
    plot(h_ax, x_fit, y_fit, 'Color', color_fitline, 'LineWidth', 2.0); 
    
    hold off;
    
    % 3. Format main axes
    set(h_ax, 'FontSize', font_sz_axis, 'LineWidth', 1.5, 'TickDir', 'out', 'TickLength', [0.02, 0.02], 'XGrid', 'off', 'YGrid', 'off');
    box off;
    
    % 4. Overlay transparent axes to complete the bounding box
    h_ax_overlay = axes('Units', 'normalized', 'Position', ax_pos); box on;
    set(h_ax_overlay, 'Color', 'none', 'HitTest', 'off', 'XTick', [], 'YTick', [], 'LineWidth', 1.5);
    
    % 5. Enforce 1:1 aspect ratio and synchronize
    pbaspect(h_ax, [1 1 1]); pbaspect(h_ax_overlay, [1 1 1]);
    linkaxes([h_ax, h_ax_overlay], 'xy');
    
    % 6. Apply unified axis limits
    xlim(h_ax, unified_xlim);
    ylim(h_ax, unified_ylim);
    
    % 7. Add statistical info text inside the plot (bottom-left)
    if p_val < 0.001
        p_str = 'p < 0.001';
    else
        p_str = sprintf('p = %.3f', p_val);
    end
    stats_text = sprintf('\\gamma = %.3f\nR^2 = %.2f\n%s', -slope, r2_val, p_str);
    text(h_ax, 0.05, 0.12, stats_text, 'Units', 'normalized', 'FontSize', 10, 'Color', color_fitline, 'FontWeight', 'bold', 'Interpreter', 'tex');
end

% -------------------------------------------------------------------------
% Figure 5: Standalone horizontal legend (for Probability Density plot)
% -------------------------------------------------------------------------
h_fig5 = figure('Name', 'Fig 5: Standalone Legend', 'Units', 'centimeters', 'Position', [2, 2, 18, 4]);
set(h_fig5, 'Color', 'w', 'DefaultAxesFontName', font_main, 'DefaultTextFontName', font_main);
h_ax_leg = axes('Position', [0 0 1 1], 'Visible', 'off'); hold on;

% Create dummy objects to generate the legend
p1 = plot(h_ax_leg, NaN, NaN, 's', 'MarkerSize', 10, 'MarkerFaceColor', [0.6, 0.7, 0.8], 'MarkerEdgeColor', 'none');
p2 = plot(h_ax_leg, NaN, NaN, ':', 'Color', color_orig_line, 'LineWidth', 2.0);
p3 = plot(h_ax_leg, NaN, NaN, '--', 'Color', color_global, 'LineWidth', 2.0);
p4 = plot(h_ax_leg, NaN, NaN, '-', 'Color', color_target, 'LineWidth', 2.0);

% Generate horizontal legend
h_leg = legend([p1, p2, p3, p4], {'Random Trials', 'Original', 'Global', 'Targeted'}, ...
    'Orientation', 'horizontal', 'FontSize', 12, 'Box', 'off', 'Location', 'none');

% Force figure update to get the actual dimensions of the legend
drawnow;

% Calculate and set legend position to center it perfectly in the figure
leg_pos = h_leg.Position;
h_leg.Position = [0.5 - leg_pos(3)/2, 0.5 - leg_pos(4)/2, leg_pos(3), leg_pos(4)];

fprintf('\nAnalysis and plotting successfully completed.\n');