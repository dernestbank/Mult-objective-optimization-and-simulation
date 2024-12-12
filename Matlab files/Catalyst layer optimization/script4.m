%% integrated_simulation_and_contour.m
% This code:
% 1) Performs the multiobjective optimization for a PEM electrolyser with separate catalysts (Pt for cathode, IrO2 for anode).
% 2) Varies current density (J) and weighting factors (w1, w2).
% 3) Visualizes Pareto fronts, anode/cathode contributions, and contour maps.
% 4) Finally, generates a contour plot of cost as a function of total overpotential and catalyst loading by a parameter sweep.

clear; clc; close all;

%% Parameters and Constants
R = 8.314;      % J/(mol*K)
F = 96485;      % C/mol
T = 353;        % K
n = 2;          % electrons transferred
alpha = 0.5;    % charge transfer coefficient

% Catalyst and Electrode Properties
% Cathode (Pt)
rho_cat_cath = 21.45;       % g/cm^3
S_cat_cath = 50e4;          % cm^2_active/g
c_cat_cath = 30;            % $/g

% Anode (IrO2)
rho_cat_an = 11.66;         % g/cm^3
S_cat_an = 20e4;            % cm^2_active/g (assumed)
c_cat_an = 50;              % $/g (assumed)

% Mass transport
D = 2.5e-5;     % cm^2/s
tau = 2;         % dimensionless
C_bulk = 0.0555; % mol/cm^3

% Cell area
A_cell = 100;    % cm^2

% Bounds for design variables
delta_min = 0.5e-4;
delta_max = 5e-4;
epsilon_min = 0.3;
epsilon_max = 0.7;

x0 = [(delta_min+delta_max)/2, (epsilon_min+epsilon_max)/2];
lb = [delta_min, epsilon_min];
ub = [delta_max, epsilon_max];

% Overpotential constraint
eta_max = 0.1;

% Current densities range
J_values = linspace(0.5, 2.0, 5);

% Normalization factors
L_max_cath = rho_cat_cath * delta_max * (1 - epsilon_min);
L_max_an   = rho_cat_an * delta_max * (1 - epsilon_min);
C_max = (L_max_cath * A_cell * c_cat_cath) + (L_max_an * A_cell * c_cat_an);
eta_norm = eta_max;

% Pareto front parameters
num_points = 30;
w1_values = linspace(0, 1, num_points);
w2_values = 1 - w1_values;

options = optimoptions('fmincon','Display','iter','Algorithm','sqp');

% Storage
all_cost_values = cell(length(J_values),1);
all_overpotential_values = cell(length(J_values),1);
all_delta_values = cell(length(J_values),1);
all_epsilon_values = cell(length(J_values),1);
all_cath_cost_values = cell(length(J_values),1);
all_an_cost_values = cell(length(J_values),1);

%% Optimization Loop over J
for j_idx = 1:length(J_values)
    J = J_values(j_idx);

    cost_values = zeros(num_points, 1);
    overpotential_values = zeros(num_points, 1);
    delta_values = zeros(num_points, 1);
    epsilon_values = zeros(num_points, 1);
    cath_cost_values = zeros(num_points, 1);
    an_cost_values = zeros(num_points, 1);

    for i = 1:num_points
        w1 = w1_values(i);
        w2 = w2_values(i);

        fun = @(x) objective_function(x, w1, w2, R, T, alpha, n, F, ...
                                      S_cat_cath, S_cat_an, J, c_cat_cath, c_cat_an, ...
                                      A_cell, rho_cat_cath, rho_cat_an, D, tau, C_bulk, C_max, eta_norm);
        nonlcon = @(x) overpotential_constraint(x, R, T, alpha, n, F, ...
                                                S_cat_cath, S_cat_an, J, ...
                                                c_cat_cath, c_cat_an, A_cell, ...
                                                rho_cat_cath, rho_cat_an, D, tau, C_bulk, eta_max);

        [x_opt, ~] = fmincon(fun, x0, [], [], [], [], lb, ub, nonlcon, options);

        delta_opt = x_opt(1);
        epsilon_opt = x_opt(2);

        [C_opt, eta_total_opt, C_cath, C_an] = calculate_cost_overpotential(x_opt, R, T, alpha, n, F, ...
                                                              S_cat_cath, S_cat_an, J, c_cat_cath, c_cat_an, ...
                                                              A_cell, rho_cat_cath, rho_cat_an, D, tau, C_bulk);

        cost_values(i) = C_opt;
        overpotential_values(i) = eta_total_opt;
        delta_values(i) = delta_opt;
        epsilon_values(i) = epsilon_opt;
        cath_cost_values(i) = C_cath;
        an_cost_values(i) = C_an;
    end

    all_cost_values{j_idx} = cost_values;
    all_overpotential_values{j_idx} = overpotential_values;
    all_delta_values{j_idx} = delta_values;
    all_epsilon_values{j_idx} = epsilon_values;
    all_cath_cost_values{j_idx} = cath_cost_values;
    all_an_cost_values{j_idx} = an_cost_values;
end

%% Visualization of Optimization Results
set(0, 'DefaultAxesFontSize', 14, 'DefaultAxesFontWeight', 'bold', 'DefaultLineLineWidth', 2);
set(0, 'DefaultLineMarkerSize', 8);
set(0, 'DefaultFigureColor', 'w');
colormap(parula);

% Plot Pareto fronts for different J (Total cost vs Overpotential)
figure;
hold on; grid on;
for j_idx = 1:length(J_values)
    [sorted_cost, sort_idx] = sort(all_cost_values{j_idx});
    sorted_overpot = all_overpotential_values{j_idx}(sort_idx);
    plot(sorted_cost, sorted_overpot, '-o', 'DisplayName', sprintf('J=%.2f A/cm^2', J_values(j_idx)));
end
xlabel('Total Cost ($)');
ylabel('Overpotential (V)');
title('Pareto Fronts for Various Current Densities (Total)');
legend('Location','best');

% Choose one J and show separate anode/cathode costs
chosen_idx = 3;
[sorted_cost, sort_idx] = sort(all_cost_values{chosen_idx});
sorted_overpot = all_overpotential_values{chosen_idx}(sort_idx);
sorted_cath_cost = all_cath_cost_values{chosen_idx}(sort_idx);
sorted_an_cost = all_an_cost_values{chosen_idx}(sort_idx);

figure;
plot(sorted_cost, sorted_overpot, '-o');
xlabel('Total Cost ($)');
ylabel('Overpotential (V)');
title(sprintf('Pareto Front (Total) at J=%.2f A/cm^2', J_values(chosen_idx)));
grid on;

figure;
hold on; grid on;
plot(sorted_cath_cost, sorted_overpot, '-s', 'MarkerFaceColor','b', 'DisplayName','Cathode Cost');
plot(sorted_an_cost, sorted_overpot, '-^', 'MarkerFaceColor','r', 'DisplayName','Anode Cost');
plot(sorted_cost, sorted_overpot, '-o', 'MarkerFaceColor','g', 'DisplayName','Total Cost');
xlabel('Cost ($)');
ylabel('Overpotential (V)');
title(sprintf('Anode vs. Cathode vs. Total Costs at J=%.2f A/cm^2', J_values(chosen_idx)));
legend('Location','best');

%% Parametric Sweep to Generate Contour Data
% We want a contour plot: x-axis = Overpotential, y-axis = Catalyst Loading, colormap = Cost.
% We'll define arrays for delta and epsilon, compute overpotential and cost for each, then plot.

N_points = 50;  % resolution of the grid
delta_range = linspace(delta_min, delta_max, N_points);
epsilon_range = linspace(epsilon_min, epsilon_max, N_points);

% We'll pick a fixed J for this contour, or we could choose a representative J
J_for_contour = 1.0; % for example

% Initialize matrices
overpotential_matrix = zeros(N_points, N_points);
loading_matrix = zeros(N_points, N_points);
cost_matrix = zeros(N_points, N_points);

for i = 1:N_points
    for j = 1:N_points
        delta_val = delta_range(i);
        epsilon_val = epsilon_range(j);

        % Compute loading
        L_cath = rho_cat_cath * delta_val * (1 - epsilon_val);
        % Catalyst loading per unit area for cost plot (we can just use L_cath for demonstration or total loading)
        % If you want total loading, consider L_total = L_cath + L_an, but user wants catalyst loading (y) axis:
        % We'll choose just L_cath here as a representative loading axis:
        L_total = L_cath; 

        % Compute overpotential & cost at these conditions
        [C_val, eta_val, ~, ~] = calculate_cost_overpotential([delta_val, epsilon_val], R, T, alpha, n, F, ...
                                                  S_cat_cath, S_cat_an, J_for_contour, c_cat_cath, c_cat_an, ...
                                                  A_cell, rho_cat_cath, rho_cat_an, D, tau, C_bulk);

        overpotential_matrix(i,j) = eta_val;
        loading_matrix(i,j) = L_total;
        cost_matrix(i,j) = C_val;
    end
end

% Now we have overpotential_matrix, loading_matrix, and cost_matrix.
% To create a contour plot with Overpotential on x-axis, Loading on y-axis:
% We should create a meshgrid from overpotential and loading arrays.
% However, we do not have a direct overpotential array since overpotential depends on delta, epsilon.
% Let's rearrange the data to get a monotonic grid in overpotential and loading.

% Sort data to form a grid of monotonic overpotential and loading
% One approach: since delta and epsilon define both overpotential and loading,
% we can choose an axis directly. The user wants total overpotential on x and catalyst loading on y.
% overpotential_matrix(i,j) gives eta at each delta(i), epsilon(j)
% loading_matrix(i,j) gives loading at each delta(i), epsilon(j)

% Problem: overpotential and loading are not strictly monotonic in delta and epsilon.
% A simpler approach:
% - We'll treat delta_range as defining rows and epsilon_range as defining columns.
% - We now have a 2D map of cost, but the axes are delta and epsilon.
%
% To strictly follow user request: "Plot for total overpotential (x) vs catalyst loading (y) vs cost (color)."
% That implies we need monotonic axes in overpotential and loading.
% We must pick one pair of design variables. Let's assume we can rearrange:
%
% Let's vectorize the matrices and use scatteredInterpolant to map onto a grid of (overpotential, loading).

OP_vals = overpotential_matrix(:);
Load_vals = loading_matrix(:);
Cost_vals = cost_matrix(:);

% Create a regular grid in terms of Overpotential (x) and Loading (y)
OP_min = min(OP_vals); OP_max = max(OP_vals);
Load_min = min(Load_vals); Load_max = max(Load_vals);

OP_lin = linspace(OP_min, OP_max, N_points);
Load_lin = linspace(Load_min, Load_max, N_points);
[OP_grid, Load_grid] = meshgrid(OP_lin, Load_lin);

% Interpolate cost onto (OP, Load) grid
Finterp = scatteredInterpolant(OP_vals, Load_vals, Cost_vals, 'natural', 'none');
Cost_grid = Finterp(OP_grid, Load_grid);

% Now we have OP_grid (x-axis), Load_grid (y-axis), and Cost_grid (color)

figure;
contourf(OP_grid, Load_grid, Cost_grid, 20, 'LineColor','none');
colorbar;
xlabel('Total Overpotential (V)');
ylabel('Catalyst Loading (g/cm^2)');
title(sprintf('Cost Contour at J=%.2f A/cm^2', J_for_contour));
colormap(parula);
grid on;

%% Functions

function Z = objective_function(x, w1, w2, R, T, alpha, n, F, ...
                                S_cat_cath, S_cat_an, J, c_cat_cath, c_cat_an, ...
                                A_cell, rho_cat_cath, rho_cat_an, D, tau, C_bulk, C_norm, eta_norm)
    [C, eta_total, ~, ~] = calculate_cost_overpotential(x, R, T, alpha, n, F, ...
                                                  S_cat_cath, S_cat_an, J, c_cat_cath, c_cat_an, ...
                                                  A_cell, rho_cat_cath, rho_cat_an, D, tau, C_bulk);
    C_normalized = C / C_norm;
    eta_normalized = eta_total / eta_norm;
    Z = w1 * C_normalized + w2 * eta_normalized;
end

function [C, eta_total, C_cath, C_an] = calculate_cost_overpotential(x, R, T, alpha, n, F, ...
                                                       S_cat_cath, S_cat_an, J, c_cat_cath, c_cat_an, ...
                                                       A_cell, rho_cat_cath, rho_cat_an, D, tau, C_bulk)
    delta = x(1);
    epsilon = x(2);

    % Loadings
    L_cath = rho_cat_cath * delta * (1 - epsilon);
    L_an   = rho_cat_an   * delta * (1 - epsilon);

    % Costs
    C_cath = L_cath * A_cell * c_cat_cath;
    C_an = L_an * A_cell * c_cat_an;
    C = C_cath + C_an;

    % Exchange current densities
    j0_base_cath = 1e-6;
    j0_cath = j0_base_cath * L_cath * S_cat_cath;
    j0_base_an = 5e-7;
    j0_an = j0_base_an * L_an * S_cat_an;

    J0_eff = sqrt(j0_cath * j0_an);
    if J0_eff <= 0
        J0_eff = 1e-14;
    end

    % Overpotentials
    eta_act = (R*T)/(alpha*n*F)*log(J/J0_eff);

    D_eff = D*(epsilon/tau);
    if D_eff <= 0
        D_eff = 1e-14;
    end

    C_surface = C_bulk - (J * delta)/(n*F*D_eff);
    if C_surface <= 0
        C_surface = 1e-10;
    end

    eta_conc = (R*T)/(n*F)*log(C_bulk/C_surface);
    eta_total = eta_act + eta_conc;
end

function [c, ceq] = overpotential_constraint(x, R, T, alpha, n, F, ...
                                              S_cat_cath, S_cat_an, J, ...
                                              c_cat_cath, c_cat_an, A_cell, ...
                                              rho_cat_cath, rho_cat_an, D, tau, C_bulk, eta_max)
    [~, eta_total, ~, ~] = calculate_cost_overpotential(x, R, T, alpha, n, F, ...
                                                  S_cat_cath, S_cat_an, J, c_cat_cath, c_cat_an, ...
                                                  A_cell, rho_cat_cath, rho_cat_an, D, tau, C_bulk);
    c = eta_total - eta_max;
    ceq = [];
end
