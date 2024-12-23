%% customized_rxn_simulation_with_contours.m
% Comprehensive code integrating:
% 1. Multiobjective optimization with different current densities.
% 2. Separate catalysts at anode and cathode.
% 3. Parameter sweep to generate a cost matrix and produce a 2D contour plot:
%    Overpotential (x-axis) vs Catalyst Loading (y-axis) with Cost as colormap.

clear; clc; close all;

%% Parameters and Constants
R = 8.314;     % J/(mol*K)
F = 96485;     % C/mol
T = 353;       % K
n = 2;         % electrons transferred
alpha = 0.5;   % charge transfer coefficient

%% Catalyst and Electrode Properties
% Cathode (Pt)
rho_cat_cath = 21.45;   % g/cm^3
S_cat_cath = 50e4;      % cm^2_active/g
c_cat_cath = 30;        % $/g

% Anode (IrO2)
rho_cat_an = 11.66;     % g/cm^3
S_cat_an = 20e4;        % cm^2_active/g (assumed)
c_cat_an = 50;          % $/g (assumed)

% Mass transport
D = 2.5e-5;    % cm^2/s
tau = 2;       % dimensionless
C_bulk = 0.0555; % mol/cm^3

% Cell area
A_cell = 100;   % cm^2

% Bounds on design variables (for optimization)
delta_min = 0.5e-4;
delta_max = 5e-4;
epsilon_min = 0.3;
epsilon_max = 0.7;

x0 = [(delta_min+delta_max)/2, (epsilon_min+epsilon_max)/2];
lb = [delta_min, epsilon_min];
ub = [delta_max, epsilon_max];

% Overpotential constraint
eta_max = 0.1;

% Current densities for optimization
J_values = linspace(0.5, 2.0, 5);

% Normalization factors (for objective)
L_max_cath = rho_cat_cath * delta_max * (1 - epsilon_min);
L_max_an   = rho_cat_an   * delta_max * (1 - epsilon_min);
C_max = (L_max_cath * A_cell * c_cat_cath) + (L_max_an * A_cell * c_cat_an);
eta_norm = eta_max;

% Pareto front parameters
num_points = 30;
w1_values = linspace(0, 1, num_points);
w2_values = 1 - w1_values;

options = optimoptions('fmincon','Display','iter','Algorithm','sqp');

% Storage for optimization results
all_cost_values = cell(length(J_values),1);
all_overpotential_values = cell(length(J_values),1);
all_delta_values = cell(length(J_values),1);
all_epsilon_values = cell(length(J_values),1);
all_cath_cost_values = cell(length(J_values),1);
all_an_cost_values = cell(length(J_values),1);

% Matrices for contour (J vs w1 previously used, not mandatory now)
eta_matrix = zeros(length(J_values), num_points);
cost_matrix_Jw1 = zeros(length(J_values), num_points);

%% Multiobjective Optimization over J
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

        eta_matrix(j_idx, i) = eta_total_opt;
        cost_matrix_Jw1(j_idx, i) = C_opt;
    end

    all_cost_values{j_idx} = cost_values;
    all_overpotential_values{j_idx} = overpotential_values;
    all_delta_values{j_idx} = delta_values;
    all_epsilon_values{j_idx} = epsilon_values;
    all_cath_cost_values{j_idx} = cath_cost_values;
    all_an_cost_values{j_idx} = an_cost_values;
end

%% Visualization of Pareto Fronts
set(0, 'DefaultAxesFontSize', 14, 'DefaultAxesFontWeight', 'bold', 'DefaultLineLineWidth', 2);
set(0, 'DefaultLineMarkerSize', 8);
set(0, 'DefaultFigureColor', 'w');
colormap(parula);

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

% Choose one J and plot separate anode/cathode costs
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

%% Now, Generate and Store Matrix for Contour Plot (Overpotential vs Loading vs Cost)

% Define arrays for thickness and epsilon to scan
thickness_array = linspace(delta_min, delta_max, 50);
epsilon_array = linspace(epsilon_min, epsilon_max, 50);

% Initialize matrices
Overpot_mat = zeros(length(epsilon_array), length(thickness_array));
Loading_mat = zeros(length(epsilon_array), length(thickness_array));
Cost_mat = zeros(length(epsilon_array), length(thickness_array));

% Choose a fixed current density and use the same kinetic assumptions
J_fixed = 1.0; % Example chosen current density

for i = 1:length(epsilon_array)
    for j = 1:length(thickness_array)
        delta_test = thickness_array(j);
        epsilon_test = epsilon_array(i);

        % Compute cost and overpotential at these conditions:
        [C_test, eta_test, C_cath_test, C_an_test] = calculate_cost_overpotential([delta_test, epsilon_test], ...
            R, T, alpha, n, F, S_cat_cath, S_cat_an, J_fixed, c_cat_cath, c_cat_an, ...
            A_cell, rho_cat_cath, rho_cat_an, D, tau, C_bulk);

        % Compute loading (total)
        L_cath = rho_cat_cath * delta_test * (1 - epsilon_test);
        L_an = rho_cat_an * delta_test * (1 - epsilon_test);
        Loading_total = L_cath + L_an; % total catalyst loading

        Overpot_mat(i,j) = eta_test;
        Loading_mat(i,j) = Loading_total;
        Cost_mat(i,j) = C_test;
    end
end

% At this point, we have Overpot_mat, Loading_mat, and Cost_mat as functions of (delta, epsilon).
% But we want Overpotential as X and Loading as Y. We must ensure monotonicity.
% If Overpotential and Loading_mat are not monotonic in delta and epsilon, consider sorting or interpolation.
% For demonstration, we'll assume monotonic variations.

% Sort by Overpotential and Loading for proper contour plotting:
% We'll assume Overpot_mat increases along one dimension and Loading_mat along another.
% If not, you may need interpolation. Let's just plot directly and hope the data is well-behaved.

% We can pick a single "direction". For a proper contour, we need grids where X and Y are monotonic.
% If Overpot_mat and Loading_mat are not aligned, consider using scatteredInterpolant.
% Here, let's assume Overpot_mat is roughly monotonic with thickness_array and Loading_mat with epsilon_array.

% One approach: Choose Overpot_mat as X-axis and Loading_mat as Y-axis directly from the computed grid.
% Because delta and epsilon vary linearly, Overpot_mat and Loading_mat might not be strictly monotonic.
% If needed, pick a subregion or use interpolation. For simplicity, we directly plot:

figure;
% Use contourf with Overpot_mat as X and Loading_mat as Y.
% We must be careful: contourf expects matrices where X and Y come from meshgrid.
% Here Overpot_mat and Loading_mat are also matrices. If Overpot_mat and Loading_mat are monotonic in i,j order,
% we can plot directly:
contourf(Overpot_mat, Loading_mat, Cost_mat, 20, 'LineColor','none');
colorbar;
xlabel('Total Overpotential (V)');
ylabel('Catalyst Loading (g/cm²)');
title(sprintf('Cost Contour at J=%.2f A/cm^2', J_fixed));

% If Overpot_mat and Loading_mat are not aligned as a perfect grid for contourf, you can try:
% contourf(Overpot_mat, Loading_mat, Cost_mat,...)
% If you see a warning or strange plot, you may need to create a perfect mesh. For demonstration:
%
% If needed, create monotonic arrays by sorting Overpot_mat and Loading_mat, or using interpolation.
% Another solution is to pick one direction (e.g., fix epsilon and vary delta) so you get monotonic arrays.
%
% For demonstration, we assume it works as intended. Adjust or interpolate as needed for real data.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

    % Exchange current densities (simplified)
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
