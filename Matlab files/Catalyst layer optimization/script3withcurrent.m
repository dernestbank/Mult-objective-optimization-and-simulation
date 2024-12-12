%% customized_rxn_simulation_with_contours.m
% Multiobjective optimization for a PEM electrolyser with separate catalysts:
% - Anode: IrO2
% - Cathode: Pt
% Including multiple current densities and providing separate visualizations
% for anode, cathode, and total costs, as well as contour plots.

clear; clc; close all;

%% Parameters and Constants
R = 8.314;     % J/(mol*K)
F = 96485;     % C/mol
T = 353;       % K
n = 2;         % electrons transferred
alpha = 0.5;   % charge transfer coefficient

%% Catalyst and Electrode Properties
% Cathode (Pt)
rho_cat_cath = 21.45;      % g/cm^3 for Pt
S_cat_cath = 50e4;         % cm^2_active/g
c_cat_cath = 30;           % $/g

% Anode (IrO2)
rho_cat_an = 11.66;        % g/cm^3 for IrO2
S_cat_an = 20e4;           % cm^2_active/g (assumed)
c_cat_an = 50;             % $/g (assumed)

% Mass transport
D = 2.5e-5;    % cm^2/s
tau = 2;        % dimensionless
C_bulk = 0.0555; % mol/cm^3

% Cell area
A_cell = 100;   % cm^2

% Bounds
delta_min = 0.5e-4;
delta_max = 5e-4;
epsilon_min = 0.3;
epsilon_max = 0.7;

x0 = [(delta_min+delta_max)/2, (epsilon_min+epsilon_max)/2];
lb = [delta_min, epsilon_min];
ub = [delta_max, epsilon_max];

% Overpotential constraint
eta_max = 0.1;

% Current densities
J_values = linspace(0.5, 2.0, 5); 

% Normalization factors
L_max_cath = rho_cat_cath * delta_max * (1 - epsilon_min);
L_max_an   = rho_cat_an   * delta_max * (1 - epsilon_min);
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

% Also store anode and cathode costs separately
all_cath_cost_values = cell(length(J_values),1);
all_an_cost_values = cell(length(J_values),1);

% Matrices for contour plots: dimension (J, w1)
eta_matrix = zeros(length(J_values), num_points);
cost_matrix = zeros(length(J_values), num_points);

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

        % For contour matrix
        eta_matrix(j_idx, i) = eta_total_opt;
        cost_matrix(j_idx, i) = C_opt;
    end

    all_cost_values{j_idx} = cost_values;
    all_overpotential_values{j_idx} = overpotential_values;
    all_delta_values{j_idx} = delta_values;
    all_epsilon_values{j_idx} = epsilon_values;
    all_cath_cost_values{j_idx} = cath_cost_values;
    all_an_cost_values{j_idx} = an_cost_values;
end

%% Visualization
set(0, 'DefaultAxesFontSize', 14, 'DefaultAxesFontWeight', 'bold', 'DefaultLineLineWidth', 2);
set(0, 'DefaultLineMarkerSize', 8);
set(0, 'DefaultFigureColor', 'w');
colormap(parula);

% Plot Pareto fronts for different Js (Total cost vs Overpotential)
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

% Choose one J and show separate anode/cathode costs along Pareto front
chosen_idx = 3; % For example, J = J_values(3)
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

% Now plot Cathode, Anode, and Total costs separately as a function of w1
% (or sorted index)
figure;
hold on; grid on;
plot(sorted_cath_cost, sorted_overpot, '-s', 'MarkerFaceColor','b', 'DisplayName','Cathode Cost');
plot(sorted_an_cost, sorted_overpot, '-^', 'MarkerFaceColor','r', 'DisplayName','Anode Cost');
plot(sorted_cost, sorted_overpot, '-o', 'MarkerFaceColor','g', 'DisplayName','Total Cost');
xlabel('Cost ($)');
ylabel('Overpotential (V)');
title(sprintf('Anode vs. Cathode vs. Total Costs at J=%.2f A/cm^2', J_values(chosen_idx)));
legend('Location','best');

% 3D Visualization (Thickness)
sorted_delta = all_delta_values{chosen_idx}(sort_idx);
figure;
scatter3(sorted_cost, sorted_overpot, sorted_delta, 70, sorted_delta, 'filled');
xlabel('Total Cost ($)');
ylabel('Overpotential (V)');
zlabel('Thickness \delta (cm)');
title(sprintf('3D Visualization of Pareto Front (Thickness) at J=%.2f A/cm^2', J_values(chosen_idx)));
grid on;
rotate3d on;

% Contour Plots for Overpotential and Cost vs J and w1
% Create a mesh: x-axis = J_values, y-axis = w1_values
[J_grid, w1_grid] = meshgrid(J_values, w1_values); 
% Note: our data is in (j_idx,i) format where j_idx corresponds to J_values and i to w1_values
% We need to transpose because meshgrid & indexing: 
% Currently, eta_matrix(j_idx, i), cost_matrix(j_idx, i)
% w1_values(i), J_values(j_idx)
% Let's reorder to (i,j) for contourf
eta_mat_plot = eta_matrix';  % Now rows correspond to w1 and columns to J
cost_mat_plot = cost_matrix'; 

figure;
subplot(1,2,1);
contourf(J_values, w1_values, eta_mat_plot, 20, 'LineColor','none');
colorbar;
xlabel('Current Density (A/cm^2)');
ylabel('w_1 (Weight on Cost)');
title('Overpotential Contour');

subplot(1,2,2);
contourf(J_values, w1_values, cost_mat_plot, 20, 'LineColor','none');
colorbar;
xlabel('Current Density (A/cm^2)');
ylabel('w_1 (Weight on Cost)');
title('Cost Contour');

% 3D Contour (using contour3 or surf)
figure;
contour3(J_values, w1_values, eta_mat_plot, 20);
xlabel('Current Density (A/cm^2)');
ylabel('w_1 (Weight on Cost)');
zlabel('Overpotential (V)');
title('3D Contour of Overpotential');
grid on;

figure;
surf(J_values, w1_values, cost_mat_plot);
xlabel('Current Density (A/cm^2)');
ylabel('w_1 (Weight on Cost)');
zlabel('Cost ($)');
title('Surface Plot of Cost');
shading interp; colorbar; grid on; rotate3d on;


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

    % Exchange current densities (simplified model)
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
