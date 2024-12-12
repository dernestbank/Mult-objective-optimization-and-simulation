%% improved_rxn_simulation.m
% Enhanced Multiobjective Optimization for PEM Electrolyser Catalyst Loading
% Balances cost (catalyst usage) and performance (overpotential) with improved visualization.

clear; clc; close all;

%% Parameters and Constants

% Physical constants
R = 8.314;              % J/(mol*K)
F = 96485;              % C/mol
T = 353;                % K
n = 2;                  % electrons transferred
alpha = 0.5;            % charge transfer coefficient

% Catalyst properties
rho_cat = 21.45;        % g/cm^3
S_cat = 50e4;           % cm^2_active/g
c_cat = 30;             % $/g
j0 = 1e-6;              % A/cm^2_active

% Mass transport properties
D = 2.5e-5;             % cm^2/s
tau = 2;                % dimensionless
C_bulk = 0.0555;        % mol/cm^3

% Operating conditions
A_cell = 100;           % cm^2
J = 1.0;                % A/cm^2

% Decision variable bounds
delta_min = 0.5e-4;     
delta_max = 5e-4;       
epsilon_min = 0.3;      
epsilon_max = 0.7;      

% Overpotential constraint
eta_max = 0.1;          % V

% Initial guess (midpoint of bounds)
x0 = [ (delta_min + delta_max)/2 , (epsilon_min + epsilon_max)/2 ];

lb = [delta_min, epsilon_min];
ub = [delta_max, epsilon_max];

%% Normalization Factors
L_max = rho_cat * delta_max * (1 - epsilon_min);
C_max = L_max * A_cell * c_cat;  
eta_norm = eta_max;              

%% Optimization Settings
options = optimoptions('fmincon','Display','iter','Algorithm','sqp');

%% Generate Pareto Front
% Increase number of points for smoother front
num_points = 50;
w1_values = linspace(0, 1, num_points);
w2_values = 1 - w1_values;

cost_values = zeros(num_points, 1);
overpotential_values = zeros(num_points, 1);
delta_values = zeros(num_points, 1);
epsilon_values = zeros(num_points, 1);

for i = 1:num_points
    w1 = w1_values(i);
    w2 = w2_values(i);

    fun = @(x) objective_function(x, w1, w2, R, T, alpha, n, F, j0, S_cat, J, ...
                                  c_cat, A_cell, rho_cat, D, tau, C_bulk, C_max, eta_norm);
    nonlcon = @(x) overpotential_constraint(x, R, T, alpha, n, F, j0, S_cat, J, ...
                                            c_cat, A_cell, rho_cat, D, tau, C_bulk, eta_max);

    [x_opt, ~] = fmincon(fun, x0, [], [], [], [], lb, ub, nonlcon, options);

    delta_opt = x_opt(1);
    epsilon_opt = x_opt(2);

    [C_opt, eta_total_opt] = calculate_cost_overpotential(x_opt, R, T, alpha, n, F, j0, ...
                                                          S_cat, J, c_cat, A_cell, rho_cat, D, tau, C_bulk);

    cost_values(i) = C_opt;
    overpotential_values(i) = eta_total_opt;
    delta_values(i) = delta_opt;
    epsilon_values(i) = epsilon_opt;
end

%% Visualization Enhancements
% Choose a color scheme and figure properties
set(0, 'DefaultAxesFontSize', 14, 'DefaultAxesFontWeight', 'bold', 'DefaultLineLineWidth', 2);
set(0, 'DefaultLineMarkerSize', 8);
set(0, 'DefaultFigureColor', 'w');
colormap(parula);

% Sort results by cost (for smoother line)
[sorted_cost, sort_idx] = sort(cost_values);
sorted_overpot = overpotential_values(sort_idx);
sorted_delta = delta_values(sort_idx);
sorted_epsilon = epsilon_values(sort_idx);

% 1. Pareto Front (Cost vs. Overpotential)
figure;
plot(sorted_cost, sorted_overpot, '-o', 'MarkerFaceColor','b');
xlabel('Cost ($)', 'FontSize',14);
ylabel('Overpotential (V)', 'FontSize',14);
title('Pareto Front for Catalyst Loading Optimization', 'FontSize',16);
grid on;
% Add annotation for a selected "balanced" point (e.g., the middle one)
mid_idx = round(num_points/2);
text(sorted_cost(mid_idx)*1.02, sorted_overpot(mid_idx), 'Balanced Solution \rightarrow', ...
     'HorizontalAlignment','left','FontSize',12);

% 2. Color-Coded Scatter (by Porosity)
figure;
scatter(sorted_cost, sorted_overpot, 70, sorted_epsilon, 'filled');
xlabel('Cost ($)', 'FontSize',14);
ylabel('Overpotential (V)', 'FontSize',14);
title('Pareto Front Colored by Porosity \epsilon', 'FontSize',16);
colorbar;
grid on;

% 3. 3D Visualization (Thickness as Third Dimension)
figure;
scatter3(sorted_cost, sorted_overpot, sorted_delta, 70, sorted_delta, 'filled');
xlabel('Cost ($)', 'FontSize',14);
ylabel('Overpotential (V)', 'FontSize',14);
zlabel('Thickness \delta (cm)', 'FontSize',14);
title('3D Visualization of Pareto Front (Thickness)', 'FontSize',16);
grid on;
rotate3d on;

% OPTIONAL: If you have target values, you can add reference lines:
% For example, if you have a cost target of $15:
% xline(15, '--r', 'Cost Target');

% If you want to highlight overpotential less than 0.18 V:
% yline(0.18, '--g', 'Overpotential Target');

%% Functions

function Z = objective_function(x, w1, w2, R, T, alpha, n, F, j0, S_cat, J, ...
                                c_cat, A_cell, rho_cat, D, tau, C_bulk, C_norm, eta_norm)
    [C, eta_total] = calculate_cost_overpotential(x, R, T, alpha, n, F, j0, S_cat, ...
                                                  J, c_cat, A_cell, rho_cat, D, tau, C_bulk);
    C_normalized = C / C_norm;
    eta_normalized = eta_total / eta_norm;
    Z = w1 * C_normalized + w2 * eta_normalized;
end

function [C, eta_total] = calculate_cost_overpotential(x, R, T, alpha, n, F, j0, ...
                                                       S_cat, J, c_cat, A_cell, rho_cat, D, tau, C_bulk)
    delta = x(1);
    epsilon = x(2);

    % Catalyst loading
    L = rho_cat * delta * (1 - epsilon);
    C = L * A_cell * c_cat;

    % Exchange current density
    J0 = j0 * L * S_cat;  
    if J0 <= 0
        J0 = 1e-14; 
    end

    % Activation overpotential
    eta_act = (R * T) / (alpha * n * F) * log(J / J0);

    % Effective diffusivity
    D_eff = D * (epsilon / tau);
    if D_eff <= 0
        D_eff = 1e-14; 
    end

    % Concentration overpotential
    C_surface = C_bulk - (J * delta) / (n * F * D_eff);
    if C_surface <= 0
        C_surface = 1e-10;
    end
    eta_conc = (R * T) / (n * F) * log(C_bulk / C_surface);

    eta_total = eta_act + eta_conc;
end

function [c, ceq] = overpotential_constraint(x, R, T, alpha, n, F, j0, S_cat, J, ...
                                              c_cat, A_cell, rho_cat, D, tau, C_bulk, eta_max)
    [~, eta_total] = calculate_cost_overpotential(x, R, T, alpha, n, F, j0, ...
                                                  S_cat, J, c_cat, A_cell, rho_cat, D, tau, C_bulk);
    c = eta_total - eta_max;
    ceq = [];
end
